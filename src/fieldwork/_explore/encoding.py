"""Validation and canonical, typed scalar identities for the explorer."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

TYPE_ORDER = {
    "boolean": 0,
    "integer": 1,
    "float": 2,
    "string": 3,
    "date": 4,
    "datetime_naive": 5,
    "datetime_aware": 6,
    "timedelta": 7,
    "tuple": 8,
    "missing": 9,
}


@dataclass(frozen=True)
class ScalarIdentity:
    """Hashable identity that does not inherit Python's mixed-number equality."""

    kind: str
    value: Any = None
    metadata: tuple[tuple[str, str], ...] = ()

    def sort_key(self) -> tuple[Any, ...]:
        if self.kind == "float":
            if self.value == "-inf":
                value = (0, 0)
            elif self.value == "inf":
                value = (2, 0)
            elif self.value == "nan":
                value = (3, 0)
            else:
                value = (1, float.fromhex(self.value))
        elif self.kind in {"integer", "timedelta"}:
            value = int(self.value)
        elif self.kind == "tuple":
            value = tuple(item.sort_key() for item in self.value)
        else:
            value = self.value
        return (TYPE_ORDER[self.kind], value, self.metadata)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.kind}
        if self.kind != "missing":
            if self.kind == "tuple":
                out["value"] = [item.to_dict() for item in self.value]
            else:
                out["value"] = self.value
        out.update(dict(self.metadata))
        return out


MISSING = ScalarIdentity("missing")


@dataclass(frozen=True)
class MissingCode:
    """Missing-code metadata when an FD kernel only needs equivalence classes.

    Grain never displays cell values, so discovery can reuse integer groups
    without retaining or rebuilding every high-cardinality scalar dictionary.
    """

    code: int | None


def missing_code(values):
    if isinstance(values, MissingCode):
        return values.code
    # Canonical dictionaries sort MISSING last. Avoid a linear dictionary scan
    # for each candidate/target pair on continuous or unique-ID columns.
    return len(values) - 1 if values and values[-1] == MISSING else None


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def normalize_scalar(value: Any, *, label: bool = False) -> ScalarIdentity:
    """Return the v0 canonical identity for a supported scalar."""

    if label:
        if isinstance(value, tuple):
            return ScalarIdentity(
                "tuple", tuple(normalize_scalar(item, label=True) for item in value)
            )
        if isinstance(value, str):
            return ScalarIdentity("string", value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return ScalarIdentity("integer", str(int(value)))
        raise TypeError(
            f"Unsupported column label {value!r} of type {type(value).__name__}; "
            "use strings, integers, or recursively tuple-valued labels"
        )
    if _is_missing(value):
        return MISSING
    if isinstance(value, (bool, np.bool_)):
        return ScalarIdentity("boolean", bool(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return ScalarIdentity("integer", str(int(value)))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return MISSING
        if math.isinf(number):
            return ScalarIdentity("float", "inf" if number > 0 else "-inf")
        if number == 0:
            number = 0.0
        return ScalarIdentity("float", number.hex())
    if isinstance(value, str):
        return ScalarIdentity("string", value)
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            utc = value.tz_convert("UTC")
            return ScalarIdentity(
                "datetime_aware",
                utc.isoformat(),
                (("resolution", "nanosecond"),),
            )
        return ScalarIdentity("datetime_naive", value.isoformat(), (("resolution", "nanosecond"),))
    if isinstance(value, np.datetime64):
        return normalize_scalar(pd.Timestamp(value))
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            utc = value.astimezone(UTC)
            return ScalarIdentity(
                "datetime_aware",
                utc.isoformat(),
                (("resolution", "nanosecond"),),
            )
        return ScalarIdentity("datetime_naive", value.isoformat(), (("resolution", "nanosecond"),))
    if isinstance(value, date):
        return ScalarIdentity("date", value.isoformat())
    if isinstance(value, pd.Timedelta):
        return ScalarIdentity("timedelta", str(value.value), (("resolution", "nanosecond"),))
    if isinstance(value, np.timedelta64):
        return normalize_scalar(pd.Timedelta(value))
    if isinstance(value, timedelta):
        microseconds = (value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds
        return ScalarIdentity(
            "timedelta", str(microseconds * 1_000), (("resolution", "nanosecond"),)
        )
    raise TypeError(
        f"Unsupported {'column label' if label else 'value'} {value!r} "
        f"of type {type(value).__name__}"
    )


def display_scalar(value: ScalarIdentity, missing_label: str = "<NA>") -> str:
    """Display typed values without conflating strings with numbers or missingness."""

    if value.kind == "missing":
        return missing_label
    if value.kind == "string":
        label = repr(value.value)
    elif value.kind == "integer":
        label = value.value
    elif value.kind == "float":
        label = value.value if value.value in {"inf", "-inf"} else repr(float.fromhex(value.value))
    elif value.kind == "boolean":
        label = "True" if value.value else "False"
    elif value.kind == "tuple":
        items = ", ".join(display_scalar(v, missing_label) for v in value.value)
        label = "(" + items + ("," if len(value.value) == 1 else "") + ")"
    elif value.kind == "timedelta":
        nanoseconds = int(value.value)
        label = ("-" if nanoseconds < 0 else "") + str(pd.Timedelta(abs(nanoseconds), unit="ns"))
    else:
        label = str(value.value)
    return f"{value.kind}({label})" if label == missing_label else label


def validate_frame(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if df.columns.has_duplicates:
        duplicates = [repr(c) for c in df.columns[df.columns.duplicated()].tolist()]
        raise ValueError(f"Duplicate column labels are not supported: {duplicates}")
    for column in df.columns:
        normalize_scalar(column, label=True)


def resolve_columns(
    df: pd.DataFrame,
    columns: Iterable[Any] | None,
    *,
    argument: str,
    default_all: bool = False,
) -> tuple[Any, ...]:
    validate_frame(df)
    if columns is None:
        if default_all:
            return tuple(df.columns.tolist())
        raise ValueError(f"{argument} is required")
    selected = tuple(columns)
    if not selected:
        raise ValueError(f"{argument} must contain at least one column")
    tokens = [normalize_scalar(c, label=True) for c in selected]
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{argument} contains repeated columns")
    available = {normalize_scalar(c, label=True): c for c in df.columns}
    unknown = [c for c, token in zip(selected, tokens) if token not in available]
    if unknown:
        raise KeyError(
            f"Unknown columns in {argument}: {unknown!r}; available columns: "
            f"{df.columns.tolist()!r}"
        )
    return tuple(available[token] for token in tokens)


def encode_column(df: pd.DataFrame, column: Any) -> tuple[list[ScalarIdentity], np.ndarray]:
    """Encode one column once per analysis call, however many components need it."""
    from .._runtime import current_session

    session = current_session()
    key = (id(df), column)
    cached = session.encodings.get(key) if session else None
    if cached is not None and cached[0] is df:
        return cached[1], cached[2]
    values, codes = encode_series(df[column])
    if session:
        session.remember_encoding(key, (df, values, codes))
    return values, codes


def encode_series(series: pd.Series) -> tuple[list[ScalarIdentity], np.ndarray]:
    """Encode a series with deterministic dictionary ordering."""

    inferred = pd.api.types.infer_dtype(series.array, skipna=True)
    if inferred == "unknown-array":
        # Some pandas versions do not inspect object NumpyExtensionArray values.
        # Inspect their array, not an assumed dtype: mixed bool/int/float values
        # must still take the typed fallback below.
        inferred = pd.api.types.infer_dtype(series.to_numpy(copy=False), skipna=True)
    homogeneous = {
        "empty",
        "string",
        "bytes",
        "boolean",
        "integer",
        "floating",
        "datetime",
        "datetime64",
        "date",
        "timedelta",
        "timedelta64",
    }
    if inferred in homogeneous and inferred != "bytes":
        raw_codes, uniques = pd.factorize(series, sort=False, use_na_sentinel=False)
        unique_tokens = [normalize_scalar(value) for value in uniques]
        values = sorted(set(unique_tokens), key=ScalarIdentity.sort_key)
        canonical = {value: index for index, value in enumerate(values)}
        remap = np.fromiter((canonical[value] for value in unique_tokens), dtype=np.int64)
        if raw_codes.size:
            return values, remap[raw_codes]
        return values, np.empty(0, dtype=np.int64)
    normalized = [normalize_scalar(value) for value in series.array]
    values = sorted(set(normalized), key=ScalarIdentity.sort_key)
    lookup = {value: i for i, value in enumerate(values)}
    codes = np.fromiter((lookup[value] for value in normalized), dtype=np.int64)
    return values, codes


def validate_limit(name: str, value: int | None, *, zero: bool = True) -> None:
    if value is None:
        return
    minimum = 0 if zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if zero else "positive"
        raise ValueError(f"{name} must be {qualifier} integer or None")


def validate_schema(df: pd.DataFrame, schema: dict[Any, str] | None) -> None:
    if schema is None:
        return
    if not isinstance(schema, dict):
        raise TypeError("schema must be a column-to-role mapping")
    available = {normalize_scalar(column, label=True) for column in df.columns}
    allowed = {"id", "categorical", "continuous", "unknown"}
    for column, role in schema.items():
        if normalize_scalar(column, label=True) not in available:
            raise KeyError(f"Unknown schema column {column!r}")
        if role not in allowed:
            raise ValueError(
                f"Unsupported schema role {role!r}; expected one of {sorted(allowed)!r}"
            )
