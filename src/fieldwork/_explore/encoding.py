"""Column labels, native value codes, and the plain JSON form of emitted values.

Analyses work on ``pd.factorize`` codes. A column's dictionary lists its distinct
values as Python scalars in canonical order (booleans, integers, floats, strings,
dates, naive datetimes, aware datetimes as UTC instants, timedeltas), with missing
values collapsed into one final ``None`` entry. Code order is therefore value
order, whatever the row order or dtype.

Identity rules: native missing spellings (None, NaN, NaT, pd.NA) are one missing
value; booleans, numbers and strings never match each other; integers and floats
stay distinct (``1`` and ``1.0`` are different levels of an object column);
aware datetimes compare as instants. Only values that are emitted are converted
to JSON: numbers stay numbers, infinities become ``"inf"``/``"-inf"``, temporal
values become ISO text (timedeltas use pandas' own text, which it parses back).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

MISSING = None

# Object columns whose inferred kind cannot mix booleans, integers and floats can
# be factorized directly; pandas would otherwise merge True, 1 and 1.0.
_SORTABLE = {"string", "integer", "floating", "boolean", "empty"}
_TEMPORAL = {"datetime", "datetime64", "date", "timedelta", "timedelta64"}


class MissingCode:
    """Missing-code metadata when an FD kernel only needs equivalence classes.

    Grain never displays cell values, so discovery can reuse integer groups
    without retaining or rebuilding every high-cardinality dictionary.
    """

    def __init__(self, code: int | None):
        self.code = code


def missing_code(values: list[Any] | MissingCode) -> int | None:
    if isinstance(values, MissingCode):
        return values.code
    return len(values) - 1 if values and values[-1] is None else None


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def python_value(value: Any) -> Any:
    """Canonical Python scalar for a cell, or None when it is missing.

    Raises TypeError for unsupported cells such as lists, dicts or Decimal.
    """
    kind = type(value)
    if kind is str or kind is int or kind is bool:
        return value
    if kind is float:
        return None if math.isnan(value) else value + 0.0  # -0.0 and 0.0 are one value
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) + 0.0  # -0.0 and 0.0 are one value
    if isinstance(value, str):
        return value
    if isinstance(value, (datetime, np.datetime64)):
        stamp = pd.Timestamp(value)
        return stamp.tz_convert("UTC") if stamp.tzinfo is not None else stamp
    if isinstance(value, date):
        return value
    if isinstance(value, (timedelta, np.timedelta64)):
        return pd.Timedelta(value)
    # The type, never the value: exception messages must not echo source cells.
    raise TypeError(f"Unsupported value of type {type(value).__name__}")


def value_key(value: Any) -> tuple[int, Any]:
    """Identity and canonical sort key of a Python value from python_value."""
    if isinstance(value, bool):
        return (0, value)
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, float):
        return (2, value)
    if isinstance(value, str):
        return (3, value)
    if isinstance(value, pd.Timestamp):
        return (6 if value.tzinfo is not None else 5, value)
    if isinstance(value, date):
        return (4, value)
    return (7, value)


def json_value(value: Any) -> Any:
    """Plain JSON form of a Python value from python_value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    return str(value)


def cell(frame: pd.DataFrame, column: str, row: int, present: bool = True) -> Any:
    """JSON value of one source cell; absent cells (including sentinels) are None."""
    return json_value(python_value(frame[column].iloc[row])) if present else None


def same_json(left: Any, right: Any) -> bool:
    """Equality of exported values that keeps True, 1 and 1.0 apart."""
    return type(left) is type(right) and left == right


def code_of(values: list[Any], value: Any) -> int | None:
    """Dictionary code of a value, or None when it is not observed.

    Values match by exported form, so a saved value (a timestamp saved as ISO
    text, say) finds the code of the original cell.
    """
    wanted = json_value(python_value(value))
    if wanted is None:
        return missing_code(values)
    return next(
        (i for i, v in enumerate(values) if v is not None and same_json(json_value(v), wanted)),
        None,
    )


_LOOKALIKES = {"true", "false", "none", "null", "na", "n/a", "nan", "nat", "<na>"}


def _ambiguous(text: str) -> bool:
    """Whether a string could be read as a number, boolean or missing value."""
    if not text or text != text.strip() or text[0] in "'\"" or text.lower() in _LOOKALIKES:
        return True
    try:
        float(text)
    except ValueError:
        return False
    return True


def display(value: Any, missing_label: str = "<NA>") -> str:
    """Readable text for an exported value.

    Strings are quoted only when they could be mistaken for a number, boolean or
    missing value, so ``1``, ``1.0``, ``'1'`` and ``True`` stay distinguishable.
    """
    if value is None:
        return missing_label
    if isinstance(value, str):
        label = repr(value) if _ambiguous(value) or value == missing_label else value
    else:
        label = repr(value) if isinstance(value, float) else str(value)
    return f"value({label})" if label == missing_label else label


def json_order(value: Any) -> tuple[int, Any]:
    """Canonical order of exported values, for structure-only presentations."""
    if value is None:
        return (9, 0)
    if isinstance(value, bool):
        return (0, value)
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, float):
        return (2, value)
    return (3, str(value))


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    """The frame with columns named by str(label); names must stay unique.

    Non-string labels (for example integers from a headerless CSV) are analyzed
    and reported under their string form. The source is never modified.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    names = [c if isinstance(c, str) else str(c) for c in df.columns]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Duplicate column labels are not supported: {duplicates}")
    if all(isinstance(c, str) for c in df.columns):
        return df
    from .._runtime import current_session

    session = current_session()
    cached = session.labelled.get(id(df)) if session else None
    if cached is not None and cached[0] is df:
        return cached[1]
    output = df.copy(deep=False)
    output.columns = names
    if session:
        session.labelled[id(df)] = (df, output)
    return output


def resolve_columns(
    df: pd.DataFrame,
    columns: Iterable[Any] | None,
    *,
    argument: str,
    default_all: bool = False,
) -> tuple[str, ...]:
    """Requested columns of a labelled frame, named by str(label)."""
    if columns is None:
        if default_all:
            return tuple(df.columns)
        raise ValueError(f"{argument} is required")
    selected = tuple(c if isinstance(c, str) else str(c) for c in columns)
    if not selected:
        raise ValueError(f"{argument} must contain at least one column")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{argument} contains repeated columns")
    unknown = [c for c in selected if c not in df.columns]
    if unknown:
        raise KeyError(
            f"Unknown columns in {argument}: {unknown!r}; available columns: "
            f"{df.columns.tolist()!r}"
        )
    return selected


def _canonical(codes: np.ndarray, uniques: list[Any]) -> tuple[list[Any], np.ndarray]:
    """Sort a first-observed dictionary canonically and append the missing value."""
    keyed = [(value_key(v), i) for i, v in enumerate(uniques)]
    order = [i for _, i in sorted(keyed, key=lambda item: item[0])]
    remap = np.empty(len(order) + 1, dtype=np.int64)
    remap[order] = np.arange(len(order))
    remap[len(order)] = -1  # code -1 (missing) stays missing
    return _with_missing([uniques[i] for i in order], remap[codes])


def _with_missing(values: list[Any], codes: np.ndarray) -> tuple[list[Any], np.ndarray]:
    codes = np.asarray(codes, dtype=np.int64)
    missing = codes < 0
    if missing.any():
        codes = np.where(missing, len(values), codes)
        values = [*values, None]
    return values, codes


def _python_values(uniques: Any) -> list[Any]:
    """Python scalars for factorized uniques, vectorized for numeric and temporal dtypes."""
    if isinstance(uniques, pd.DatetimeIndex) and uniques.tz is not None:
        return list(uniques.tz_convert("UTC"))
    if isinstance(uniques, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return list(uniques)
    dtype = getattr(uniques, "dtype", None)
    if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
        array = np.asarray(uniques)
        return (array + 0.0 if dtype.kind == "f" else array).tolist()
    return [python_value(v) for v in uniques]


def _object_kind(series: pd.Series) -> str:
    return pd.api.types.infer_dtype(series.to_numpy(dtype=object, copy=False), skipna=True)


def encode_series(series: pd.Series) -> tuple[list[Any], np.ndarray]:
    """Dictionary of canonical values and one int64 code per row."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        if not len(dtype.categories):
            return _with_missing([], np.full(len(series), -1, dtype=np.int64))
        categories, category_codes = encode_series(pd.Series(dtype.categories))
        cat_codes = series.cat.codes.to_numpy()
        codes = np.where(cat_codes >= 0, category_codes[np.maximum(cat_codes, 0)], -1)
        used = np.unique(codes[codes >= 0])
        remap = np.full(len(categories) + 1, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))  # unobserved categories are not levels
        return _with_missing([categories[i] for i in used], remap[codes])
    kind = _object_kind(series) if dtype == object else None
    if kind is None or kind in _SORTABLE:
        codes, uniques = pd.factorize(series, sort=True)
        return _with_missing(_python_values(uniques), codes)
    if kind in _TEMPORAL:
        codes, uniques = pd.factorize(series, sort=False)
        return _canonical(codes, _python_values(uniques))
    values = [python_value(v) for v in series.array]
    lookup: dict[tuple[int, Any], int] = {}
    uniques: list[Any] = []
    codes = np.empty(len(values), dtype=np.int64)
    for row, value in enumerate(values):
        if value is None:
            codes[row] = -1
            continue
        key = value_key(value)
        code = lookup.get(key)
        if code is None:
            code = lookup[key] = len(uniques)
            uniques.append(value)
        codes[row] = code
    return _canonical(codes, uniques)


def encode_column(df: pd.DataFrame, column: str) -> tuple[list[Any], np.ndarray]:
    """Encode one column of a frame; see encode_series."""
    return encode_series(df[column])


def validate_limit(name: str, value: int | None, *, zero: bool = True) -> None:
    if value is None:
        return
    minimum = 0 if zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if zero else "positive"
        raise ValueError(f"{name} must be {qualifier} integer or None")


def validate_schema(df: pd.DataFrame, schema: dict[Any, str] | None) -> dict[str, str]:
    """Advisory roles keyed by column name."""
    if schema is None:
        return {}
    if not isinstance(schema, dict):
        raise TypeError("schema must be a column-to-role mapping")
    allowed = {"id", "categorical", "continuous", "unknown"}
    roles = {}
    for column, role in schema.items():
        name = column if isinstance(column, str) else str(column)
        if name not in df.columns:
            raise KeyError(f"Unknown schema column {column!r}")
        if role not in allowed:
            raise ValueError(
                f"Unsupported schema role {role!r}; expected one of {sorted(allowed)!r}"
            )
        roles[name] = role
    return roles
