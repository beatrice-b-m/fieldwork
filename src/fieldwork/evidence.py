"""Portable findings, position-based inspection, and reusable population scopes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Unpack

import numpy as np
import pandas as pd

from ._explore.encoding import (
    display,
    encode_column,
    json_value,
    labelled,
    python_value,
    resolve_columns,
)
from ._runtime import checkpoint, current_session, operation, phase
from .result import Result
from .typing import Runtime


def fingerprint(df: pd.DataFrame) -> str:
    """Identify ordered source values and labels, including duplicate indexes."""
    labelled(df)
    session = current_session()
    if session and id(df) in session.fingerprints:
        checkpoint()
        return session.fingerprints[id(df)][1]
    with phase("fingerprinting", len(df.columns), "columns") as tracker:
        identity = _fingerprint(df, tracker)
    if session:
        session.fingerprints[id(df)] = (df, identity)
    return identity


def _fingerprint(df, progress):
    # SHA-256 over column labels, then vectorized per-value hashes of the index
    # and each column. Dtype is part of the identity: an object column holding
    # 1 differs from an int64 column holding 1.
    digest = hashlib.sha256()
    labels = [f"{type(c).__qualname__}:{c!r}" for c in df.columns]
    digest.update(json.dumps([len(df), labels]).encode())
    digest.update(_value_hashes(df.index))
    for column in df:
        checkpoint()
        digest.update(_value_hashes(df[column]))
        progress.advance(detail=str(column))
    return digest.hexdigest()


def _value_hashes(values):
    if isinstance(values, pd.MultiIndex):
        return pd.util.hash_pandas_object(values).to_numpy().tobytes()
    if values.dtype == object:
        # pandas hashes object values by str(), which conflates 1, 1.0 and "1"
        # and rejects lists; hash a typed representation instead.
        values = pd.Index([f"{type(v).__qualname__}:{v!r}" for v in values], dtype=object)
    return pd.util.hash_pandas_object(pd.Index(values)).to_numpy().tobytes()


@dataclass(frozen=True)
class Scope:
    """Identify a reusable population by positions in one ordered source frame.

    Parameters
    ----------
    dataset_id : str
        Canonical source fingerprint. Prefer from_positions to compute it.
    positions : tuple of int
        Unique nonnegative source row positions, sorted into source order.
        from_positions additionally validates bounds against the dataframe.
    name : str, optional
        Population label; default 'selection'.
    parent : str or None, optional
        Parent scope name recording lineage; default None.

    Attributes
    ----------
    dataset_id : str
        Identity of ordered column labels, index labels, and cell values.
    positions : tuple[int, ...]
        Absolute source positions, never dataframe index labels or offsets within
        a parent selection. Duplicate dataframe indexes are therefore safe.
    name : str
        Displayed population label.
    parent : str or None
        Parent scope name, when refined or selected from saved evidence.

    Raises
    ------
    ValueError
        Positions repeat or are negative/noninteger (booleans are invalid).

    Notes
    -----
    Scopes are frozen and source-bound. Reordering or changing values, labels
    or column dtypes invalidates reuse. Scopes store
    positions, not source cells. Search/display budgets do not modify a scope.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({'x': [10, 20, 30]}, index=['a', 'a', 'b'])
    >>> selected = fw.Scope.from_positions(df, [2, 0], name='eligible')
    >>> selected.positions
    (0, 2)
    >>> selected.refine(df, [2]).parent
    'eligible'
    """

    dataset_id: str
    positions: tuple[int, ...]
    name: str = "selection"
    parent: str | None = None

    def __post_init__(self):
        positions = tuple(self.positions)
        if any(
            isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 0 for p in positions
        ):
            raise ValueError("Scope positions must be nonnegative integers")
        if len(set(positions)) != len(positions):
            raise ValueError("Scope positions must not repeat")
        object.__setattr__(self, "positions", tuple(sorted(int(p) for p in positions)))

    @classmethod
    @operation("scope selection")
    def from_positions(
        cls,
        df: pd.DataFrame,
        positions: Iterable[int | np.integer[Any]],
        *,
        name: str = "selection",
        **runtime: Unpack[Runtime],
    ) -> Scope:
        """Create a source-bound scope from absolute row positions.

        Parameters
        ----------
        df : pandas.DataFrame
            Source frame, read without mutation. Labels may be unique strings, integers,
            or recursively nested tuples. Duplicate index labels are supported;
            selections use integer row positions. Unsupported scalars raise TypeError.
        positions : iterable of int
            Unique, nonnegative, in-bounds positions in the original source frame.
            Input order is normalized to source order; an empty iterable is valid.
        name : str, optional
            Scope label; default 'selection'.
        **runtime : Unpack[Runtime]
            Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

        Returns
        -------
        Scope
            Frozen selection with the full source fingerprint and sorted positions.

        Raises
        ------
        KeyError
            A requested column is unknown.
        ValueError
            Columns, limits, thresholds, constraints, or source scope are invalid.
        TypeError
            The frame, column labels, or scalar values are unsupported.
        AnalysisCancelled
            Cancellation or the cooperative timeout stops analysis.

        Notes
        -----
        Reads and fingerprints the whole frame without mutation. Positions are not
        index labels; duplicate dataframe indexes are allowed.
        """
        selected = tuple(positions)
        if any(
            isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 0 or p >= len(df)
            for p in selected
        ):
            raise ValueError("Scope positions must be valid nonnegative row positions")
        if len(set(selected)) != len(selected):
            raise ValueError("Scope positions must not repeat")
        return cls(fingerprint(df), tuple(sorted(int(p) for p in selected)), name)

    @operation("scope refinement")
    def refine(
        self,
        df: pd.DataFrame,
        positions: Iterable[int | np.integer[Any]],
        *,
        name: str = "refined",
        **runtime: Unpack[Runtime],
    ) -> Scope:
        """Select a subset of this scope using absolute source positions.

        Parameters
        ----------
        df : pandas.DataFrame
            Source frame, read without mutation. Labels may be unique strings, integers,
            or recursively nested tuples. Duplicate index labels are supported;
            selections use integer row positions. Unsupported scalars raise TypeError.
        positions : iterable of int
            Unique absolute source positions contained in this scope, not offsets
            within its selected rows. An empty iterable creates an empty child.
        name : str, optional
            Child scope label; default 'refined'.
        **runtime : Unpack[Runtime]
            Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

        Returns
        -------
        Scope
            Child scope with this scope's name as parent; the parent is unchanged.

        Raises
        ------
        KeyError
            A requested column is unknown.
        ValueError
            Columns, limits, thresholds, constraints, or source scope are invalid.
        TypeError
            The frame, column labels, or scalar values are unsupported.
        AnalysisCancelled
            Cancellation or the cooperative timeout stops analysis.

        Notes
        -----
        The supplied frame must match the parent's ordered source. A valid row
        position outside the parent is rejected with ValueError.
        """
        child = Scope.from_positions(df, positions, name=name)
        if child.dataset_id != self.dataset_id or not set(child.positions) <= set(self.positions):
            raise ValueError("Refinement must select positions from its parent scope")
        return Scope(child.dataset_id, child.positions, name, self.name)


def columns(df, selected=None):
    """Requested column names (str of each label); None selects every column."""
    frame = labelled(df)
    if selected is None:
        return list(frame.columns)
    selected = [c if isinstance(c, str) else str(c) for c in selected]
    if len(set(selected)) != len(selected):
        raise ValueError("Columns must not repeat")
    for c in selected:
        if c not in frame:
            raise KeyError(c)
    return selected


def _sentinel_key(value):
    """Sentinels match by exported value, and numbers numerically (-999 == -999.0).

    Matching exported values lets saved sentinels, such as a timestamp saved as
    ISO text, apply again unchanged. Booleans never match numbers.
    """
    exported = json_value(value)
    if isinstance(exported, (int, float)) and not isinstance(exported, bool):
        return ("number", exported)
    return (type(exported).__name__, exported)


def limit(name, value, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def prepare(
    df,
    *,
    scope=None,
    missing=None,
    table_id="table",
    features=None,
    presence_features=(),
    optional=(),
):
    """Prepare source identity, scope, sentinel conventions and encodings.

    Returns the scoped frame, its source positions, codes of ``features``
    (default every column), presence masks of ``features`` and
    ``presence_features``, and the base payload every result starts from.
    Columns in ``optional`` were selected automatically rather than named by the
    caller. If their values cannot be encoded they are omitted from the returned
    encodings and listed in ``base["skipped_features"]`` instead of raising.
    """
    if not isinstance(table_id, str) or not table_id:
        raise ValueError("table_id must be a nonempty string")
    identity = fingerprint(df)
    if scope is not None and scope.dataset_id != identity:
        raise ValueError("Scope belongs to a different ordered dataset")
    if scope is not None and any(p >= len(df) for p in scope.positions):
        raise ValueError("Scope positions exceed the source population")
    source = labelled(df)
    sentinel_keys = _sentinels(source, missing)
    frame, positions, all_encoded, all_available = _scoped(df, source, scope, sentinel_keys)
    selected = list(dict.fromkeys(source.columns if features is None else features))
    presence_columns = list(dict.fromkeys([*selected, *presence_features]))
    needed = [
        c
        for c in presence_columns
        if c not in all_available or (c in selected and c not in all_encoded)
    ]
    skipped = {}
    with phase("encoding", len(needed), "columns") as tracker:
        for c in needed:
            try:
                _encode(frame, c, c in selected, sentinel_keys[c], all_encoded, all_available)
            except TypeError as error:
                if c not in optional:
                    raise TypeError(f"Column {c!r}: {error}") from error
                skipped[c] = _unsupported_type(frame[c])
            tracker.advance(detail=str(c))
    encoded = {c: all_encoded[c] for c in selected if c not in skipped}
    available = {c: all_available[c] for c in presence_columns if c not in skipped}
    base = {
        "status": "computed" if len(frame) else "empty",
        "source": {
            "dataset_id": identity,
            "table_id": table_id,
            "rows": len(df),
            "columns": len(df.columns),
        },
        # The one population record: nested measurements count evaluated and
        # excluded rows within scope["evaluated_rows"].
        "scope": {
            "name": scope.name if scope else "input",
            "parent": scope.parent if scope else None,
            "input_rows": len(df),
            "evaluated_rows": len(frame),
            "restriction_excluded_rows": len(df) - len(frame),
            "selection_positions": list(scope.positions) if scope else None,
        },
        "missing_convention": {
            "sentinels": {
                c: [value for _, value in keys] for c, keys in sentinel_keys.items() if keys
            }
        },
        "skipped_features": [{"feature": c, "value_type": kind} for c, kind in skipped.items()],
        "findings": [],
    }
    return frame, positions, encoded, available, base


def _sentinels(source, missing):
    """Sorted sentinel keys declared for each column (empty when none)."""
    declared = {}
    if missing:
        names = resolve_columns(source, missing, argument="missing")
        declared = dict(zip(names, missing.values()))
    return {
        c: sorted({_sentinel_key(python_value(v)) for v in declared.get(c, [])}) for c in source
    }


def _scoped(df, source, scope, sentinel_keys):
    """The scoped frame and its per-session encoding caches, shared across analyses."""
    key = (
        id(df),
        scope.positions if scope else None,
        tuple((c, tuple(keys)) for c, keys in sentinel_keys.items()),
    )
    session = current_session()
    cached = session.prepared.get(key) if session else None
    if cached is None:
        positions = np.array(scope.positions, dtype=np.int64) if scope else np.arange(len(df))
        frame = source.iloc[positions] if scope else source
        # df is kept so its id cannot be reused while the entry lives.
        cached = (df, frame, positions, {}, {})
        if session:
            session.prepared[key] = cached
    return cached[1:]


def _encode(frame, c, selected, sentinel_keys, encoded, available):
    """Record a column's presence (and, when selected, its codes) in the caches."""
    dtype = frame[c].dtype
    native_only = (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
        or isinstance(dtype, pd.StringDtype)
    )
    if not selected and not sentinel_keys and native_only:
        available[c] = frame[c].notna().to_numpy(dtype=bool)
        return
    values, codes = encode_column(frame, c)
    sentinels = set(sentinel_keys)
    mask = np.array(
        [v is not None and _sentinel_key(v) not in sentinels for v in values], dtype=bool
    )
    available[c] = mask[codes]
    if selected:
        encoded[c] = codes


def prepare_values(df, columns, *, scope=None, missing=None, table_id="table"):
    """Prepare named columns as dictionaries whose sentinels share the missing level.

    Returns the scoped frame, source positions, ``{column: (values, codes)}`` and
    the base payload. Unsupported cells in these columns raise TypeError.
    """
    frame, positions, codes, present, base = prepare(
        df, scope=scope, missing=missing, table_id=table_id, features=list(columns)
    )
    return frame, positions, normalized_encoding(frame, codes, present), base


def _unsupported_type(series):
    for value in series.array:
        try:
            python_value(value)
        except TypeError:
            return type(value).__name__
    return "unknown"


def analyzable(selected, base):
    """Drop automatically selected columns that preparation skipped."""
    skipped = {record["feature"] for record in base["skipped_features"]}
    return [c for c in selected if c not in skipped]


def normalized_encoding(frame, codes, present):
    """Dictionaries whose native and declared missing values share one final level."""
    output = {}
    for c, code in codes.items():
        checkpoint()
        values = encode_column(frame, c)[0]
        absent = set(np.unique(code[~present[c]]).tolist())
        if all(values[i] is None for i in absent):
            output[c] = (values, code)
            continue
        # Dropping sentinel values keeps the remaining canonical order.
        kept = [i for i, v in enumerate(values) if i not in absent and v is not None]
        remap = np.full(len(values), len(kept), dtype=np.int64)
        remap[kept] = np.arange(len(kept))
        output[c] = ([values[i] for i in kept] + [None], remap[code])
    return output


def selection(positions, total, example_limit):
    return {
        "positions": [int(p) for p in positions[:example_limit]],
        "total": int(total),
        "omitted": max(0, int(total) - example_limit),
        "method": "first_in_source_order",
        "limit": example_limit,
    }


@dataclass(frozen=True)
class EvidenceRows:
    """Bounded examples with the full source-row count, private to finding assembly."""

    positions: Any
    total: int

    def __len__(self):
        return self.total

    def __getitem__(self, key):
        return self.positions[key]


def bounded_rows(positions, mask, limit):
    from ._explore._kernels import first_indices

    return EvidenceRows(positions[first_indices(mask, limit)], int(np.count_nonzero(mask)))


def finding(
    base,
    kind,
    statement,
    features,
    metrics,
    positions,
    *,
    exceptions=(),
    example_limit=5,
    unit="rows",
    selector=None,
    structure=None,
):
    record = {
        "id": f"f{len(base['findings'])}",
        "pattern": kind,
        "statement": statement,
        "features": [{"table": base["source"]["table_id"], "column": c} for c in features],
        "counting_unit": unit,
        "structure": structure or {},
        "measurements": metrics,
        "examples": selection(positions, len(positions), example_limit),
        "exceptions": selection(exceptions, len(exceptions), example_limit),
        "selector": {
            "dataset_id": base["source"]["dataset_id"],
            "scope_ref": "scope",
            "parameters_ref": "parameters",
            "missing_convention_ref": "missing_convention",
            "finding_id": f"f{len(base['findings'])}",
            **(selector or {}),
        },
    }
    base["findings"].append(record)
    return record


def result(kind, base):
    return Result(kind, base)


def qualitative_analysis_unit(base, record):
    """Describe a finding's own unit without exporting support or population sizes."""
    section = record.get("selector", {}).get("analysis_section")
    if section:
        base = base["sections"][section]
    unit = record["counting_unit"]
    analysis = base.get("analysis_unit", {})
    output = {"counting_unit": unit}
    if unit == analysis.get("counting_unit"):
        output.update(
            {k: analysis[k] for k in ("entity_keys", "presence_aggregation") if k in analysis}
        )
    elif unit == "rows":
        output.update(entity_keys=[], presence_aggregation="per_row")
    else:
        output["entity_keys"] = record.get("structure", {}).get("entity_keys", [])
    if record["pattern"] in {"entity_availability", "entity_summary"}:
        output.pop("presence_aggregation", None)
    return output


def saved_context(base):
    """Restore source-bound scope and typed missing conventions from saved evidence."""
    data = base["scope"]
    positions = data.get("selection_positions")
    return {
        "scope": Scope(
            base["source"]["dataset_id"], tuple(positions), data["name"], data.get("parent")
        )
        if positions is not None
        else None,
        "missing": {c: values for c, values in base["missing_convention"]["sentinels"].items()},
        "table_id": base["source"]["table_id"],
    }


def context_statement(context):
    return ", ".join(f"{feature} = {display(value)}" for feature, value in context.items())
