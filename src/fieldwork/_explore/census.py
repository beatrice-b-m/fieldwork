"""Independent level counts and a bounded tree of observed dimension prefixes."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Unpack

import numpy as np
import pandas as pd

from .._runtime import checkpoint, operation, phase
from ..result import Result
from ..typing import Runtime, SchemaRole
from .encoding import json_value, validate_limit, validate_schema, value_key

if TYPE_CHECKING:
    from ..evidence import Scope

_FAMILIES = (
    "boolean",
    "integer",
    "float",
    "string",
    "date",
    "datetime_naive",
    "datetime_aware",
    "timedelta",
)


def _ranked(codes: np.ndarray) -> list[tuple[int, int]]:
    """(code, count) pairs, most frequent first; ties follow canonical value order."""
    if not len(codes):
        return []
    observed, counts = np.unique(codes, return_counts=True)
    order = np.lexsort((observed, -counts))
    return list(zip(observed[order].tolist(), counts[order].tolist()))


def _feature_warnings(column: str, values: Iterable[Any], role: str | None) -> list[dict[str, Any]]:
    """Mixed value types, and advisory roles unsuited to categorical counting."""
    warnings = []
    families = [_FAMILIES[f] for f in sorted({value_key(v)[0] for v in values if v is not None})]
    if len(families) > 1:
        warnings.append({"code": "MIXED_LEVEL_TYPES", "column": column, "families": families})
    if role in {"id", "continuous"}:
        warnings.append({"code": "EXPLICIT_ROLE_SELECTION", "column": column, "role": role})
    return warnings


def _complete(encoded: list[tuple[list[Any], np.ndarray]], size: int) -> np.ndarray:
    """Rows where no given column is missing."""
    mask = np.ones(size, dtype=bool)
    for values, codes in encoded:
        if values and values[-1] is None:
            mask &= codes != len(values) - 1
    return mask


@operation("levels")
def levels(
    df: pd.DataFrame,
    features: Iterable[str] | None = None,
    *,
    top_n: int | None = None,
    max_levels: int | None = 100,
    min_count: int = 1,
    dropna: bool = False,
    schema: dict[str, SchemaRole] | None = None,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Count observed values independently for each requested column.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    features : iterable of str or None, optional
        Columns to count; default None counts every column.
    top_n : int or None, optional
        Report only the top_n most frequent levels; default None. Output only.
    max_levels : int or None, optional
        Displayed levels per feature; default 100, None is unbounded.
    min_count : int, optional
        Minimum displayed count; default 1. Output only.
    dropna : bool, optional
        Default False counts missing values as a level; True excludes them per
        feature, so denominators can differ between features.
    schema : dict or None, optional
        Advisory roles ('id', 'categorical', 'continuous', 'unknown') by column;
        'id' and 'continuous' roles add a warning. Values are never cast.
    scope, missing, table_id
        Source context shared by every analysis: a Scope restricting rows,
        extra missing sentinels per column, and a source label.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'levels': one record per feature with ranked levels (count, then
        value order), evaluated and missing-excluded rows, and omitted mass.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.levels(pd.DataFrame({"site": ["A", "A", "B"]}), top_n=1)
    >>> result["per_feature"][0]["levels"][0]["value"]
    'A'
    """
    from ..evidence import columns, prepare_values

    selected = columns(df, features)
    validate_limit("top_n", top_n, zero=False)
    validate_limit("max_levels", max_levels)
    validate_limit("min_count", min_count)
    frame, _, encoded, base = prepare_values(
        df, selected, scope=scope, missing=missing, table_id=table_id
    )
    roles = validate_schema(frame, schema)
    limits = {"top_n": top_n, "max_levels": max_levels, "min_count": min_count}
    records, warnings = [], []
    with phase("level counts", len(selected), "columns") as tracker:
        for column in selected:
            values, codes = encoded[column]
            records.append(_level_record(column, values, codes, roles.get(column), dropna, limits))
            warnings += _feature_warnings(column, values, roles.get(column))
            tracker.advance(detail=column)
    base["parameters"] = {"features": selected, **limits, "dropna": dropna, "schema": roles}
    base["per_feature"] = records
    base["warnings"] = warnings
    return Result("levels", base)


def _level_record(
    column: str,
    values: list[Any],
    codes: np.ndarray,
    role: str | None,
    dropna: bool,
    limits: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = codes[_complete([(values, codes)], len(codes))] if dropna else codes
    ranked = _ranked(eligible)
    shown = ranked[: limits["top_n"]] if limits["top_n"] is not None else ranked
    shown = [item for item in shown if item[1] >= limits["min_count"]]
    shown = shown[: limits["max_levels"]] if limits["max_levels"] is not None else shown
    evaluated = len(eligible)
    reported = sum(count for _, count in shown)
    return {
        "column": column,
        "role": role,
        "status": "computed" if evaluated else "empty",
        "evaluated_rows": evaluated,
        "missing_excluded_rows": len(codes) - evaluated,
        "levels_total": len(ranked),
        "levels_reported": len(shown),
        "omitted_levels": len(ranked) - len(shown),
        "reported_rows": reported,
        "unreported_rows": evaluated - reported,
        "levels": [
            {
                "rank": rank,
                "value": json_value(values[code]),
                "count": count,
                "share_of_feature": count / evaluated if evaluated else None,
            }
            for rank, (code, count) in enumerate(shown, 1)
        ],
    }


def _preselect(
    codes: list[np.ndarray], eligible: np.ndarray, top_n: int, per_parent: bool
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Rows keeping only the top_n levels of every dimension, and the kept levels.

    Globally, each dimension keeps its top_n levels over the eligible rows. Per
    parent, each prefix keeps the top_n levels among its own rows.
    """
    if per_parent:
        return _preselect_per_parent(codes, eligible, top_n)
    mask = np.zeros(codes[0].shape[0] if codes else 0, dtype=bool)
    mask[eligible] = True
    retained = []
    for depth, dimension in enumerate(codes):
        chosen = sorted(code for code, _ in _ranked(dimension[eligible])[:top_n])
        retained.append({"depth": depth + 1, "level_codes": chosen})
        mask &= np.isin(dimension, chosen)
    return mask, retained


def _preselect_per_parent(
    codes: list[np.ndarray], eligible: np.ndarray, top_n: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    surviving = np.zeros(codes[0].shape[0], dtype=bool)
    retained: list[dict[str, Any]] = []
    queue: deque[tuple[int, np.ndarray, tuple[int, ...]]] = deque([(0, eligible, ())])
    while queue:
        depth, rows, path = queue.popleft()
        chosen = [code for code, _ in _ranked(codes[depth][rows])[:top_n]]
        retained.append({"path": list(path), "depth": depth + 1, "level_codes": chosen})
        for code in chosen:
            child_rows = rows[codes[depth][rows] == code]
            if depth + 1 == len(codes):
                surviving[child_rows] = True
            else:
                queue.append((depth + 1, child_rows, (*path, code)))
    return surviving, retained


@dataclass(frozen=True)
class _Limits:
    top_n: int | None
    per_parent: bool
    post: bool
    max_levels: int | None
    max_nodes: int | None
    min_count: int


@operation("census")
def census(
    df: pd.DataFrame,
    dimensions: Iterable[str],
    *,
    top_n: int | None = None,
    top_n_mode: Literal["pre", "post"] = "post",
    top_n_per_parent: bool = False,
    min_retained_fraction: float = 0.01,
    max_depth: int | None = None,
    max_levels: int | None = 100,
    max_nodes: int | None = 10000,
    min_count: int = 1,
    dropna: bool = False,
    schema: dict[str, SchemaRole] | None = None,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Build a bounded tree of observed dimension prefixes with exact counts.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    dimensions : iterable of str
        Nonempty ordered columns; order changes the tree.
    top_n : int or None, optional
        Keep the top_n levels per dimension; default None keeps all.
    top_n_mode : {'post', 'pre'}, optional
        'post' (default) counts every row and limits output; 'pre' keeps only
        rows whose levels are all kept, recording the excluded rows.
    top_n_per_parent : bool, optional
        Choose leading levels within each parent prefix instead of globally.
    min_retained_fraction : float, optional
        Warn when pre-selection keeps less than this share of rows; default 0.01.
    max_depth : int or None, optional
        Use only the first max_depth dimensions; default None uses all.
    max_levels, max_nodes, min_count : optional
        Display limits (children per node, total nodes, minimum count); defaults
        100, 10000 and 1. Omitted children keep their mass in the parent.
    dropna : bool, optional
        True excludes rows missing any active dimension; default False keeps
        missing values as a level.
    schema : dict or None, optional
        Advisory roles by column, as in levels.
    scope, missing, table_id
        Source context shared by every analysis.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'census': ``tree`` with its evaluated, missing-excluded and
        pre-selection-excluded rows, parent-linked nodes naming their column and
        value, kept levels of a pre-selection, and warnings. No unobserved
        combinations are invented.

    Raises
    ------
    ValueError
        Pre-selection removes every row, or options are invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"site": ["A", "A", "B"], "visit": [1, 2, 1]})
    >>> fw.census(df, ["site", "visit"])["tree"]["nodes"][0]["count"]
    2
    """
    from ..evidence import columns, prepare_values

    selected = columns(df, dimensions)
    if not selected:
        raise ValueError("dimensions must contain at least one column")
    for name, value, zero in [
        ("top_n", top_n, False),
        ("max_depth", max_depth, False),
        ("max_levels", max_levels, True),
        ("max_nodes", max_nodes, True),
        ("min_count", min_count, True),
    ]:
        validate_limit(name, value, zero=zero)
    if top_n_mode not in {"pre", "post"}:
        raise ValueError("top_n_mode must be 'pre' or 'post'")
    if isinstance(min_retained_fraction, bool) or not 0 <= min_retained_fraction <= 1:
        raise ValueError("min_retained_fraction must be between 0 and 1")
    active = selected[:max_depth] if max_depth is not None else selected
    frame, _, encoded, base = prepare_values(
        df, active, scope=scope, missing=missing, table_id=table_id
    )
    roles = validate_schema(frame, schema)
    values = [encoded[c][0] for c in active]
    codes = [encoded[c][1] for c in active]
    eligible = (
        np.flatnonzero(_complete(list(zip(values, codes)), len(frame)))
        if dropna
        else np.arange(len(frame))
    )
    evaluated, retained, warnings = eligible, [], []
    if top_n_mode == "pre" and top_n is not None and len(eligible):
        evaluated, retained, warning = _cohort(codes, eligible, top_n, top_n_per_parent)
        if warning["retained_fraction"] < min_retained_fraction:
            warnings.append(warning)
    for column, dictionary in zip(active, values):
        warnings += _feature_warnings(column, dictionary, roles.get(column))
    limits = _Limits(
        top_n, top_n_per_parent, top_n_mode == "post", max_levels, max_nodes, min_count
    )
    tree = _tree(active, values, codes, evaluated, limits)
    tree.update(
        evaluated_rows=len(evaluated),
        missing_excluded_rows=len(frame) - len(eligible),
        restriction_excluded_rows=len(eligible) - len(evaluated),
        retained_sets=[_retained(record, active, values) for record in retained],
    )
    base["status"] = tree.pop("status")
    base["parameters"] = {
        "dimensions": selected,
        "top_n": top_n,
        "top_n_mode": top_n_mode,
        "top_n_per_parent": top_n_per_parent,
        "min_retained_fraction": min_retained_fraction,
        "max_depth": max_depth,
        "max_levels": max_levels,
        "max_nodes": max_nodes,
        "min_count": min_count,
        "dropna": dropna,
        "schema": roles,
    }
    base["features"] = [
        {"column": c, "role": roles.get(c), "dictionary_cardinality": len(v)}
        for c, v in zip(active, values)
    ]
    base["tree"] = tree
    base["warnings"] = warnings
    return Result("census", base)


def _cohort(
    codes: list[np.ndarray], eligible: np.ndarray, top_n: int, per_parent: bool
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Rows kept by a pre-selection, the kept levels, and the retention record."""
    mask, retained = _preselect(codes, eligible, top_n, per_parent)
    evaluated = np.flatnonzero(mask)
    if not len(evaluated):
        raise ValueError(
            "DEGENERATE_TOP_N: pre selection removed every eligible row "
            f"({len(eligible)} eligible rows)"
        )
    warning = {
        "code": "LOW_RETAINED_FRACTION",
        "retained_fraction": len(evaluated) / len(eligible),
        "eligible_rows": len(eligible),
        "retained_rows": len(evaluated),
    }
    return evaluated, retained, warning


def _retained(record: dict[str, Any], active: list[str], values: list[list[Any]]) -> dict:
    """Kept levels of a pre-selection, reported by value."""
    depth = record["depth"] - 1
    output = {
        "depth": record["depth"],
        "column": active[depth],
        "values": [json_value(values[depth][code]) for code in record["level_codes"]],
    }
    if "path" in record:
        output["path"] = [json_value(values[i][code]) for i, code in enumerate(record["path"])]
    return output


def _tree(
    active: list[str],
    values: list[list[Any]],
    codes: list[np.ndarray],
    evaluated: np.ndarray,
    limits: _Limits,
) -> dict[str, Any]:
    """Expand prefixes breadth first under the display limits."""
    total = len(evaluated)
    root = _node("root", None, None, None, 0, total, total, active)
    # Global post-selection keeps the same top_n levels under every parent.
    global_top = [
        {code for code, _ in _ranked(dimension[evaluated])[: limits.top_n]}
        if limits.top_n is not None and limits.post and not limits.per_parent
        else None
        for dimension in codes
    ]
    nodes: list[dict[str, Any]] = []
    queue = deque([(root, 0, evaluated)])
    with phase("census tree", unit="parents") as tracker:
        while queue:
            parent, depth, rows = queue.popleft()
            checkpoint()
            ranked = _ranked(codes[depth][rows])
            chosen, truncated = _children(ranked, global_top[depth], limits, len(nodes))
            _record_omissions(parent, ranked, chosen, limits, truncated)
            for code, count in chosen:
                node = _node(
                    f"n{len(nodes)}",
                    parent["node_id"],
                    active[depth],
                    json_value(values[depth][code]),
                    depth + 1,
                    count,
                    parent["count"],
                    active,
                    total,
                )
                nodes.append(node)
                if depth + 1 < len(active):
                    queue.append((node, depth + 1, rows[codes[depth][rows] == code]))
            tracker.advance()
    status = "empty" if not total else "computed"
    if total and any(node["omitted_child_rows"] for node in [root, *nodes]):
        status = "partial"
    return {
        "status": status,
        "dimensions": list(active),
        "requested_depth": len(active),
        "root": root,
        "nodes": nodes,
    }


def _node(
    node_id: str,
    parent_id: str | None,
    column: str | None,
    value: Any,
    depth: int,
    count: int,
    parent_count: int,
    active: list[str],
    total: int | None = None,
) -> dict[str, Any]:
    total = count if total is None else total
    return {
        "node_id": node_id,
        "parent_id": parent_id,
        "column": column,
        "value": value,
        "depth": depth,
        "count": count,
        "share_of_parent": count / parent_count if parent_id and parent_count else None,
        "share_of_total": count / total if total else None,
        "expansion_state": "complete" if depth == len(active) else "unexpanded",
        "omitted_child_rows": 0,
        "omitted_child_levels": 0,
        "stop_reasons": [],
    }


def _children(
    ranked: list[tuple[int, int]], global_top: set[int] | None, limits: _Limits, emitted: int
) -> tuple[list[tuple[int, int]], bool]:
    """Children shown under a parent, and whether the node budget cut them."""
    chosen = ranked
    if limits.top_n is not None and limits.post and limits.per_parent:
        chosen = chosen[: limits.top_n]
    elif global_top is not None:
        chosen = [item for item in chosen if item[0] in global_top]
    chosen = [item for item in chosen if item[1] >= limits.min_count]
    if limits.max_levels is not None:
        chosen = chosen[: limits.max_levels]
    if limits.max_nodes is None:
        return chosen, False
    remaining = max(0, limits.max_nodes - emitted)
    return chosen[:remaining], len(chosen) > remaining or remaining == 0


def _record_omissions(
    parent: dict[str, Any],
    ranked: list[tuple[int, int]],
    chosen: list[tuple[int, int]],
    limits: _Limits,
    truncated: bool,
) -> None:
    """Keep omitted children's mass and the limits that omitted them on the parent."""
    shown = {code for code, _ in chosen}
    parent["omitted_child_rows"] = sum(count for code, count in ranked if code not in shown)
    parent["omitted_child_levels"] = len(ranked) - len(chosen)
    parent["expansion_state"] = "expanded"
    if len(chosen) == len(ranked):
        return
    reasons = set()
    if limits.top_n is not None:
        reasons.add("top_n")
    if limits.max_levels is not None and len(ranked) > limits.max_levels:
        reasons.add("max_levels")
    if truncated:
        reasons.add("max_nodes")
    if any(count < limits.min_count for _, count in ranked):
        reasons.add("min_count")
    parent["stop_reasons"] = sorted(reasons)
