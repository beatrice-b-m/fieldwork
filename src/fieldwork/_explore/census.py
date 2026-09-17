"""Independent level counts and bounded nested census."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from ._kernels import dense_counts
from .encoding import (
    MISSING,
    ScalarIdentity,
    encode_series,
    normalize_scalar,
    resolve_columns,
    validate_limit,
    validate_schema,
)
from .result import ExplorerResult


def _source(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "dtypes": [
            {
                "column": normalize_scalar(column, label=True).to_dict(),
                "dtype": str(df[column].dtype),
            }
            for column in df.columns
        ],
    }


def _scope(
    scope_id: str,
    input_rows: int,
    missing_excluded_rows: int,
    restriction_excluded_rows: int,
    conditional: bool,
    lineage: list[str] | None = None,
    *,
    parent_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if parent_scope is not None:
        input_rows = parent_scope["input_rows"]
        missing_excluded_rows += parent_scope["missing_excluded_rows"]
        restriction_excluded_rows += parent_scope["restriction_excluded_rows"]
        conditional = conditional or parent_scope["conditional"]
        lineage = [*parent_scope["lineage"], parent_scope["scope_id"], *(lineage or [])]
    evaluated = input_rows - missing_excluded_rows - restriction_excluded_rows
    return {
        "scope_id": scope_id,
        "input_rows": input_rows,
        "missing_excluded_rows": missing_excluded_rows,
        "restriction_excluded_rows": restriction_excluded_rows,
        "evaluated_rows": evaluated,
        "retained_rows": evaluated,
        "conditional": conditional,
        "lineage": lineage or [],
    }


def _rank_counts(counts: dict[int, int], values: list[ScalarIdentity]) -> list[tuple[int, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], values[item[0]].sort_key()))


def _mixed_warning(
    feature_id: str, column: Any, values: Iterable[ScalarIdentity]
) -> dict[str, Any] | None:
    families = sorted({value.kind for value in values if value is not MISSING})
    if len(families) > 1:
        return {
            "code": "MIXED_LEVEL_TYPES",
            "feature_id": feature_id,
            "column": normalize_scalar(column, label=True).to_dict(),
            "families": families,
        }
    return None


def levels(
    df: pd.DataFrame,
    features: Iterable[Any] | None = None,
    *,
    top_n: int | None = None,
    max_levels: int | None = 100,
    min_count: int = 1,
    dropna: bool = False,
    schema: dict[Any, str] | None = None,
    engine_metadata: bool = False,
    scope_metadata: dict[str, Any] | None = None,
) -> ExplorerResult:
    """Count levels independently for every requested feature."""

    selected = resolve_columns(df, features, argument="features", default_all=True)
    validate_schema(df, schema)
    validate_limit("top_n", top_n, zero=False)
    validate_limit("max_levels", max_levels)
    validate_limit("min_count", min_count)
    records: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    scopes: list[dict[str, Any]] = []
    for position, column in enumerate(selected):
        feature_id = f"f{position}"
        values, codes = encode_series(df[column])
        missing_code = values.index(MISSING) if MISSING in values else None
        eligible = np.arange(len(df), dtype=np.int64)
        missing_excluded = 0
        if dropna and missing_code is not None:
            keep = codes != missing_code
            missing_excluded = int((~keep).sum())
            eligible = eligible[keep]
        counts = dense_counts(codes, eligible)
        ranked = _rank_counts(counts, values)
        semantic = ranked[:top_n] if top_n is not None else ranked
        semantic = [item for item in semantic if item[1] >= min_count]
        reported = semantic[:max_levels] if max_levels is not None else semantic
        evaluated_rows = len(eligible)
        output_levels = [
            {
                "level_id": f"{feature_id}:l{code}",
                "rank": rank,
                "value": values[code].to_dict(),
                "count": count,
                "share_of_feature": count / evaluated_rows if evaluated_rows else None,
                "share_reason": None if evaluated_rows else "empty_population",
            }
            for rank, (code, count) in enumerate(reported, 1)
        ]
        reported_rows = sum(item[1] for item in reported)
        scope_id = f"s1:{feature_id}"
        scopes.append(_scope(scope_id, len(df), missing_excluded, 0, False))
        records.append(
            {
                "feature_id": feature_id,
                "column": normalize_scalar(column, label=True).to_dict(),
                "role": (schema or {}).get(column),
                "scope_id": scope_id,
                "status": "empty" if evaluated_rows == 0 else "computed",
                "levels_total": len(ranked),
                "levels_reported": len(reported),
                "omitted_levels": len(ranked) - len(reported),
                "reported_rows": reported_rows,
                "unreported_rows": evaluated_rows - reported_rows,
                "levels": output_levels,
            }
        )
        warning = _mixed_warning(feature_id, column, values)
        if warning:
            warnings.append(warning)
        role = (schema or {}).get(column)
        if role in {"id", "continuous"}:
            warnings.append(
                {
                    "code": "EXPLICIT_ROLE_SELECTION",
                    "feature_id": feature_id,
                    "column": normalize_scalar(column, label=True).to_dict(),
                    "role": role,
                }
            )
    payload: dict[str, Any] = {
        "status": "empty" if len(df) == 0 else "computed",
        "source": _source(df),
        "scopes": scopes,
        "effective_limits": {
            "top_n": top_n,
            "max_levels": max_levels,
            "min_count": min_count,
            "dropna": dropna,
        },
        "per_feature": records,
        "warnings": warnings,
    }
    if scope_metadata:
        payload["scope_metadata"] = scope_metadata
    if engine_metadata:
        payload["engine"] = {"name": "typed_dense_counts"}
    return ExplorerResult("levels", payload)


def _pre_mask_per_parent(
    codes: list[np.ndarray],
    values: list[list[ScalarIdentity]],
    eligible: np.ndarray,
    top_n: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    surviving = np.zeros(codes[0].shape[0], dtype=bool)
    retained: list[dict[str, Any]] = []
    queue: deque[tuple[int, np.ndarray, tuple[int, ...]]] = deque([(0, eligible, ())])
    while queue:
        depth, rows, path = queue.popleft()
        counts = dense_counts(codes[depth], rows)
        chosen = _rank_counts(counts, values[depth])[:top_n]
        retained.append(
            {
                "path": list(path),
                "depth": depth + 1,
                "level_codes": [code for code, _ in chosen],
            }
        )
        for code, _ in chosen:
            child_rows = rows[codes[depth][rows] == code]
            child_path = (*path, code)
            if depth + 1 == len(codes):
                surviving[child_rows] = True
            else:
                queue.append((depth + 1, child_rows, child_path))
    return surviving, retained


def census(
    df: pd.DataFrame,
    dimensions: Iterable[Any],
    *,
    top_n: int | None = None,
    top_n_mode: str = "post",
    top_n_per_parent: bool = False,
    min_retained_fraction: float = 0.01,
    max_depth: int | None = None,
    max_levels: int | None = 100,
    max_nodes: int | None = 10000,
    min_count: int = 1,
    dropna: bool = False,
    schema: dict[Any, str] | None = None,
    engine_metadata: bool = False,
) -> ExplorerResult:
    """Build a deterministic, ancestor-closed observed-prefix census."""

    selected = resolve_columns(df, dimensions, argument="dimensions")
    validate_schema(df, schema)
    validate_limit("top_n", top_n, zero=False)
    validate_limit("max_depth", max_depth, zero=False)
    validate_limit("max_levels", max_levels)
    validate_limit("max_nodes", max_nodes)
    validate_limit("min_count", min_count)
    if top_n_mode not in {"pre", "post"}:
        raise ValueError("top_n_mode must be 'pre' or 'post'")
    if (
        not isinstance(min_retained_fraction, (int, float))
        or isinstance(min_retained_fraction, bool)
        or not 0 <= min_retained_fraction <= 1
    ):
        raise ValueError("min_retained_fraction must be between 0 and 1")
    active = selected[:max_depth] if max_depth is not None else selected
    dictionaries: list[list[ScalarIdentity]] = []
    code_arrays: list[np.ndarray] = []
    missing_codes: list[int | None] = []
    for column in active:
        values, codes = encode_series(df[column])
        dictionaries.append(values)
        code_arrays.append(codes)
        missing_codes.append(values.index(MISSING) if MISSING in values else None)
    eligible_mask = np.ones(len(df), dtype=bool)
    if dropna:
        for codes, missing_code in zip(code_arrays, missing_codes):
            if missing_code is not None:
                eligible_mask &= codes != missing_code
    eligible = np.flatnonzero(eligible_mask)
    missing_excluded = len(df) - len(eligible)
    retained_metadata: list[dict[str, Any]] = []
    evaluated = eligible
    warnings: list[dict[str, Any]] = []
    if top_n_mode == "pre" and top_n is not None and len(eligible):
        if top_n_per_parent:
            final_mask, retained_metadata = _pre_mask_per_parent(
                code_arrays, dictionaries, eligible, top_n
            )
        else:
            final_mask = eligible_mask.copy()
            for depth, (codes, values) in enumerate(zip(code_arrays, dictionaries)):
                ranked = _rank_counts(dense_counts(codes, eligible), values)
                chosen = {code for code, _ in ranked[:top_n]}
                retained_metadata.append({"depth": depth + 1, "level_codes": sorted(chosen)})
                final_mask &= np.isin(codes, list(chosen))
        evaluated = np.flatnonzero(final_mask)
        if not len(evaluated):
            raise ValueError(
                "DEGENERATE_TOP_N: pre selection removed every eligible row "
                f"({len(eligible)} eligible rows)"
            )
        fraction = len(evaluated) / len(eligible)
        if fraction < min_retained_fraction:
            warnings.append(
                {
                    "code": "LOW_RETAINED_FRACTION",
                    "retained_fraction": fraction,
                    "eligible_rows": len(eligible),
                    "retained_rows": len(evaluated),
                }
            )
    restriction_excluded = len(eligible) - len(evaluated)
    scope = _scope(
        "s2",
        len(df),
        missing_excluded,
        restriction_excluded,
        top_n_mode == "pre" and restriction_excluded > 0,
        ["input", "dropna" if dropna else "include_missing", top_n_mode],
    )
    features = [
        {
            "feature_id": f"f{index}",
            "column": normalize_scalar(column, label=True).to_dict(),
            "role": (schema or {}).get(column),
            "dictionary_cardinality": len(dictionaries[index]),
        }
        for index, column in enumerate(active)
    ]
    for index, values in enumerate(dictionaries):
        warning = _mixed_warning(f"f{index}", active[index], values)
        if warning:
            warnings.append(warning)
        role = (schema or {}).get(active[index])
        if role in {"id", "continuous"}:
            warnings.append(
                {
                    "code": "EXPLICIT_ROLE_SELECTION",
                    "feature_id": f"f{index}",
                    "column": normalize_scalar(active[index], label=True).to_dict(),
                    "role": role,
                }
            )
    global_chosen: list[set[int] | None] = []
    for codes, values in zip(code_arrays, dictionaries):
        if top_n is not None and not top_n_per_parent and top_n_mode == "post":
            global_chosen.append(
                {code for code, _ in _rank_counts(dense_counts(codes, evaluated), values)[:top_n]}
            )
        else:
            global_chosen.append(None)
    nodes: list[dict[str, Any]] = []
    emitted_levels: set[tuple[int, int]] = set()
    queue: deque[tuple[str, int, np.ndarray, int]] = deque()
    queue.append(("root", 0, evaluated, len(evaluated)))
    root = {
        "node_id": "root",
        "parent_id": None,
        "feature_id": None,
        "level_id": None,
        "depth": 0,
        "count": len(evaluated),
        "share_of_parent": None,
        "share_of_total": 1.0 if len(evaluated) else None,
        "share_reason": None if len(evaluated) else "empty_population",
        "expansion_state": "unexpanded" if active else "complete",
        "omitted_child_rows": 0,
        "omitted_child_levels": 0,
        "stop_reasons": [],
    }
    node_lookup: dict[str, dict[str, Any]] = {"root": root}
    next_id = 0
    while queue:
        parent_id, depth, rows, parent_count = queue.popleft()
        parent = node_lookup[parent_id]
        if depth >= len(active):
            parent["expansion_state"] = "complete"
            continue
        counts = dense_counts(code_arrays[depth], rows)
        ranked = _rank_counts(counts, dictionaries[depth])
        chosen = ranked
        if top_n is not None:
            if top_n_per_parent and top_n_mode == "post":
                chosen = chosen[:top_n]
            elif global_chosen[depth] is not None:
                chosen = [item for item in chosen if item[0] in global_chosen[depth]]
        chosen = [item for item in chosen if item[1] >= min_count]
        if max_levels is not None:
            chosen = chosen[:max_levels]
        remaining_budget = None if max_nodes is None else max_nodes - len(nodes)
        budget_truncated = remaining_budget is not None and len(chosen) > max(0, remaining_budget)
        if remaining_budget is not None:
            chosen = chosen[: max(0, remaining_budget)]
        chosen_codes = {code for code, _ in chosen}
        omitted_rows = sum(count for code, count in ranked if code not in chosen_codes)
        parent["omitted_child_rows"] = omitted_rows
        parent["omitted_child_levels"] = len(ranked) - len(chosen)
        parent["expansion_state"] = "expanded"
        reasons: list[str] = []
        if len(chosen) < len(ranked):
            if top_n is not None:
                reasons.append("top_n")
            if max_levels is not None and len(ranked) > max_levels:
                reasons.append("max_levels")
            if budget_truncated or (max_nodes is not None and len(nodes) >= max_nodes):
                reasons.append("max_nodes")
            if any(count < min_count for _, count in ranked):
                reasons.append("min_count")
        parent["stop_reasons"] = sorted(set(reasons))
        for code, count in chosen:
            node_id = f"n{next_id}"
            next_id += 1
            child_rows = rows[code_arrays[depth][rows] == code]
            node = {
                "node_id": node_id,
                "parent_id": parent_id,
                "feature_id": f"f{depth}",
                "level_id": f"f{depth}:l{code}",
                "depth": depth + 1,
                "count": count,
                "share_of_parent": count / parent_count if parent_count else None,
                "share_of_total": count / len(evaluated) if len(evaluated) else None,
                "share_reason": None if len(evaluated) else "empty_population",
                "expansion_state": "complete" if depth + 1 == len(active) else "unexpanded",
                "omitted_child_rows": 0,
                "omitted_child_levels": 0,
                "stop_reasons": [],
            }
            nodes.append(node)
            node_lookup[node_id] = node
            emitted_levels.add((depth, code))
            if depth + 1 < len(active):
                queue.append((node_id, depth + 1, child_rows, count))
    # Pre-selection metadata must remain decodable even when no corresponding
    # tree node survives the output budgets or the conjunctive pre filter.
    referenced_levels = emitted_levels.copy()
    for retained in retained_metadata:
        referenced_levels.update((retained["depth"] - 1, code) for code in retained["level_codes"])
        referenced_levels.update(enumerate(retained.get("path", [])))
    level_dictionary = [
        {
            "level_id": f"f{depth}:l{code}",
            "feature_id": f"f{depth}",
            "value": dictionaries[depth][code].to_dict(),
        }
        for depth, code in sorted(
            referenced_levels,
            key=lambda item: (item[0], dictionaries[item[0]][item[1]].sort_key()),
        )
    ]
    status = "empty" if not len(evaluated) else "computed"
    if any(node["omitted_child_rows"] for node in [root, *nodes]):
        status = "partial" if len(evaluated) else status
    payload: dict[str, Any] = {
        "status": status,
        "source": _source(df),
        "scopes": [scope],
        "features": features,
        "level_dictionary": level_dictionary,
        "tree": {
            "status": status,
            "scope_id": "s2",
            "dimensions": [f"f{i}" for i in range(len(active))],
            "requested_depth": len(active),
            "root": root,
            "nodes": nodes,
            "retained_sets": retained_metadata,
        },
        "effective_limits": {
            "top_n": top_n,
            "top_n_mode": top_n_mode,
            "top_n_per_parent": top_n_per_parent,
            "max_depth": max_depth,
            "max_levels": max_levels,
            "max_nodes": max_nodes,
            "min_count": min_count,
            "dropna": dropna,
        },
        "warnings": warnings,
    }
    if engine_metadata:
        payload["engine"] = {"name": "encoded_observed_prefix_refinement"}
    return ExplorerResult("census", payload)
