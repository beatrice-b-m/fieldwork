"""Combined explorer orchestration without alternate analysis implementations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .census import _pre_mask_per_parent, _rank_counts, _source, census, levels
from .encoding import encode_series, missing_code, resolve_columns
from .grain import grain
from .relations import pairs
from .result import ExplorerResult


def _pre_cohort(
    df: pd.DataFrame,
    dimensions: tuple[Any, ...],
    *,
    top_n: int,
    top_n_per_parent: bool,
    dropna: bool,
) -> pd.DataFrame:
    dictionaries = []
    codes = []
    eligible_mask = np.ones(len(df), dtype=bool)
    for column in dimensions:
        values, encoded = encode_series(df[column])
        dictionaries.append(values)
        codes.append(encoded)
        absent = missing_code(values)
        if dropna and absent is not None:
            eligible_mask &= encoded != absent
    eligible = np.flatnonzero(eligible_mask)
    if top_n_per_parent:
        mask, _ = _pre_mask_per_parent(codes, dictionaries, eligible, top_n)
    else:
        mask = eligible_mask.copy()
        for encoded, values in zip(codes, dictionaries):
            counts = {
                int(code): int((encoded[eligible] == code).sum())
                for code in np.unique(encoded[eligible])
            }
            chosen = {code for code, _ in _rank_counts(counts, values)[:top_n]}
            mask &= np.isin(encoded, list(chosen))
    return df.iloc[np.flatnonzero(mask)]


def explore(
    df: pd.DataFrame,
    dimensions: Iterable[Any],
    *,
    features: Iterable[Any] | None = None,
    candidate_keys: Iterable[Any] | None = None,
    top_n: int | None = None,
    top_n_mode: str = "post",
    top_n_per_parent: bool = False,
    top_n_applies_to: str = "census",
    min_retained_fraction: float = 0.01,
    max_depth: int | None = None,
    max_levels: int | None = 100,
    max_nodes: int | None = 10000,
    min_count: int = 1,
    dropna: bool = False,
    schema: dict[Any, str] | None = None,
    include_pairs: bool = True,
    include_absence: bool = False,
    reference_domains: Mapping[Any, Iterable[Any]] | None = None,
    pair_contexts: Iterable[Mapping[Any, Any]] | None = None,
    max_absence_cells: int | None = 1000,
    max_contexts: int | None = 32,
    max_pairs: int | None = 15,
) -> ExplorerResult:
    selected = resolve_columns(df, dimensions, argument="dimensions")
    active = selected[:max_depth] if max_depth is not None else selected
    if top_n_applies_to not in {"census", "both"}:
        raise ValueError("top_n_applies_to must be 'census' or 'both'")
    if include_absence and not include_pairs:
        raise ValueError("include_absence=True requires include_pairs=True")
    if top_n_applies_to == "both" and (top_n_mode != "pre" or top_n is None):
        raise ValueError("top_n_applies_to='both' requires pre mode with top_n")
    level_result = levels(
        df,
        features=features if features is not None else selected,
        top_n=top_n,
        max_levels=max_levels,
        min_count=min_count,
        dropna=dropna,
        schema=schema,
    )
    census_result = census(
        df,
        selected,
        top_n=top_n,
        top_n_mode=top_n_mode,
        top_n_per_parent=top_n_per_parent,
        min_retained_fraction=min_retained_fraction,
        max_depth=max_depth,
        max_levels=max_levels,
        max_nodes=max_nodes,
        min_count=min_count,
        dropna=dropna,
        schema=schema,
    )
    cohort = df
    lineage = None
    if top_n_mode == "pre" and top_n is not None:
        cohort = _pre_cohort(
            df,
            active,
            top_n=top_n,
            top_n_per_parent=top_n_per_parent,
            dropna=dropna,
        )
        lineage = {
            "source_scope": "s2",
            "conditional": len(cohort) != len(df),
            "scope": census_result["scopes"][0],
        }
    grain_frame = cohort if top_n_applies_to == "both" else df
    grain_data = (
        grain(
            grain_frame,
            candidate_keys,
            dropna=dropna,
            scope_metadata=lineage if top_n_applies_to == "both" else None,
        ).to_dict()
        if candidate_keys is not None
        else {"status": "not_requested"}
    )
    pair_data = (
        pairs(
            cohort,
            active,
            dropna=dropna,
            include_absence=include_absence,
            reference_domains=reference_domains,
            pair_contexts=pair_contexts,
            max_absence_cells=max_absence_cells,
            max_contexts=max_contexts,
            max_pairs=max_pairs,
            scope_metadata=lineage,
        ).to_dict()
        if include_pairs
        else {"status": "not_requested"}
    )
    payload: dict[str, Any] = {
        "status": "empty" if len(df) == 0 else "computed",
        "source": _source(df),
        "sections": {
            "levels": level_result.to_dict(),
            "census": census_result.to_dict(),
            "grain": grain_data,
            "pairs": pair_data,
        },
        "warnings": [
            *level_result.payload.get("warnings", []),
            *census_result.payload.get("warnings", []),
        ],
    }
    return ExplorerResult("explore", payload)
