"""The profile of chosen dimensions: levels, census, grain and pairs together."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Unpack

import numpy as np
import pandas as pd

from .._runtime import operation
from ..result import Result
from ..typing import Runtime, SchemaRole
from .census import _complete, _preselect, census, levels
from .grain import KeySpec, grain
from .relations import pairs

if TYPE_CHECKING:
    from ..evidence import Scope


def _cohort(
    df: pd.DataFrame,
    dimensions: list[str],
    *,
    top_n: int,
    per_parent: bool,
    dropna: bool,
    scope: Scope | None,
    missing: Mapping[str, Iterable[Any]] | None,
    table_id: str,
) -> Scope:
    """The census pre-selection as a Scope, so other analyses share its rows."""
    from ..evidence import Scope, prepare_values

    frame, positions, encoded, base = prepare_values(
        df, dimensions, scope=scope, missing=missing, table_id=table_id
    )
    encoded_list = [encoded[c] for c in dimensions]
    eligible = _complete(encoded_list, len(frame)) if dropna else np.ones(len(frame), bool)
    mask, _ = _preselect(
        [codes for _, codes in encoded_list], np.flatnonzero(eligible), top_n, per_parent
    )
    return Scope(
        base["source"]["dataset_id"],
        tuple(positions[mask].tolist()),
        "census top_n cohort",
        scope.name if scope else None,
    )


@operation("profile")
def profile(
    df: pd.DataFrame,
    dimensions: Iterable[str],
    *,
    candidate_keys: Iterable[str | KeySpec | Mapping[str, Any]] | None = None,
    features: Iterable[str] | None = None,
    top_n: int | None = None,
    top_n_mode: Literal["pre", "post"] = "post",
    top_n_per_parent: bool = False,
    top_n_applies_to: Literal["census", "both"] = "census",
    min_retained_fraction: float = 0.01,
    max_depth: int | None = None,
    max_levels: int | None = 100,
    max_nodes: int | None = 10000,
    min_count: int = 1,
    dropna: bool = False,
    schema: dict[str, SchemaRole] | None = None,
    include_pairs: bool = True,
    include_absence: bool = False,
    reference_domains: Mapping[str, Iterable[Any]] | None = None,
    pair_contexts: Iterable[Mapping[str, Any]] | None = None,
    max_absence_cells: int | None = 1000,
    max_contexts: int | None = 32,
    max_pairs: int | None = 15,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Profile chosen dimensions: levels, census, optional grain, and pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    dimensions : iterable of str
        Nonempty ordered census dimensions; pairs use the first max_depth.
    candidate_keys : iterable of str, KeySpec or mapping, or None, optional
        Keys for a grain section; default None skips grain.
    features : iterable of str or None, optional
        Columns for levels; default None uses the dimensions.
    top_n, top_n_mode, top_n_per_parent, min_retained_fraction, max_depth,
    max_levels, max_nodes, min_count, schema : optional
        Census options (see census); levels share top_n, max_levels, min_count
        and schema.
    top_n_applies_to : {'census', 'both'}, optional
        With top_n_mode='pre', pairs always analyze the census pre-selection;
        'both' makes grain analyze it too. Requires top_n and pre mode.
    dropna : bool, optional
        Exclude missing values in every section, each on its own complete cases.
    include_pairs : bool, optional
        Compute the pairs section; default True.
    include_absence, reference_domains, pair_contexts, max_absence_cells,
    max_contexts, max_pairs : optional
        Pair options (see pairs); include_absence requires include_pairs.
    scope, missing, table_id
        Source context shared by every section.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'profile': ``sections`` holding the levels, census, grain and pairs
        results (unrequested ones with status 'not_requested'), and the levels
        and census warnings. A pre-selection is shared as a Scope named
        "census top_n cohort".

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"site": ["A", "A", "B"], "visit": [1, 2, 1]})
    >>> fw.profile(df, ["site", "visit"], include_pairs=False).section("census").kind
    'census'
    """
    from ..evidence import columns

    selected = columns(df, dimensions)
    if not selected:
        raise ValueError("dimensions must contain at least one column")
    active = selected[:max_depth] if max_depth is not None else selected
    if top_n_applies_to not in {"census", "both"}:
        raise ValueError("top_n_applies_to must be 'census' or 'both'")
    if include_absence and not include_pairs:
        raise ValueError("include_absence=True requires include_pairs=True")
    if top_n_applies_to == "both" and (top_n_mode != "pre" or top_n is None):
        raise ValueError("top_n_applies_to='both' requires pre mode with top_n")
    context = {"missing": missing, "table_id": table_id}
    level_result = levels(
        df,
        features=features if features is not None else selected,
        top_n=top_n,
        max_levels=max_levels,
        min_count=min_count,
        dropna=dropna,
        schema=schema,
        scope=scope,
        **context,
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
        scope=scope,
        **context,
    )
    cohort = scope
    if top_n_mode == "pre" and top_n is not None:
        cohort = _cohort(
            df,
            active,
            top_n=top_n,
            per_parent=top_n_per_parent,
            dropna=dropna,
            scope=scope,
            **context,
        )
    sections = {
        "levels": level_result.to_dict(),
        "census": census_result.to_dict(),
        "grain": grain(
            df,
            candidate_keys,
            dropna=dropna,
            scope=cohort if top_n_applies_to == "both" else scope,
            **context,
        ).to_dict()
        if candidate_keys is not None
        else {"status": "not_requested"},
        "pairs": pairs(
            df,
            active,
            dropna=dropna,
            include_absence=include_absence,
            reference_domains=reference_domains,
            pair_contexts=pair_contexts,
            max_absence_cells=max_absence_cells,
            max_contexts=max_contexts,
            max_pairs=max_pairs,
            scope=cohort,
            **context,
        ).to_dict()
        if include_pairs
        else {"status": "not_requested"},
    }
    payload = {
        key: census_result.payload[key]
        for key in ("status", "source", "scope", "missing_convention")
    }
    payload["sections"] = sections
    payload["warnings"] = [*level_result["warnings"], *census_result["warnings"]]
    return Result("profile", payload)
