"""The profile of chosen dimensions: levels, census, grain and pairs together."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Unpack

import numpy as np
import pandas as pd

from .._runtime import operation
from ..result import Result
from ..typing import CensusOptions, PairOptions, Runtime
from .census import _complete, _preselect, levels
from .census import census as census_analysis
from .grain import KeySpec, grain
from .relations import pairs as pair_analysis

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
    census: CensusOptions | None = None,
    pairs: PairOptions | bool = True,
    top_n_applies_to: Literal["census", "both"] = "census",
    dropna: bool = False,
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
        Nonempty ordered census dimensions; pairs use the first ``max_depth``.
    candidate_keys : iterable of str, KeySpec or mapping, or None, optional
        Keys for a grain section; default None skips grain.
    features : iterable of str or None, optional
        Columns for levels; default None uses the dimensions.
    census : CensusOptions or None, optional
        Census options (see census) except ``dropna``; levels share ``top_n``,
        ``max_levels``, ``min_count`` and ``schema``.
    pairs : PairOptions or bool, optional
        Pair options (see pairs); True (default) uses the defaults and False
        skips the pairs section.
    top_n_applies_to : {'census', 'both'}, optional
        With ``top_n_mode='pre'``, pairs always analyze the census pre-selection;
        'both' makes grain analyze it too. Requires top_n and pre mode.
    dropna : bool, optional
        Exclude missing values in every section, each on its own complete cases.
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
    >>> fw.profile(df, ["site", "visit"], pairs=False).section("census").kind
    'census'
    """
    from ..evidence import columns

    selected = columns(df, dimensions)
    if not selected:
        raise ValueError("dimensions must contain at least one column")
    census_options = dict(census or {})
    if "dropna" in census_options:
        raise ValueError("Pass dropna to profile; it applies to every section")
    inspect.signature(census_analysis).bind(df, selected, **census_options)
    pair_options = {} if isinstance(pairs, bool) else dict(pairs)
    inspect.signature(pair_analysis).bind(df, selected, **pair_options)
    top_n, max_depth = census_options.get("top_n"), census_options.get("max_depth")
    pre = census_options.get("top_n_mode", "post") == "pre" and top_n is not None
    if top_n_applies_to not in {"census", "both"}:
        raise ValueError("top_n_applies_to must be 'census' or 'both'")
    if top_n_applies_to == "both" and not pre:
        raise ValueError("top_n_applies_to='both' requires pre mode with top_n")
    active = selected[:max_depth] if max_depth is not None else selected
    context = {"missing": missing, "table_id": table_id}
    shared = ("top_n", "max_levels", "min_count", "schema")
    level_result = levels(
        df,
        features=features if features is not None else selected,
        dropna=dropna,
        scope=scope,
        **{k: census_options[k] for k in shared if k in census_options},
        **context,
    )
    census_result = census_analysis(
        df, selected, **census_options, dropna=dropna, scope=scope, **context
    )
    cohort = scope
    if pre:
        cohort = _cohort(
            df,
            active,
            top_n=top_n,
            per_parent=census_options.get("top_n_per_parent", False),
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
        "pairs": pair_analysis(
            df, active, **pair_options, dropna=dropna, scope=cohort, **context
        ).to_dict()
        if pairs is not False
        else {"status": "not_requested"},
    }
    payload = {
        key: census_result.payload[key]
        for key in ("status", "source", "scope", "missing_convention")
    }
    payload["sections"] = sections
    payload["warnings"] = [*level_result["warnings"], *census_result["warnings"]]
    return Result("profile", payload)
