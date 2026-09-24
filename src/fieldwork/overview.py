"""The automatic overview: independent sections and the ranked leads they yield."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Unpack

import pandas as pd

from ._explore.encoding import labelled
from ._runtime import operation, phase
from .availability import missingness
from .discovery import discover_dependencies
from .evidence import prepare
from .families import feature_network
from .leads import rank
from .navigation import suggest_paths
from .patterns import value_patterns
from .result import Result, overview_findings
from .typing import Runtime, Section, SectionOptions

if TYPE_CHECKING:
    from .evidence import Scope

SECTIONS = ("missingness", "dependencies", "paths", "value_patterns")
_ORDER = ("paths", "missingness", "dependencies", "value_patterns")
_OPERATIONS = {
    "missingness": missingness,
    "dependencies": discover_dependencies,
    "paths": suggest_paths,
    "value_patterns": value_patterns,
}
# The overview searches less than the standalone analyses do.
_DEFAULTS = {
    "missingness": {},
    "dependencies": {"max_key_size": 1, "limits": {"max_candidates": 20}},
    "paths": {},
    "value_patterns": {"limits": {"max_pairs": 20}},
}
_CONTEXT = {"scope", "missing", "table_id", "progress", "cancel", "timeout", "safe_errors"}


@operation("overview")
def explore(
    df: pd.DataFrame,
    *,
    features: Iterable[str] | None = None,
    by: Iterable[str] | None = None,
    entity: str | Iterable[str] | None = None,
    unit: Literal["rows", "entities"] = "rows",
    entity_presence: Literal["any", "all"] = "any",
    sections: Iterable[Section] | None = None,
    options: SectionOptions | None = None,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Survey a table: availability, dependencies, census paths and value patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    features : iterable of str or None, optional
        Columns every section analyzes; default None selects every column,
        skipping (and listing) columns with unsupported values.
    by : iterable of str or None, optional
        Context columns for availability, dependencies and value patterns.
    entity, unit, entity_presence : optional
        Entity keys and counting unit for the availability section; see
        missingness. Other sections always count rows.
    sections : iterable of str or None, optional
        Sections to run, from 'missingness', 'dependencies', 'paths' and
        'value_patterns'; default None runs all four.
    options : SectionOptions or None, optional
        Per-section options for requested sections, such as
        ``{"dependencies": {"max_key_size": 2}}``; not source context or runtime
        controls. Defaults: single-column keys and 20 dependency candidates, 20
        value-pattern pairs; a section's ``limits`` merge with them key by key.
    scope, missing, table_id
        Source context shared by every analysis and section.
    **runtime : Unpack[Runtime]
        Optional runtime controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'overview': section results under ``sections`` (unrequested ones
        'not_requested'), ranked ``leads`` (resolved by ``Result.findings``, IDs
        ``f0``, ``f1``, ...) and a ``feature_network``.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"site": ["A", "A", "B"], "visit": [1, 2, None]})
    >>> overview = fw.explore(df, sections=["missingness"])
    >>> overview.findings[0]["statement"]
    'visit: populated values'
    """
    requested = _requested(sections)
    shared = {"features": features, "by": by}
    configs = _configs(
        df,
        requested,
        options or {},
        shared,
        {"entity": entity, "unit": unit, "entity_presence": entity_presence},
    )
    context = {"scope": scope, "missing": missing, "table_id": table_id}
    analyses = {
        name: _OPERATIONS[name](df, **configs[name], **context)
        for name in _ORDER
        if name in requested
    }
    base = prepare(df, features=[], **context)[-1]
    del base["findings"]
    base["parameters"] = {
        "features": None if features is None else list(features),
        "by": None if by is None else list(by),
        "entity": entity if entity is None or isinstance(entity, str) else list(entity),
        "unit": unit,
        "entity_presence": entity_presence,
        "sections": [name for name in SECTIONS if name in requested],
        "options": {name: dict(value) for name, value in (options or {}).items()},
    }
    return _assemble(df, base, analyses)


def _requested(sections: Iterable[str] | None) -> list[str]:
    requested = list(SECTIONS if sections is None else sections)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(SECTIONS):
        raise ValueError(f"sections must contain distinct names from {SECTIONS}")
    return requested


def _configs(df, requested, options, shared, availability) -> dict[str, dict[str, Any]]:
    """Each section's options, validated before any expensive work starts."""
    if options.keys() - set(requested):
        raise ValueError("options must refer to requested sections")
    configs = {}
    for name in requested:
        overrides = dict(options.get(name, {}))
        if _CONTEXT & overrides.keys() or any(key.startswith("_") for key in overrides):
            raise ValueError("options cannot override source context or runtime controls")
        config = {**_DEFAULTS[name], "features": shared["features"]}
        if name != "paths":
            config["by"] = shared["by"]
        if name == "missingness":
            config.update(availability)
        # Budgets merge key by key, so overriding one keeps the overview's others.
        limits = {**config.get("limits", {}), **(overrides.pop("limits", None) or {})}
        config.update(overrides)
        if limits:
            config["limits"] = limits
        inspect.signature(_OPERATIONS[name]).bind(df, **config)
        configs[name] = config
    return configs


def _assemble(df: pd.DataFrame, base: dict[str, Any], analyses: dict[str, Result]) -> Result:
    base["sections"] = {
        name: analyses[name].to_dict() if name in analyses else {"status": "not_requested"}
        for name in SECTIONS
    }
    skipped = {}
    for analysis in analyses.values():
        for record in analysis["skipped_features"]:
            skipped.setdefault(record["feature"], record)
    base["skipped_features"] = list(skipped.values())
    candidates = [
        {**record, "section": name}
        for name in SECTIONS
        if name in analyses
        for record in analyses[name]["findings"]
    ]
    # IDs follow lead rank, so f0 is the most promising finding.
    base["leads"] = [
        {
            "id": f"f{rank_index}",
            "section": record["section"],
            "finding_id": record["id"],
            "lead": record["lead"],
        }
        for rank_index, record in enumerate(rank(candidates, _constant_columns(df)))
    ]
    with phase("assembling overview"):
        base["feature_network"] = feature_network(overview_findings(base), base)
    return Result("overview", base)


def _constant_columns(df: pd.DataFrame) -> set[str]:
    frame, constant = labelled(df), set()
    for column in frame.columns:
        try:
            if frame[column].nunique(dropna=True) <= 1:
                constant.add(column)
        except TypeError:  # Unhashable cells: such columns are skipped anyway.
            pass
    return constant
