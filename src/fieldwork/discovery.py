"""Bounded determinant discovery with exact and modal-repair evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations, islice
from math import comb
from typing import Any, Unpack

import numpy as np
import pandas as pd

from ._explore._kernels import FDCache, first_indices, group_ids, modal_groups
from ._explore.encoding import cell
from ._explore.grain import Encoded, KeySpec, evaluate
from ._runtime import checkpoint, operation, phase
from .evidence import (
    Scope,
    analyzable,
    bounded_rows,
    budgets,
    columns,
    context_statement,
    finding,
    limit,
    prepare,
    result,
    selection,
)
from .result import Result
from .typing import DependencyLimits, Runtime

_LIMITS = {
    "max_candidates": 100,
    "max_contexts": 32,
    "max_dependency_tests": None,
    "max_grain_views": None,
    "example_limit": 5,
}


@operation("dependencies")
def discover_dependencies(
    df: pd.DataFrame,
    *,
    features: Iterable[str] | None = None,
    max_key_size: int = 2,
    min_accuracy: float = 0.95,
    by: Iterable[str] | None = None,
    dropna: bool = True,
    include_grain: bool = True,
    limits: DependencyLimits | None = None,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Find observed exact and approximate functional dependencies among columns.

    Each candidate determinant is tested against every other column. Dependencies
    describe this delivery, not a guarantee; populations and repeated support are
    defined in docs/algorithms.md.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    features : iterable of str or None, optional
        Columns to use as determinants and targets; default None selects every
        column, skipping (and listing) columns with unsupported values.
    max_key_size : int, optional
        Largest determinant; default 2.
    min_accuracy : float, optional
        Minimum modal repair accuracy reported as a finding; default 0.95. Every
        completed test is kept in ``dependencies``.
    by : iterable of str or None, optional
        Context columns; tests are repeated within each joint context value.
    dropna : bool, optional
        True (default) tests each pair on rows where determinant and target are
        present; False treats missing values (and sentinels) as a category.
    include_grain : bool, optional
        Build grain graphs of the candidates; default True.
    limits : DependencyLimits or None, optional
        Budgets: ``max_candidates`` (100), ``max_contexts`` (32),
        ``max_dependency_tests`` and ``max_grain_views`` (unbounded) and
        ``example_limit`` (5). Omitted work is reported in ``coverage``.
    missing, scope, table_id
        Source context shared by every analysis.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'dependencies': ``candidates`` (with the targets each determines
        exactly), ``dependencies`` (every completed test), ``grain_views``,
        ``coverage`` and findings.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"site": ["A", "A", "B"], "region": ["N", "N", "S"]})
    >>> result = fw.discover_dependencies(df, max_key_size=1, include_grain=False)
    >>> result["candidates"][0]["determines"]
    ['region']
    """
    limit("max_key_size", max_key_size, minimum=1)
    budget = budgets(limits, _LIMITS)
    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be between zero and one")
    if not isinstance(include_grain, bool):
        raise TypeError("include_grain must be boolean")
    selected = columns(df, features)
    contexts = columns(df, by or [])
    frame, positions, codes, present, base = prepare(
        df,
        scope=scope,
        missing=missing,
        table_id=table_id,
        features=[*selected, *contexts],
        optional=selected if features is None else (),
    )
    search = _Search(
        frame,
        positions,
        # With dropna=False, native missing and declared sentinels share one category.
        {c: v if present[c].all() else np.where(present[c], v, -1) for c, v in codes.items()},
        present,
        base,
        analyzable(selected, base),
        dropna,
        min_accuracy,
        budget["example_limit"],
    )
    candidates = list(
        islice(
            (
                key
                for size in range(1, min(max_key_size, len(search.selected)) + 1)
                for key in combinations(search.selected, size)
            ),
            budget["max_candidates"],
        )
    )
    partitions, context_count = _partitions(search, contexts, budget["max_contexts"])
    possible = sum(len(search.selected) - len(key) for key in candidates) * len(partitions)
    max_tests = budget["max_dependency_tests"]
    search.budget = possible if max_tests is None else min(possible, max_tests)
    masks = _test_candidates(search, candidates, partitions)
    anchors = _anchors(masks)
    views = _grain_views(
        search, candidates, masks, anchors[: budget["max_grain_views"]] if include_grain else []
    )
    base["grain_views"] = views
    base["graph_selection"] = _graph_selection(base["candidates"], views, include_grain)
    n = len(search.selected)
    base["coverage"] = {
        "candidate_space": sum(comb(n, k) for k in range(1, min(max_key_size, n) + 1)),
        "candidates_evaluated": len(candidates),
        "dependency_tests": search.tests,
        "contexts_total": context_count,
        "contexts_evaluated": len(partitions) - 1,
        "search_order": "determinant_size_then_input_column_order",
    }
    base["parameters"] = {
        "features": search.selected,
        "max_key_size": max_key_size,
        "min_accuracy": min_accuracy,
        "by": contexts,
        "dropna": dropna,
        "include_grain": include_grain,
        "limits": budget,
    }
    if not include_grain or budget["max_grain_views"] is not None:
        base["graph_selection"].update(
            status="computed" if include_grain else "not_requested",
            views_possible=len(anchors),
            views_omitted=len(anchors) - len(views),
        )
    if max_tests is not None:
        base["coverage"].update(
            dependency_tests_possible=possible, dependency_tests_omitted=possible - search.tests
        )
    return result("dependencies", base)


@dataclass
class _Search:
    """One dependency search: prepared rows, settings, and the test budget."""

    frame: pd.DataFrame
    positions: np.ndarray
    codes: dict[str, np.ndarray]
    present: dict[str, np.ndarray]
    base: dict[str, Any]
    selected: list[str]
    dropna: bool
    min_accuracy: float
    example_limit: int
    budget: int = 0
    tests: int = 0
    cache: FDCache = field(default_factory=FDCache)

    def cell(self, column: str, row: int) -> Any:
        return cell(self.frame, column, row, self.present[column][row])


def _partitions(search: _Search, contexts: list[str], max_contexts: int):
    """The global population, then each joint context (missing is a category)."""
    partitions = [(None, np.arange(len(search.frame)))]
    context_ids = group_ids(search.codes[c] for c in contexts)
    count = int(context_ids.max()) + 1 if len(context_ids) else 0
    for group in range(min(count, max_contexts) if contexts else 0):
        rows = np.flatnonzero(context_ids == group)
        partitions.append(({c: search.cell(c, rows[0]) for c in contexts}, rows.astype(np.int64)))
    return partitions, count


def _test_candidates(search: _Search, candidates, partitions) -> list[np.ndarray]:
    """Summarize each candidate and test it against every target in every population.

    Returns each candidate's complete-case mask.
    """
    search.base["candidates"], search.base["dependencies"] = [], []
    masks = []
    with phase("dependency tests", search.budget, "tests") as tracker:
        for key in candidates:
            key_mask = np.ones(len(search.frame), dtype=bool)
            for c in key if search.dropna else ():
                key_mask &= search.present[c]
            masks.append(key_mask)
            key_ids = group_ids(search.codes[c] for c in key)
            candidate = _candidate(search, key, key_mask, key_ids)
            search.base["candidates"].append(candidate)
            for context, population in partitions:
                eligible = population[key_mask[population]]
                for target in search.selected:
                    if search.tests >= search.budget:
                        break
                    checkpoint()
                    if target in key:
                        continue
                    search.tests += 1
                    record = _test(search, key, key_ids, target, context, population, eligible)
                    if context is None:
                        _record_global(candidate, record, target)
                    tracker.advance(detail=f"{', '.join(key)} → {target}")
    return masks


def _candidate(search: _Search, key, key_mask, key_ids) -> dict[str, Any]:
    _, groups = np.unique(key_ids[key_mask], return_counts=True)
    evaluated = int(key_mask.sum())
    return {
        "columns": list(key),
        "evaluated_rows": evaluated,
        "missing_excluded_rows": int((~key_mask).sum()),
        "groups": len(groups),
        "unique": bool(len(groups)) and bool(np.all(groups == 1)),
        "uniqueness": len(groups) / evaluated if evaluated else None,
        "repeated_groups": int(np.count_nonzero(groups > 1)),
        "repeated_rows": int(groups[groups > 1].sum()),
        "determines": [],
        "determines_with_repeated_support": [],
        "global_targets_tested": 0,
        "global_targets_possible": len(search.selected) - len(key),
    }


def _record_global(candidate: dict[str, Any], record: dict[str, Any], target: str) -> None:
    """Only global exactness enters a candidate's determined targets."""
    candidate["global_targets_tested"] += 1
    if record["exact"]:
        candidate["determines"].append(target)
        if record["repeated_groups"]:
            candidate["determines_with_repeated_support"].append(target)


def _test(search: _Search, key, key_ids, target, context, population, determinant_eligible):
    """One key -> target test in one population; a finding when accurate enough."""
    observed = search.present[target][determinant_eligible]
    eligible = determinant_eligible[observed] if search.dropna else determinant_eligible
    targets = search.codes[target][eligible]
    grouped, sizes, modes, maxima, distinct = modal_groups(key_ids[eligible], targets)
    violating = distinct > 1
    repair = int((sizes - maxima).sum())
    repeated_rows = int(sizes[sizes > 1].sum())
    n, q = len(eligible), len(determinant_eligible)
    exceptions = _exception_groups(search, key, eligible, grouped, sizes, distinct)
    record = {
        "determinant": list(key),
        "target": target,
        "context": context,
        "exact": not violating.any() if n else None,
        "modal_accuracy": 1 - repair / n if n else None,
        "repair_rows": repair,
        "evaluated_rows": n,
        "determinant_evaluated_rows": q,
        "target_observed_rows": int(observed.sum()),
        "target_coverage": int(observed.sum()) / q if q else None,
        "target_missing_excluded_rows": q - n,
        "repeated_rows": repeated_rows,
        "repeat_coverage": repeated_rows / n if n else None,
        "repeat_modal_accuracy": 1 - repair / repeated_rows if repeated_rows else None,
        "missing_excluded_rows": len(population) - n,
        "evaluated_groups": len(sizes),
        "violating_groups": int(violating.sum()),
        "affected_rows": int(sizes[violating].sum()),
        "group_violation_rate": int(violating.sum()) / len(sizes) if len(sizes) else None,
        "repeated_groups": int(np.count_nonzero(sizes > 1)),
        "exception_groups": exceptions,
        "omitted_exception_groups": max(0, int(violating.sum()) - len(exceptions)),
    }
    if context is None:
        # Grain views reuse global tests instead of regrouping.
        search.cache.put(
            (key, target, search.dropna),
            {
                "evaluated_groups": len(sizes),
                "violating_groups": record["violating_groups"],
                "affected_rows": record["affected_rows"],
                "singleton_groups": int(np.count_nonzero(sizes == 1)),
                "evaluated_rows": n,
            },
        )
    search.base["dependencies"].append(record)
    accuracy = record["modal_accuracy"]
    if accuracy is not None and accuracy >= search.min_accuracy:
        _finding(search, record, eligible, targets == modes[grouped])
    return record


def _exception_groups(search: _Search, key, eligible, grouped, sizes, distinct) -> list[dict]:
    """The first violating groups, with bounded source positions."""
    limit, groups = search.example_limit, []
    for group in first_indices(distinct > 1, limit):
        rows = eligible[first_indices(grouped == group, max(1, limit))]
        groups.append(
            {
                "key_values": {c: search.cell(c, rows[0]) for c in key},
                "rows": int(sizes[group]),
                "distinct_targets": int(distinct[group]),
                "positions": search.positions[rows[:limit]].tolist(),
                "omitted_rows": max(0, int(sizes[group]) - limit),
            }
        )
    return groups


def _finding(search: _Search, record, eligible, good) -> None:
    key, target, context = record["determinant"], record["target"], record["context"]
    positions, limit = search.positions[eligible], search.example_limit
    finding(
        search.base,
        "exact_dependency" if record["exact"] else "approximate_dependency",
        f"{', '.join(key)} determines {target}"
        + (" within " + context_statement(context) if context else ""),
        list(dict.fromkeys([*key, target, *(context or {})])),
        # The finding samples its own exceptions; the test keeps its groups once.
        {k: v for k, v in record.items() if "exception_groups" not in k},
        bounded_rows(positions, good, limit),
        exceptions=bounded_rows(positions, ~good, limit),
        example_limit=limit,
        structure={"context": context} if context is not None else {},
        selector={
            "operation": "dependency",
            "determinant": list(key),
            "target": target,
            "context": context,
            "dropna": search.dropna,
        },
    )


def _anchors(masks: list[np.ndarray]) -> list[tuple[int, np.ndarray]]:
    """One population per distinct supported candidate mask, largest first.

    Candidates whose complete cases include an anchor's rows can be compared on
    that anchor without shrinking it.
    """
    anchors = {}
    for i, mask in enumerate(masks):
        if mask.any():
            anchors.setdefault(np.packbits(mask).tobytes(), (i, mask))
    return sorted(anchors.values(), key=lambda item: (-int(item[1].sum()), item[0]))


def _grain_views(search: _Search, candidates, masks, anchors) -> list[dict[str, Any]]:
    encoded = Encoded(
        {c: search.codes[c] for c in search.selected},
        {c: None if search.present[c].all() else -1 for c in search.selected},
        search.cache,
    )
    views = []
    with phase("grain views", len(anchors), "views") as tracker:
        for anchor, mask in anchors:
            members = [i for i, eligible in enumerate(masks) if np.all(eligible[mask])]
            specs = tuple(KeySpec(f"key{i}", candidates[i]) for i in members)
            view = {k: search.base[k] for k in ("status", "source", "scope", "missing_convention")}
            view["parameters"] = {
                "candidate_keys": [{"name": s.name, "columns": list(s.columns)} for s in specs],
                "dropna": search.dropna,
            }
            size = len(search.frame)
            view.update(
                evaluate(size, search.selected, specs, dropna=search.dropna, encoded=encoded)
            )
            rows = int(mask.sum())
            views.append(
                {
                    "id": f"g{len(views)}",
                    "candidate_ids": [f"key{i}" for i in members],
                    "population": {
                        "evaluated_rows": rows,
                        "missing_excluded_rows": size - rows,
                        # The anchor's complete cases define the population, so
                        # only bounded examples are stored, not every position.
                        "anchor_candidate_id": f"key{anchor}",
                        "examples": selection(search.positions[mask], rows, search.example_limit),
                        "rule": "complete_cases_of_candidate_components"
                        if search.dropna
                        else "missing_as_category",
                    },
                    "grain": Result("grain", view).to_dict(),
                }
            )
            tracker.advance()
    return views


def _graph_selection(candidates, views, include_grain) -> dict[str, Any]:
    selection_record = {
        "strategy": "candidate_population_anchors_with_superset_candidates",
        "primary_view": views[0]["id"] if views else None,
        "excluded": [],
    }
    for i, candidate in enumerate(candidates):
        candidate["id"] = f"key{i}"
        candidate["graph_views"] = [v["id"] for v in views if f"key{i}" in v["candidate_ids"]]
        if candidate["graph_views"]:
            continue
        reason = "graph_view_budget" if include_grain else "graph_not_requested"
        selection_record["excluded"].append(
            {
                "candidate_id": f"key{i}",
                "columns": candidate["columns"],
                "reason": reason if candidate["evaluated_rows"] else "no_evaluated_support",
            }
        )
    return selection_record
