"""Bounded determinant discovery with exact and modal-repair evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations, islice
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from ._explore import KeySpec
from ._explore._kernels import FDCache, first_indices, group_ids, modal_groups
from ._explore.encoding import MissingCode, normalize_scalar
from ._explore.grain import _grain
from ._runtime import checkpoint, operation, phase
from .evidence import (
    InvestigationResult,
    Scope,
    bounded_rows,
    columns,
    context_statement,
    contextual_result,
    finding,
    limit,
    prepare,
    result,
)
from .progress import CancellationToken, Progress


@operation("dependencies")
def discover_dependencies(
    df: pd.DataFrame,
    *,
    features: Iterable[str] | None = None,
    max_key_size: int = 2,
    max_candidates: int = 100,
    min_accuracy: float = 0.95,
    by: Iterable[str] | None = None,
    max_contexts: int = 32,
    dropna: bool = True,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    example_limit: int = 5,
    include_grain: bool = True,
    max_grain_views: int | None = None,
    max_dependency_tests: int | None = None,
    progress: Progress = None,
    cancel: CancellationToken | None = None,
    timeout: float | None = None,
) -> InvestigationResult:
    """Find observed exact and approximate dependencies over bounded candidates.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation. Discovery requires unique string
        column names. Duplicate index labels are supported; source selections use
        integer row positions. Unsupported scalar objects raise TypeError.
    features : iterable of str or None, optional
        Unique column names to analyze, in requested order; default None selects
        all columns. Restricts analysis, not full-source identity validation.
    max_key_size : int, optional
        Positive maximum determinant size; default 2. Candidate combinations are
        visited in increasing size, then requested column order.
    max_candidates : int, optional
        Nonnegative maximum number of determinants; default 100. Zero searches
        none. Candidate summaries are computed even when the test budget is zero.
    min_accuracy : float, optional
        Minimum reported modal repair accuracy in [0, 1]; default 0.95. Accuracy
        is the sum of modal target counts per determinant group divided by
        evaluated rows. This threshold filters reports, not test work.
    by : iterable of str or None, optional
        Additional joint context analyses besides the global population; default
        None. Missing context values are categories.
    max_contexts : int, optional
        Nonnegative context-group budget in addition to global analysis; default
        32. Zero keeps only global analysis.
    dropna : bool, optional
        Default True uses complete cases for each determinant/target test. False
        treats native and declared missing values as one category. Different
        tests may therefore have different populations.
    scope : Scope or None, optional
        Source-bound population selection; default None uses all rows. The scope
        must match the ordered source. Fingerprinting still scans the full frame.
    missing : mapping or None, optional
        Additional missing sentinels per column; default None. Native missing
        values are always absent. Numeric sentinels match integer/float values
        numerically; booleans remain distinct. The source is not modified.
    table_id : str, optional
        Nonempty source label; default 'table'. Does not replace the fingerprint.
    example_limit : int, optional
        Nonnegative maximum saved example/exception source rows per finding side;
        default 5. Zero retains totals without row examples. This display limit
        does not restrict the population recovered by select or all_matches.
    include_grain : bool, optional
        Build exact foundation grain views when True (default). False skips graph
        work while retaining dependency tests and candidate summaries.
    max_grain_views : int or None, optional
        Nonnegative maximum supported-population graph views; default None permits
        all. Zero builds none. Independent of max_dependency_tests.
    max_dependency_tests : int or None, optional
        Nonnegative candidate/target/context test budget; default None tests all
        within other budgets. Zero skips tests. Does not cap foundation graph work.
    progress : bool or callable, optional
        Default None is silent; True uses the built-in display. A callback receives
        ProgressEvent objects synchronously. False is also silent. Callback errors
        propagate unchanged; do not mutate the frame from a callback.
    cancel : CancellationToken or None, optional
        Cooperative cancellation token; default None. A cancelled token raises
        AnalysisCancelled at the next checkpoint, with no partial result.
    timeout : float or None, optional
        Finite nonnegative seconds from call start; default None disables the
        deadline. Expiration raises AnalysisCancelled cooperatively, after the
        current pandas/NumPy work item returns, rather than at a hard deadline.

    Returns
    -------
    InvestigationResult
        Kind 'dependencies', with candidates, dependencies, conditional evidence,
        grain views, findings, and coverage recording omitted tests/views. The
        dependencies table retains every completed test, including those below
        min_accuracy. Candidate determines lists global exact targets;
        determines_with_repeated_support includes only those with repeated groups
        in that target's evaluated population. global_targets_tested/possible
        count completed/selected non-key global targets, independently of graphs.

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
    Search and display budgets never sample rows. Evidence records evaluated
    populations and omissions separately. Source identity covers ordered column
    labels, index labels, and all cell values (not dtype metadata); changing or
    reordering them invalidates inspection against saved findings.

    A functional dependency here is observed evidence, not a guarantee about
    future deliveries or causality. Singleton determinant groups satisfy exact
    mappings trivially. Untested relationships are not negative evidence.

    For each context, determinant_evaluated_rows counts complete determinant
    cases when dropna=True, otherwise all context rows (Q). target_observed_rows
    counts rows of Q with an observed target (O), and target_coverage is O/Q.
    evaluated_rows (E) is O when dropna=True, otherwise Q;
    target_missing_excluded_rows is Q-E. missing_excluded_rows still counts all
    exclusions from the context. With dropna=False, observed target coverage can
    be below one even though missing values participate in consistency tests.

    repeated_rows (R) counts rows in determinant groups of size at least two
    within E. repeat_coverage is R/E; repeat_modal_accuracy is 1-repair_rows/R.
    Undefined fractions are None (JSON null). Candidate repeated_rows instead
    describes the determinant population, before target exclusions. Measurements
    are row-counted and row-weighted, without resampling or entity aggregation.
    Repeated support describes observed consistency, not statistical reliability
    or a meaningful entity interpretation.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"site": ["A", "A", "B"], "region": ["N", "N", "S"]})
    >>> result = fw.discover_dependencies(df, max_key_size=1, include_grain=False)
    >>> result.kind
    'dependencies'
    """
    limit("max_key_size", max_key_size, minimum=1)
    limit("max_candidates", max_candidates)
    limit("max_contexts", max_contexts)
    limit("example_limit", example_limit)
    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be between zero and one")
    if not isinstance(include_grain, bool):
        raise TypeError("include_grain must be boolean")
    for name, value in (
        ("max_grain_views", max_grain_views),
        ("max_dependency_tests", max_dependency_tests),
    ):
        if value is not None:
            limit(name, value)
    selected = columns(df, features)
    contexts = columns(df, by or [])
    frame, positions, codes, present, base = prepare(
        df, scope=scope, missing=missing, table_id=table_id, features=[*selected, *contexts]
    )
    # With dropna=False, native missing and declared sentinels share one category.
    codes = {
        c: values if present[c].all() else np.where(present[c], values, -1)
        for c, values in codes.items()
    }
    candidates = list(
        islice(
            (
                key
                for size in range(1, min(max_key_size, len(selected)) + 1)
                for key in combinations(selected, size)
            ),
            max_candidates,
        )
    )
    partitions = [(None, np.arange(len(frame)))]
    context_ids = group_ids(codes[c] for c in contexts)
    context_count = int(context_ids.max()) + 1 if len(context_ids) else 0
    if contexts:
        for group in range(min(context_count, max_contexts)):
            rows = np.flatnonzero(context_ids == group)
            partitions.append(
                (
                    {
                        c: normalize_scalar(
                            frame[c].iloc[rows[0]] if present[c][rows[0]] else None
                        ).to_dict()
                        for c in contexts
                    },
                    np.array(rows, dtype=np.int64),
                )
            )
    base["candidates"], base["dependencies"] = [], []
    candidate_masks = []
    graph_cache = FDCache()
    tests = 0
    possible_tests = sum(len(selected) - len(key) for key in candidates) * len(partitions)
    total_tests = (
        min(possible_tests, max_dependency_tests)
        if max_dependency_tests is not None
        else possible_tests
    )
    with phase("dependency tests", total_tests, "tests") as tracker:
        for key in candidates:
            key_mask = np.ones(len(frame), dtype=bool)
            if dropna:
                for c in key:
                    key_mask &= present[c]
            candidate_masks.append(key_mask)
            key_ids = group_ids(codes[c] for c in key)
            _, key_groups = np.unique(key_ids[key_mask], return_counts=True)
            candidate = {
                "columns": list(key),
                "evaluated_rows": int(key_mask.sum()),
                "missing_excluded_rows": int((~key_mask).sum()),
                "groups": len(key_groups),
                "unique": bool(len(key_groups)) and bool(np.all(key_groups == 1)),
                "uniqueness": len(key_groups) / int(key_mask.sum()) if key_mask.any() else None,
                "repeated_groups": int(np.count_nonzero(key_groups > 1)),
                "repeated_rows": int(key_groups[key_groups > 1].sum()),
                "determines": [],
                "determines_with_repeated_support": [],
                "global_targets_tested": 0,
                "global_targets_possible": len(selected) - len(key),
            }
            base["candidates"].append(candidate)
            for context, population in partitions:
                determinant_eligible = population[key_mask[population]]
                q = len(determinant_eligible)
                for target in selected:
                    if tests >= total_tests:
                        break
                    checkpoint()
                    if target in key:
                        continue
                    tests += 1
                    observed = present[target][determinant_eligible]
                    observed_rows = int(observed.sum())
                    eligible = determinant_eligible[observed] if dropna else determinant_eligible
                    grouped, sizes, modes, maxima, distinct = modal_groups(
                        key_ids[eligible], codes[target][eligible]
                    )
                    violating_mask = distinct > 1
                    violating = int(violating_mask.sum())
                    repair = int((sizes - maxima).sum())
                    repeated = int(np.count_nonzero(sizes > 1))
                    repeated_rows = int(sizes[sizes > 1].sum())
                    affected = int(sizes[violating_mask].sum())
                    good = codes[target][eligible] == modes[grouped]
                    exception_groups = []
                    for group in first_indices(violating_mask, example_limit):
                        rows = eligible[first_indices(grouped == group, max(1, example_limit))]
                        exception_groups.append(
                            {
                                "key_values": {
                                    c: normalize_scalar(
                                        frame[c].iloc[rows[0]] if present[c][rows[0]] else None
                                    ).to_dict()
                                    for c in key
                                },
                                "rows": int(sizes[group]),
                                "distinct_targets": int(distinct[group]),
                                "positions": positions[rows[:example_limit]].tolist(),
                                "omitted_rows": max(0, int(sizes[group]) - example_limit),
                            }
                        )
                    n = len(eligible)
                    accuracy = 1 - repair / n if n else None
                    record = {
                        "determinant": list(key),
                        "target": target,
                        "context": context,
                        "exact": violating == 0 if n else None,
                        "modal_accuracy": accuracy,
                        "repair_rows": repair,
                        "evaluated_rows": n,
                        "determinant_evaluated_rows": q,
                        "target_observed_rows": observed_rows,
                        "target_coverage": observed_rows / q if q else None,
                        "target_missing_excluded_rows": q - n,
                        "repeated_rows": repeated_rows,
                        "repeat_coverage": repeated_rows / n if n else None,
                        "repeat_modal_accuracy": 1 - repair / repeated_rows
                        if repeated_rows
                        else None,
                        "missing_excluded_rows": len(population) - n,
                        "evaluated_groups": len(sizes),
                        "violating_groups": violating,
                        "affected_rows": affected,
                        "group_violation_rate": violating / len(sizes) if len(sizes) else None,
                        "repeated_groups": repeated,
                        "exception_groups": exception_groups,
                        "omitted_exception_groups": max(0, violating - len(exception_groups)),
                    }
                    if context is None:
                        graph_cache.global_records[(key, target, dropna)] = {
                            "evaluated_groups": len(sizes),
                            "violating_groups": violating,
                            "affected_rows": affected,
                            "singleton_groups": int(np.count_nonzero(sizes == 1)),
                            "evaluated_rows": n,
                        }
                    base["dependencies"].append(record)
                    if context is None:
                        candidate["global_targets_tested"] += 1
                        if record["exact"]:
                            candidate["determines"].append(target)
                            if repeated:
                                candidate["determines_with_repeated_support"].append(target)
                    if accuracy is not None and accuracy >= min_accuracy:
                        finding(
                            base,
                            "exact_dependency" if record["exact"] else "approximate_dependency",
                            f"{', '.join(key)} determines {target}"
                            + (" within " + context_statement(context) if context else ""),
                            list(dict.fromkeys([*key, target, *(context or {})])),
                            record,
                            bounded_rows(positions[eligible], good, example_limit),
                            exceptions=bounded_rows(positions[eligible], ~good, example_limit),
                            example_limit=example_limit,
                            structure={"context": context} if context is not None else {},
                            selector={
                                "operation": "dependency",
                                "determinant": list(key),
                                "target": target,
                                "context": context,
                                "dropna": dropna,
                            },
                        )
                    tracker.advance(detail=f"{', '.join(key)} → {target}")
    graph_frame = frame[selected] if include_grain and max_grain_views != 0 else None
    graph_codes = (
        {c: (MissingCode(-1 if not present[c].all() else None), codes[c]) for c in selected}
        if include_grain
        else {}
    )
    # Each supported candidate supplies a population anchor. Candidates with a
    # superset of those rows can be compared on that anchor without shrinking it.
    anchors = {}
    for i, mask in enumerate(candidate_masks):
        if mask.any():
            anchors.setdefault(np.packbits(mask).tobytes(), (i, mask))
    views = []
    ordered_anchors = sorted(anchors.values(), key=lambda item: (-int(item[1].sum()), item[0]))
    chosen_anchors = ordered_anchors[:max_grain_views] if include_grain else []
    with phase("grain views", len(chosen_anchors), "views") as tracker:
        for _, mask in chosen_anchors:
            members = [i for i, eligible in enumerate(candidate_masks) if np.all(eligible[mask])]
            analysis = _grain(
                graph_frame,
                [KeySpec(f"key{i}", candidates[i]) for i in members],
                dropna=dropna,
                _encoded=graph_codes,
                _cache=graph_cache,
            )
            views.append(
                {
                    "id": f"g{len(views)}",
                    "candidate_ids": [f"key{i}" for i in members],
                    "population": {
                        "input_rows": len(df),
                        "scope_rows": len(frame),
                        "evaluated_rows": int(mask.sum()),
                        "restriction_excluded_rows": len(df) - len(frame),
                        "missing_excluded_rows": int((~mask).sum()),
                        "positions": positions[mask].tolist(),
                        "rule": "complete_cases_of_candidate_components"
                        if dropna
                        else "missing_as_category",
                    },
                    "grain": contextual_result(analysis, df, base).to_dict(),
                }
            )
            tracker.advance()
    base["grain_views"] = views
    base["exact_grain"] = views[0]["grain"] if views else None
    base["graph_selection"] = {
        "strategy": "candidate_population_anchors_with_superset_candidates",
        "primary_view": views[0]["id"] if views else None,
        "excluded": [],
    }
    for i, candidate in enumerate(base["candidates"]):
        candidate["id"] = f"key{i}"
        candidate["graph_views"] = [v["id"] for v in views if f"key{i}" in v["candidate_ids"]]
        if not candidate["graph_views"]:
            base["graph_selection"]["excluded"].append(
                {
                    "candidate_id": f"key{i}",
                    "columns": candidate["columns"],
                    "reason": "no_evaluated_support"
                    if not candidate["evaluated_rows"]
                    else ("graph_not_requested" if not include_grain else "graph_view_budget"),
                }
            )
    base["coverage"] = {
        "candidate_space": sum(
            comb(len(selected), k) for k in range(1, min(max_key_size, len(selected)) + 1)
        ),
        "candidates_evaluated": len(candidates),
        "dependency_tests": tests,
        "contexts_total": context_count,
        "contexts_evaluated": len(partitions) - 1,
        "search_order": "determinant_size_then_input_column_order",
    }
    base["parameters"] = {
        "features": selected,
        "max_key_size": max_key_size,
        "max_candidates": max_candidates,
        "min_accuracy": min_accuracy,
        "by": contexts,
        "max_contexts": max_contexts,
        "dropna": dropna,
        "example_limit": example_limit,
    }
    if not include_grain or max_grain_views is not None:
        base["parameters"].update(include_grain=include_grain, max_grain_views=max_grain_views)
        base["graph_selection"].update(
            status="computed" if include_grain else "not_requested",
            views_possible=len(ordered_anchors),
            views_omitted=len(ordered_anchors) - len(views),
        )
    if max_dependency_tests is not None:
        base["parameters"]["max_dependency_tests"] = max_dependency_tests
        base["coverage"].update(
            dependency_tests_possible=possible_tests,
            dependency_tests_omitted=possible_tests - tests,
        )
    return result("dependencies", base)
