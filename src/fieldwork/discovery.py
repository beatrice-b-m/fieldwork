"""Bounded determinant discovery with exact and modal-repair evidence."""

from __future__ import annotations

from itertools import combinations, islice
from math import comb

import numpy as np

from ._explore import KeySpec, grain
from ._explore._kernels import FDCache, first_indices, group_ids, modal_groups
from ._explore.encoding import MissingCode, normalize_scalar
from ._runtime import checkpoint, operation, phase
from .evidence import (
    bounded_rows,
    columns,
    context_statement,
    contextual_result,
    finding,
    limit,
    prepare,
    result,
)


@operation("dependencies")
def discover_dependencies(
    df,
    *,
    features=None,
    max_key_size=2,
    max_candidates=100,
    min_accuracy=0.95,
    by=None,
    max_contexts=32,
    dropna=True,
    scope=None,
    missing=None,
    table_id="table",
    example_limit=5,
    include_grain=True,
    max_grain_views=None,
    max_dependency_tests=None,
):
    """Evaluate determinants in size/column order; accuracy is modal repair accuracy."""
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
    with phase("dependency tests", total_tests, "tests") as progress:
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
            }
            base["candidates"].append(candidate)
            for context, population in partitions:
                for target in selected:
                    if tests >= total_tests:
                        break
                    checkpoint()
                    if target in key:
                        continue
                    tests += 1
                    eligible = (
                        population[key_mask[population] & present[target][population]]
                        if dropna
                        else population
                    )
                    grouped, sizes, modes, maxima, distinct = modal_groups(
                        key_ids[eligible], codes[target][eligible]
                    )
                    violating_mask = distinct > 1
                    violating = int(violating_mask.sum())
                    repair = int((sizes - maxima).sum())
                    repeated = int(np.count_nonzero(sizes > 1))
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
                    if context is None and record["exact"]:
                        candidate["determines"].append(target)
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
                    progress.advance(detail=f"{', '.join(key)} → {target}")
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
    with phase("grain views", len(chosen_anchors), "views") as progress:
        for _, mask in chosen_anchors:
            members = [i for i, eligible in enumerate(candidate_masks) if np.all(eligible[mask])]
            analysis = grain(
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
            progress.advance()
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
