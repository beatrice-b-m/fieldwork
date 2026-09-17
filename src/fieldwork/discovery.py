"""Bounded determinant discovery with exact and modal-repair evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, islice
from math import comb

import numpy as np

from ._explore import KeySpec, grain
from ._explore.encoding import normalize_scalar
from .evidence import columns, finding, limit, prepare, result


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
):
    """Evaluate determinants in size/column order; accuracy is modal repair accuracy."""
    limit("max_key_size", max_key_size, minimum=1)
    limit("max_candidates", max_candidates)
    limit("max_contexts", max_contexts)
    limit("example_limit", example_limit)
    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be between zero and one")
    selected = columns(df, features)
    contexts = columns(df, by or [])
    frame, positions, codes, present, base = prepare(
        df, scope=scope, missing=missing, table_id=table_id
    )
    # With dropna=False, native missing and declared sentinels share one category.
    codes = {c: np.where(present[c], values, -1) for c, values in codes.items()}
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
    groups = defaultdict(list)
    if contexts:
        for i in range(len(frame)):
            groups[tuple(int(codes[c][i]) for c in contexts)].append(i)
        for rows in list(groups.values())[:max_contexts]:
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
    tests = 0
    for key in candidates:
        key_mask = np.ones(len(frame), dtype=bool)
        if dropna:
            for c in key:
                key_mask &= present[c]
        key_groups = Counter(tuple(int(codes[c][i]) for c in key) for i in np.flatnonzero(key_mask))
        candidate = {
            "columns": list(key),
            "evaluated_rows": int(key_mask.sum()),
            "missing_excluded_rows": int((~key_mask).sum()),
            "groups": len(key_groups),
            "unique": bool(key_groups) and all(n == 1 for n in key_groups.values()),
            "uniqueness": len(key_groups) / int(key_mask.sum()) if key_mask.any() else None,
            "repeated_groups": sum(n > 1 for n in key_groups.values()),
            "repeated_rows": sum(n for n in key_groups.values() if n > 1),
            "determines": [],
        }
        base["candidates"].append(candidate)
        for context, population in partitions:
            for target in selected:
                if target in key:
                    continue
                tests += 1
                eligible = (
                    population[key_mask[population] & present[target][population]]
                    if dropna
                    else population
                )
                grouped = defaultdict(list)
                for i in eligible:
                    grouped[tuple(int(codes[c][i]) for c in key)].append(int(i))
                exceptions, affected, typical, exception_groups = [], [], [], []
                repeated, violating, repair = 0, 0, 0
                for rows in grouped.values():
                    counts = Counter(int(codes[target][i]) for i in rows)
                    modal = min(counts, key=lambda v: (-counts[v], v))
                    good = [i for i in rows if codes[target][i] == modal]
                    bad = [i for i in rows if codes[target][i] != modal]
                    repeated += len(rows) > 1
                    if bad:
                        violating += 1
                        repair += len(bad)
                        affected.extend(rows)
                        if len(exception_groups) < example_limit:
                            exception_groups.append(
                                {
                                    "key_values": {
                                        c: normalize_scalar(
                                            frame[c].iloc[rows[0]] if present[c][rows[0]] else None
                                        ).to_dict()
                                        for c in key
                                    },
                                    "rows": len(rows),
                                    "distinct_targets": len(counts),
                                    "positions": [int(positions[i]) for i in rows[:example_limit]],
                                    "omitted_rows": max(0, len(rows) - example_limit),
                                }
                            )
                    exceptions.extend(bad)
                    typical.extend(good)
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
                    "evaluated_groups": len(grouped),
                    "violating_groups": violating,
                    "affected_rows": len(affected),
                    "group_violation_rate": violating / len(grouped) if grouped else None,
                    "repeated_groups": repeated,
                    "exception_groups": exception_groups,
                    "omitted_exception_groups": max(0, violating - len(exception_groups)),
                }
                base["dependencies"].append(record)
                if context is None and record["exact"]:
                    candidate["determines"].append(target)
                if accuracy is not None and accuracy >= min_accuracy:
                    finding(
                        base,
                        "exact_dependency" if record["exact"] else "approximate_dependency",
                        f"{', '.join(key)} determines {target}"
                        + (" within context" if context else ""),
                        [*key, target],
                        record,
                        positions[sorted(typical)],
                        exceptions=positions[sorted(exceptions)],
                        example_limit=example_limit,
                        selector={
                            "operation": "dependency",
                            "determinant": list(key),
                            "target": target,
                            "context": context,
                            "dropna": dropna,
                        },
                    )
    # The inherited graph checks population compatibility and collapses equivalent keys.
    # Replace sentinel values in a private frame only; source values remain untouched.
    graph_frame = frame[selected].copy()
    for c in selected:
        graph_frame[c] = graph_frame[c].astype(object).where(present[c], None)
    base["exact_grain"] = (
        grain(
            graph_frame,
            [KeySpec(f"key{i}", key) for i, key in enumerate(candidates)],
            dropna=dropna,
        ).to_dict()
        if candidates
        else None
    )
    base["coverage"] = {
        "candidate_space": sum(
            comb(len(selected), k) for k in range(1, min(max_key_size, len(selected)) + 1)
        ),
        "candidates_evaluated": len(candidates),
        "dependency_tests": tests,
        "contexts_total": len(groups),
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
    return result("dependencies", base)
