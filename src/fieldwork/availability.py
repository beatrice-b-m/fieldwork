"""Availability signatures, directional presence and entity-balanced evidence."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, islice
from math import comb

import numpy as np

from .evidence import columns, finding, limit, prepare, result


def missingness(
    df,
    *,
    features=None,
    by=None,
    entity=None,
    missing=None,
    scope=None,
    table_id="table",
    min_implication=0.9,
    min_similarity=0.8,
    max_pairs=200,
    max_signatures=50,
    max_contexts=32,
    example_limit=5,
):
    """Describe availability; entity counts use any-present per entity, never row weighting."""
    for name, value in [
        ("max_pairs", max_pairs),
        ("max_signatures", max_signatures),
        ("max_contexts", max_contexts),
        ("example_limit", example_limit),
    ]:
        limit(name, value)
    if not 0 <= min_implication <= 1 or not 0 <= min_similarity <= 1:
        raise ValueError("Thresholds must be between zero and one")
    selected = columns(df, features)
    contexts = columns(df, by or [])
    entities = columns(df, [entity] if isinstance(entity, str) else entity or [])
    frame, positions, codes, present, base = prepare(
        df, scope=scope, missing=missing, table_id=table_id
    )
    n = len(frame)
    codes = {c: np.where(present[c], values, -1) for c, values in codes.items()}
    base["parameters"] = {
        "features": selected,
        "by": contexts,
        "entity": entities,
        "min_implication": min_implication,
        "min_similarity": min_similarity,
        "max_pairs": max_pairs,
        "max_signatures": max_signatures,
        "max_contexts": max_contexts,
        "example_limit": example_limit,
    }
    base["availability"] = []
    for c in selected:
        mask = present[c]
        metrics = {
            "populated": int(mask.sum()),
            "missing": int((~mask).sum()),
            "denominator": n,
            "populated_fraction": float(mask.mean()) if n else None,
        }
        base["availability"].append({"feature": c, **metrics})
        finding(
            base,
            "availability",
            f"{c}: populated values",
            [c],
            metrics,
            positions[mask],
            exceptions=positions[~mask],
            example_limit=example_limit,
            selector={"operation": "presence", "feature": c},
        )
    signatures = defaultdict(list)
    for i in range(n):
        signatures[tuple(bool(present[c][i]) for c in selected)].append(i)
    ordered = sorted(signatures.items(), key=lambda item: (-len(item[1]), item[0]))
    base["signatures"] = []
    for signature, rows in ordered[:max_signatures]:
        populated = [c for c, value in zip(selected, signature) if value]
        base["signatures"].append(
            {
                "present": populated,
                "absent": [c for c in selected if c not in populated],
                "count": len(rows),
                "denominator": n,
            }
        )
    groups = defaultdict(list)
    for c in selected:
        groups[present[c].tobytes()].append(c)
    base["families"] = []
    for group in groups.values():
        if len(group) > 1:
            base["families"].append(
                {
                    "features": group,
                    "evidence": ["identical_availability"],
                    "populated": int(present[group[0]].sum()),
                }
            )
            finding(
                base,
                "availability_family",
                "Same availability: " + ", ".join(group),
                group,
                {"denominator": n, "populated": int(present[group[0]].sum())},
                positions[present[group[0]]],
                example_limit=example_limit,
            )
    evaluated = 0
    for a, b in islice(combinations(selected, 2), max_pairs):
        evaluated += 1
        x, y = present[a], present[b]
        both, union = int((x & y).sum()), int((x | y).sum())
        neither = int((~x & ~y).sum())
        similarity = both / union if union else None
        metrics = {
            "both_present": both,
            "either_present": union,
            "both_absent": neither,
            "denominator": n,
            "presence_jaccard": similarity,
            "agreement": float((x == y).mean()) if n else None,
        }
        if similarity is not None and similarity >= min_similarity and not np.array_equal(x, y):
            finding(
                base,
                "similar_availability",
                f"{a} and {b} have similar presence",
                [a, b],
                metrics,
                positions[x & y],
                exceptions=positions[x != y],
                example_limit=example_limit,
            )
        if both == 0 and x.any() and y.any():
            finding(
                base,
                "mutually_exclusive",
                f"{a} and {b} are mutually exclusive",
                [a, b],
                metrics,
                positions[x | y],
                example_limit=example_limit,
            )
        for source, target, first, second in [(a, b, x, y), (b, a, y, x)]:
            denominator = int(first.sum())
            rate = both / denominator if denominator else None
            if rate is not None and rate >= min_implication:
                finding(
                    base,
                    "presence_implication",
                    f"{source} populated implies {target} populated",
                    [source, target],
                    {
                        **metrics,
                        "antecedent_populated": denominator,
                        "conditional_presence": rate,
                        "exception_rate": 1 - rate,
                        "baseline_presence": float(second.mean()) if n else None,
                    },
                    positions[first & second],
                    exceptions=positions[first & ~second],
                    example_limit=example_limit,
                    selector={
                        "operation": "presence_implication",
                        "source": source,
                        "target": target,
                    },
                )
    context_groups = defaultdict(list)
    if contexts:
        for i in range(n):
            context_groups[tuple(int(codes[c][i]) for c in contexts)].append(i)
    base["contexts"] = []
    for _, rows in list(context_groups.items())[:max_contexts]:
        from ._explore.encoding import normalize_scalar

        base["contexts"].append(
            {
                "values": {
                    c: normalize_scalar(
                        frame[c].iloc[rows[0]] if present[c][rows[0]] else None
                    ).to_dict()
                    for c in contexts
                },
                "rows": len(rows),
                "availability": [
                    {
                        "feature": c,
                        "populated": int(present[c][rows].sum()),
                        "denominator": len(rows),
                    }
                    for c in selected
                ],
            }
        )
    base["entities"] = []
    eligible = np.ones(n, dtype=bool)
    for c in entities:
        eligible &= present[c]
    entity_groups = defaultdict(list)
    if entities:
        for i in np.flatnonzero(eligible):
            entity_groups[tuple(int(codes[c][i]) for c in entities)].append(i)
        for c in selected:
            counts = [(int(present[c][rows].sum()), len(rows)) for rows in entity_groups.values()]
            base["entities"].append(
                {
                    "feature": c,
                    "denominator": len(counts),
                    "any": sum(k > 0 for k, size in counts),
                    "all": sum(k == size for k, size in counts),
                    "one": sum(k == 1 for k, size in counts),
                    "some": sum(0 < k < size for k, size in counts),
                    "none": sum(k == 0 for k, size in counts),
                }
            )
    base["coverage"] = {
        "pair_candidates": comb(len(selected), 2),
        "pairs_evaluated": evaluated,
        "signatures_total": len(signatures),
        "signatures_shown": min(len(signatures), max_signatures),
        "signature_omitted_rows": sum(len(rows) for _, rows in ordered[max_signatures:]),
        "contexts_total": len(context_groups),
        "contexts_shown": min(len(context_groups), max_contexts),
        "entity_missing_key_rows": int((~eligible).sum()) if entities else 0,
    }
    return result("missingness", base)
