"""Availability signatures, directional presence and entity-balanced evidence."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, islice
from math import comb

import numpy as np

from ._explore.encoding import normalize_scalar
from .evidence import columns, context_statement, finding, limit, prepare, result


def missingness(
    df,
    *,
    features=None,
    by=None,
    entity=None,
    unit="rows",
    entity_presence="any",
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
    """Count rows or equally weighted entities using explicit any/all presence aggregation."""
    for name, value in [
        ("max_pairs", max_pairs),
        ("max_signatures", max_signatures),
        ("max_contexts", max_contexts),
        ("example_limit", example_limit),
    ]:
        limit(name, value)
    if not 0 <= min_implication <= 1 or not 0 <= min_similarity <= 1:
        raise ValueError("Thresholds must be between zero and one")
    if unit not in {"rows", "entities"} or entity_presence not in {"any", "all"}:
        raise ValueError("unit must be rows or entities; entity_presence must be any or all")
    selected = columns(df, features)
    contexts = columns(df, by or [])
    entities = columns(df, [entity] if isinstance(entity, str) else entity or [])
    if unit == "entities" and not entities:
        raise ValueError("unit='entities' requires entity keys")
    if unit == "rows" and entity_presence != "any":
        raise ValueError("entity_presence applies only to unit='entities'")
    frame, positions, codes, present, base = prepare(
        df, scope=scope, missing=missing, table_id=table_id
    )
    codes = {c: np.where(present[c], values, -1) for c, values in codes.items()}
    eligible = np.ones(len(frame), dtype=bool)
    for c in entities:
        eligible &= present[c]
    entity_groups = defaultdict(list)
    if entities:
        for i in np.flatnonzero(eligible):
            entity_groups[tuple(int(codes[c][i]) for c in entities)].append(int(i))

    def units_for(rows):
        if unit == "rows":
            return [[int(i)] for i in rows]
        groups = defaultdict(list)
        for i in rows:
            if eligible[i]:
                groups[tuple(int(codes[c][i]) for c in entities)].append(int(i))
        return list(groups.values())

    def masks_for(units):
        aggregate = np.any if entity_presence == "any" else np.all
        return {
            c: np.array([aggregate(present[c][rows]) for rows in units], dtype=bool)
            for c in selected
        }

    units = units_for(range(len(frame)))
    masks = masks_for(units)
    n = len(units)

    def source_rows(indices):
        return positions[sorted(i for j in indices for i in units[j])]

    def emit(
        kind,
        statement,
        features,
        metrics,
        mask,
        *,
        exception_mask=None,
        structure=None,
        selector=None,
    ):
        return finding(
            base,
            kind,
            statement,
            features,
            metrics,
            source_rows(np.flatnonzero(mask)),
            exceptions=source_rows(np.flatnonzero(exception_mask))
            if exception_mask is not None
            else (),
            example_limit=example_limit,
            unit=unit,
            structure=structure,
            selector=selector,
        )

    base["parameters"] = {
        "features": selected,
        "by": contexts,
        "entity": entities,
        "unit": unit,
        "entity_presence": entity_presence,
        "min_implication": min_implication,
        "min_similarity": min_similarity,
        "max_pairs": max_pairs,
        "max_signatures": max_signatures,
        "max_contexts": max_contexts,
        "example_limit": example_limit,
    }
    base["analysis_unit"] = {
        "counting_unit": unit,
        "denominator": n,
        "entity_keys": entities,
        "presence_aggregation": entity_presence if unit == "entities" else "per_row",
        "source_rows": sum(len(rows) for rows in units),
        "missing_key_excluded_rows": int((~eligible).sum()) if unit == "entities" else 0,
        "context_aggregation": "within_each_context",
    }
    base["availability"] = []
    for c in selected:
        mask = masks[c]
        metrics = {
            "populated": int(mask.sum()),
            "missing": int((~mask).sum()),
            "denominator": n,
            "populated_fraction": float(mask.mean()) if n else None,
        }
        base["availability"].append({"feature": c, **metrics})
        emit(
            "availability",
            f"{c}: populated values",
            [c],
            metrics,
            mask,
            exception_mask=~mask,
            selector={"operation": "presence", "feature": c},
        )
    signatures = defaultdict(list)
    for i in range(n):
        signatures[tuple(bool(masks[c][i]) for c in selected)].append(i)
    ordered = sorted(signatures.items(), key=lambda item: (-len(item[1]), item[0]))
    base["signatures"] = []
    for signature, indices in ordered[:max_signatures]:
        populated = [c for c, value in zip(selected, signature) if value]
        absent = [c for c in selected if c not in populated]
        record = {"present": populated, "absent": absent, "count": len(indices), "denominator": n}
        f = emit(
            "availability_signature",
            "Present: "
            + (", ".join(populated) or "none")
            + "; absent: "
            + (", ".join(absent) or "none"),
            selected,
            {"count": len(indices), "denominator": n},
            np.isin(np.arange(n), indices),
            structure={"present": populated, "absent": absent},
            selector={
                "operation": "availability_signature",
                "present": populated,
                "absent": absent,
            },
        )
        base["signatures"].append({**record, "finding_id": f["id"]})
    groups = defaultdict(list)
    for c in selected:
        groups[masks[c].tobytes()].append(c)
    base["families"] = []
    for group in groups.values():
        if len(group) > 1:
            f = emit(
                "availability_family",
                "Same availability: " + ", ".join(group),
                group,
                {"denominator": n, "populated": int(masks[group[0]].sum())},
                masks[group[0]],
            )
            base["families"].append(
                {
                    "features": group,
                    "evidence": ["identical_availability"],
                    "populated": int(masks[group[0]].sum()),
                    "finding_id": f["id"],
                }
            )
    evaluated = 0
    for a, b in islice(combinations(selected, 2), max_pairs):
        evaluated += 1
        x, y = masks[a], masks[b]
        both, union = int((x & y).sum()), int((x | y).sum())
        similarity = both / union if union else None
        metrics = {
            "both_present": both,
            "either_present": union,
            "both_absent": int((~x & ~y).sum()),
            "denominator": n,
            "presence_jaccard": similarity,
            "agreement": float((x == y).mean()) if n else None,
        }
        if similarity is not None and similarity >= min_similarity and not np.array_equal(x, y):
            emit(
                "similar_availability",
                f"{a} and {b} have similar presence",
                [a, b],
                metrics,
                x & y,
                exception_mask=x != y,
            )
        if both == 0 and x.any() and y.any():
            emit(
                "mutually_exclusive", f"{a} and {b} are mutually exclusive", [a, b], metrics, x | y
            )
        for source, target, first, second in [(a, b, x, y), (b, a, y, x)]:
            denominator = int(first.sum())
            rate = both / denominator if denominator else None
            if rate is not None and rate >= min_implication:
                emit(
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
                    first & second,
                    exception_mask=first & ~second,
                    selector={
                        "operation": "presence_implication",
                        "source": source,
                        "target": target,
                    },
                )
    context_groups = defaultdict(list)
    if contexts:
        for i in range(len(frame)):
            context_groups[tuple(int(codes[c][i]) for c in contexts)].append(i)
    base["contexts"] = []
    for rows in list(context_groups.values())[:max_contexts]:
        values = {
            c: normalize_scalar(frame[c].iloc[rows[0]] if present[c][rows[0]] else None).to_dict()
            for c in contexts
        }
        context_units = units_for(rows)
        context_masks = masks_for(context_units)
        record = {"values": values, "rows": len(rows), "availability": [], "finding_ids": []}
        for c in selected:
            mask = context_masks[c]
            metrics = {
                "feature": c,
                "populated": int(mask.sum()),
                "denominator": len(context_units),
            }
            record["availability"].append(metrics)
            f = finding(
                base,
                "context_availability",
                f"{c}: availability within {context_statement(values)}",
                list(dict.fromkeys([*contexts, c])),
                metrics,
                positions[sorted(i for j in np.flatnonzero(mask) for i in context_units[j])],
                exceptions=positions[
                    sorted(i for j in np.flatnonzero(~mask) for i in context_units[j])
                ],
                example_limit=example_limit,
                unit=unit,
                structure={"context": values},
                selector={"operation": "context_availability", "context": values, "feature": c},
            )
            record["finding_ids"].append(f["id"])
        base["contexts"].append(record)
    base["entities"] = []
    if entities:
        for c in selected:
            counts = [(int(present[c][rows].sum()), len(rows)) for rows in entity_groups.values()]
            predicates = {
                "any": lambda k, size: k > 0,
                "all": lambda k, size: k == size,
                "one": lambda k, size: k == 1,
                "some": lambda k, size: 0 < k < size,
                "none": lambda k, size: k == 0,
            }
            summary = {"feature": c, "denominator": len(counts)}
            for pattern, predicate in predicates.items():
                matching = [
                    rows
                    for rows, (k, size) in zip(entity_groups.values(), counts)
                    if predicate(k, size)
                ]
                summary[pattern] = len(matching)
                finding(
                    base,
                    "entity_availability",
                    f"{c}: {pattern} populated rows per entity ({', '.join(entities)})",
                    list(dict.fromkeys([*entities, c])),
                    {"entities": len(matching), "denominator": len(counts)},
                    positions[sorted(i for rows in matching for i in rows)],
                    example_limit=example_limit,
                    unit="entities",
                    structure={"entity_keys": entities, "presence_pattern": pattern},
                    selector={
                        "operation": "entity_availability",
                        "feature": c,
                        "presence_pattern": pattern,
                    },
                )
            base["entities"].append(summary)
    omitted = [j for _, indices in ordered[max_signatures:] for j in indices]
    base["coverage"] = {
        "pair_candidates": comb(len(selected), 2),
        "pairs_evaluated": evaluated,
        "signatures_total": len(signatures),
        "signatures_shown": min(len(signatures), max_signatures),
        "signature_omitted_units": len(omitted),
        "signature_omitted_rows": len(source_rows(omitted)),
        "contexts_total": len(context_groups),
        "contexts_shown": min(len(context_groups), max_contexts),
        "entity_missing_key_rows": int((~eligible).sum()) if entities else 0,
    }
    return result("missingness", base)
