"""Deterministic beam search over census prefixes and supported nesting."""

from __future__ import annotations

import math
from functools import cache
from itertools import combinations

import numpy as np

from ._explore import census
from .evidence import (
    InvestigationResult,
    columns,
    finding,
    fingerprint,
    limit,
    prepare,
    saved_context,
)


class PathResult(InvestigationResult):
    @property
    def best(self):
        return self.path(0) if self.payload["paths"] else None

    def path(self, index=0):
        return Path(self.payload["paths"][index]["dimensions"], self.payload)


class Path:
    def __init__(self, dimensions, context):
        self.dimensions = tuple(dimensions)
        self._context = context

    def census(self, df, **options):
        """Evaluate this recommendation on its original scope and missing conventions."""
        if fingerprint(df) != self._context["source"]["dataset_id"]:
            raise ValueError("Source dataset differs; reapply a path recipe for a new delivery")
        if {"scope", "missing", "table_id"} & options.keys():
            raise ValueError(
                "Path census preserves its analysis context; rerun discovery to change it"
            )
        return census(df, self.dimensions, **saved_context(self._context), **options)


def suggest_paths(
    df,
    *,
    objective="structure",
    features=None,
    start_with=None,
    before=None,
    exclude=None,
    target=None,
    max_dimensions=4,
    max_candidates=200,
    max_features=20,
    max_pairs=200,
    beam_width=12,
    n_paths=3,
    display_budget=40,
    scope=None,
    missing=None,
    table_id="table",
):
    """Rank observed intermediate prefixes, not order-invariant joint information."""
    objectives = {"structure", "availability", "compact", "target", "context"}
    if objective not in objectives:
        raise ValueError(f"objective must be one of {sorted(objectives)}")
    for name, value in [
        ("max_dimensions", max_dimensions),
        ("max_candidates", max_candidates),
        ("max_features", max_features),
        ("beam_width", beam_width),
        ("n_paths", n_paths),
        ("display_budget", display_budget),
    ]:
        limit(name, value, minimum=1)
    limit("max_pairs", max_pairs)
    requested = columns(df, features)
    excluded = set(columns(df, exclude or []))
    starts = columns(df, start_with or [])
    constraints = list(before or [])
    for edge in constraints:
        if len(edge) != 2:
            raise ValueError("before entries must be pairs of columns")
        columns(df, edge)
    required = set(starts) | {c for edge in constraints for c in edge}
    if required & excluded or not required <= set(requested):
        raise ValueError("Steering columns must be included and not excluded")
    if len(starts) > max_dimensions or len(required) > max_dimensions:
        raise ValueError("max_dimensions cannot fit steering constraints")
    if objective == "context" and not starts:
        raise ValueError("context objective requires start_with")
    if objective == "target" and target is None:
        raise ValueError("target objective requires target")
    if target is not None:
        columns(df, [target])
    frame, positions, encoded, present, base = prepare(
        df, scope=scope, missing=missing, table_id=table_id
    )
    encoded = {c: np.where(present[c], code, -1) for c, code in encoded.items()}
    pool = [c for c in requested if c not in excluded and (c != target or c in required)]
    # Required columns survive the feature budget; otherwise use input order.
    selected = [c for c in pool if c in required] + [c for c in pool if c not in required]
    if len(required) > max_features:
        raise ValueError("max_features cannot fit steering columns")
    selected = selected[:max_features]
    cardinality = {c: len(set(encoded[c])) for c in selected}
    active = [c for c in selected if cardinality[c] > 1 or c in required]
    edges, aliases = set(), []
    tested_pairs = 0
    for a, b in combinations(active, 2):
        if tested_pairs >= max_pairs:
            break
        tested_pairs += 1
        pairs = set(zip(encoded[a].tolist(), encoded[b].tolist()))
        a_to_b, b_to_a = len(pairs) == cardinality[a], len(pairs) == cardinality[b]
        if a_to_b and b_to_a:
            aliases.append([a, b])
        elif a_to_b:
            edges.add((b, a))  # coarse before finer
        elif b_to_a:
            edges.add((a, b))
    for a, b in constraints:
        if a == b:
            raise ValueError("before constraints must be acyclic")
    # Validate user constraints, separately from soft observed nesting.
    pending, emitted = set(required), set()
    while pending:
        ready = {c for c in pending if all(a in emitted for a, b in constraints if b == c)}
        if not ready:
            raise ValueError("before constraints contain a cycle")
        pending -= ready
        emitted |= ready
    for i, c in enumerate(starts):
        if any(b == c and a not in starts[:i] for a, b in constraints):
            raise ValueError("start_with conflicts with before constraints")
    target_codes = encoded[target] if target is not None else None
    availability_codes = [tuple(bool(present[c][i]) for c in selected) for i in range(len(frame))]

    def impurity(labels, groups):
        if not len(labels):
            return 0.0
        counts = {}
        for group, value in zip(groups, labels):
            counts.setdefault(group, {})[value] = counts.setdefault(group, {}).get(value, 0) + 1
        return sum(sum(v.values()) - max(v.values()) for v in counts.values()) / len(labels)

    @cache
    def measure(path):
        prefixes, keys, previous, redundancy = [], [()] * len(frame), 1, 0
        target_losses, availability_losses = [], []
        for c in path:
            keys = [(*key, int(value)) for key, value in zip(keys, encoded[c])]
            count = len(set(keys))
            prefixes.append(count)
            redundancy += count == previous
            previous = count
            if target_codes is not None:
                target_losses.append(impurity(target_codes, keys))
            availability_losses.append(impurity(availability_codes, keys))
        inversions = sum(
            a in path and b in path and path.index(a) > path.index(b) for a, b in edges
        )
        alias_steps = sum(a in path and b in path for a, b in aliases)
        prefix_cost = sum(prefixes) / max(1, display_budget)
        overflow = sum(max(0, n - display_budget) for n in prefixes) / max(1, display_budget)
        # Objective penalties use prefix behavior: early useful splits and nesting.
        score = prefix_cost + overflow + 2 * redundancy + 3 * alias_steps
        if objective in {"structure", "context"}:
            score += 8 * inversions
        if objective == "target":
            score += 12 * sum(target_losses)
        if objective == "availability":
            score += 12 * sum(availability_losses)
        return {
            "score": score,
            "prefix_counts": prefixes,
            "prefix_cost": prefix_cost,
            "overflow": overflow,
            "nesting_inversions": inversions,
            "redundant_steps": redundancy,
            "alias_steps": alias_steps,
            "target_impurity_sum": sum(target_losses),
            "target_impurity_by_depth": target_losses,
            "availability_impurity_sum": sum(availability_losses),
            "availability_impurity_by_depth": availability_losses,
        }

    width = min(max_dimensions, len(active))
    beam = [tuple(starts)]
    evaluated = 0
    for depth in range(len(starts), width):
        expanded = []
        for path in beam:
            for c in active:
                if c in path or any(b == c and a not in path for a, b in constraints):
                    continue
                next_path = (*path, c)
                if len(required - set(next_path)) > width - len(next_path):
                    continue
                if evaluated >= max_candidates:
                    break
                evaluated += 1
                expanded.append((measure(next_path)["score"], next_path))
            if evaluated >= max_candidates:
                break
        if not expanded:
            break
        # Future extensions depend on the selected set, so keep its best order.
        # This reserves beam slots for different feature choices, not permutations.
        best_sets = {}
        for _, path in sorted(expanded):
            best_sets.setdefault(frozenset(path), path)
        beam = list(best_sets.values())[:beam_width]
    beam = [p for p in beam if required <= set(p)]
    alias_representative = {c: c for c in active}
    for a, b in aliases:
        old, new = alias_representative[b], alias_representative[a]
        alias_representative = {
            c: new if representative == old else representative
            for c, representative in alias_representative.items()
        }
    diverse = {}
    for path in sorted(beam, key=lambda p: (measure(p)["score"], p)):
        diverse.setdefault(frozenset(alias_representative[c] for c in path), path)
    base["paths"] = []
    for path in list(diverse.values())[:n_paths]:
        if not path:
            continue
        metrics = measure(path)
        reasons, explanation = path_reasons(path, metrics, edges, aliases, target)
        preview = census(
            df,
            path,
            scope=scope,
            missing=missing or {},
            table_id=table_id,
            max_nodes=display_budget,
            max_levels=8,
        ).to_dict()
        base["paths"].append(
            {
                "dimensions": list(path),
                "measurements": metrics,
                "explanation": explanation,
                "reasons": reasons,
                "preview": preview,
            }
        )
        finding(
            base,
            "census_path",
            " → ".join(path),
            path,
            {**metrics, "explanation": explanation, "reasons": reasons},
            positions,
            example_limit=3,
        )
    for a, b in aliases:
        finding(
            base,
            "value_alias",
            f"{a} and {b}: equivalent value partitions",
            [a, b],
            {"evaluated_rows": len(frame), "groups": cardinality[a]},
            positions,
            example_limit=3,
            structure={"relation": "equivalent_value_partitions"},
        )
    base["aliases"] = aliases
    base["nesting"] = [list(e) for e in sorted(edges)]
    base["coverage"] = {
        "features_requested": len(pool),
        "features_evaluated": len(selected),
        "features_omitted": [c for c in pool if c not in selected],
        "pairs_evaluated": tested_pairs,
        "pair_candidates": math.comb(len(active), 2),
        "paths_evaluated": evaluated,
        "distinct_alternatives": len(diverse),
        "alternative_policy": "best_order_per_feature_set_collapsing_alias_substitutions",
        "search_exhausted_budget": evaluated >= max_candidates,
        "requested_depth": width,
        "returned_depth": max((len(p["dimensions"]) for p in base["paths"]), default=0),
    }
    base["parameters"] = {
        "objective": objective,
        "features": requested,
        "start_with": starts,
        "before": constraints,
        "exclude": sorted(excluded),
        "target": target,
        "max_dimensions": max_dimensions,
        "max_candidates": max_candidates,
        "max_features": max_features,
        "max_pairs": max_pairs,
        "beam_width": beam_width,
        "n_paths": n_paths,
        "display_budget": display_budget,
    }
    return PathResult("paths", base, schema_version="1.0")


def path_reasons(path, metrics, edges, aliases, target):
    """Name observed evidence contributing to a path's score."""
    counts = metrics["prefix_counts"]
    nesting = [list(edge) for edge in sorted(edges) if all(c in path for c in edge)]
    redundant = [
        c for c, previous, current in zip(path, [1, *counts], counts) if previous == current
    ]
    alias_pairs = [pair for pair in aliases if all(c in path for c in pair)]
    reasons = [
        {
            "kind": "branching",
            "dimensions": list(path),
            "prefix_groups": counts,
            "overflow_cost": metrics["overflow"],
        },
        {
            "kind": "nesting",
            "coarse_to_fine": nesting,
            "reversed_edges": metrics["nesting_inversions"],
        },
        {"kind": "redundancy", "no_new_groups": redundant, "equivalent_pairs": alias_pairs},
        {
            "kind": "availability_separation",
            "nonmodal_fraction_by_depth": metrics["availability_impurity_by_depth"],
        },
    ]
    text = ["Observed prefix groups: " + " → ".join(f"{c}: {n}" for c, n in zip(path, counts))]
    if nesting:
        text.append(
            "Supported coarse-to-fine nesting: " + ", ".join(f"{a} → {b}" for a, b in nesting)
        )
    if redundant:
        text.append("Adds no groups: " + ", ".join(redundant))
    if alias_pairs:
        text.append(
            "Equivalent value partitions: " + ", ".join(f"{a} / {b}" for a, b in alias_pairs)
        )
    text.append(
        "Availability nonmodal fractions: "
        + ", ".join(f"{v:.3g}" for v in metrics["availability_impurity_by_depth"])
    )
    if target is not None:
        reasons.append(
            {
                "kind": "target_separation",
                "target": target,
                "nonmodal_fraction_by_depth": metrics["target_impurity_by_depth"],
            }
        )
        text.append(
            f"{target} nonmodal fractions: "
            + ", ".join(f"{v:.3g}" for v in metrics["target_impurity_by_depth"])
        )
    return reasons, "; ".join(text)
