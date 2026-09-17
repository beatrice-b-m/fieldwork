"""Deterministic beam search over census prefixes and supported nesting."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from ._explore import census
from .evidence import InvestigationResult, columns, finding, limit, prepare


class PathResult(InvestigationResult):
    @property
    def best(self):
        return Path(self.payload["paths"][0]["dimensions"]) if self.payload["paths"] else None


class Path:
    def __init__(self, dimensions):
        self.dimensions = tuple(dimensions)


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

    def measure(path):
        prefixes, keys, previous, redundancy = [], [()] * len(frame), 1, 0
        target_loss, availability_loss = 0.0, 0.0
        for c in path:
            keys = [(*key, int(value)) for key, value in zip(keys, encoded[c])]
            count = len(set(keys))
            prefixes.append(count)
            redundancy += count == previous
            previous = count
            if target_codes is not None:
                target_loss += impurity(target_codes, keys)
            if objective == "availability":
                availability_loss += impurity(availability_codes, keys)
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
            score += 12 * target_loss
        if objective == "availability":
            score += 12 * availability_loss
        return {
            "score": score,
            "prefix_counts": prefixes,
            "prefix_cost": prefix_cost,
            "overflow": overflow,
            "nesting_inversions": inversions,
            "redundant_steps": redundancy,
            "alias_steps": alias_steps,
            "target_impurity_sum": target_loss,
            "availability_impurity_sum": availability_loss,
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
        beam = [path for _, path in sorted(expanded)[:beam_width]]
    beam = [p for p in beam if required <= set(p)]
    base["paths"] = []
    for path in sorted(beam, key=lambda p: (measure(p)["score"], p))[:n_paths]:
        if not path:
            continue
        metrics = measure(path)
        preview_frame = frame[list(path)].copy()
        for c in path:
            preview_frame[c] = preview_frame[c].astype(object).where(present[c], None)
        preview = census(preview_frame, path, max_nodes=display_budget, max_levels=8).to_dict()
        base["paths"].append(
            {
                "dimensions": list(path),
                "measurements": metrics,
                "explanation": f"{objective}: prefix cost, redundancy, branching and supported nesting",
                "preview": preview,
            }
        )
        finding(base, "census_path", " → ".join(path), path, metrics, positions, example_limit=3)
    base["aliases"] = aliases
    base["nesting"] = [list(e) for e in sorted(edges)]
    base["coverage"] = {
        "features_requested": len(pool),
        "features_evaluated": len(selected),
        "features_omitted": [c for c in pool if c not in selected],
        "pairs_evaluated": tested_pairs,
        "pair_candidates": math.comb(len(active), 2),
        "paths_evaluated": evaluated,
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
