"""A common-population graph of observed candidate-key partitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._runtime import checkpoint, phase
from ._kernels import same_mask

if TYPE_CHECKING:
    from .grain import Encoded, KeySpec


def build_grain_graph(
    size: int,
    names: list[str],
    specs: tuple[KeySpec, ...],
    encoded: Encoded,
    known: dict,
    *,
    dropna: bool,
) -> dict[str, Any]:
    """Relate keys and place columns using one population for every test.

    The population is every row, or with dropna the rows where every key
    component is present. A test whose own complete cases differ from it is
    marked incompatible and never used to relate keys or place a column.
    """
    components = {column for spec in specs for column in spec.columns}
    everyone = np.ones(size, dtype=bool)
    mask = encoded.complete(components, everyone) if dropna else everyone
    support = {spec.name: _support(encoded, spec, mask) for spec in specs}
    tests = _tests(size, names, specs, encoded, known, mask, dropna, support)
    truth = {(t["key"], t["target"]): t["holds"] if t["compatible"] else None for t in tests}

    def determines(left: KeySpec, right: KeySpec) -> bool | None:
        outcomes = [truth[(left.name, column)] for column in right.columns]
        return None if any(value is None for value in outcomes) else all(outcomes)

    relations, finer = [], set()
    for left in specs:
        checkpoint()
        for right in specs:
            if left != right:
                holds = determines(left, right)
                relations.append(
                    {"determinant": left.name, "dependent": right.name, "holds": holds}
                )
                if holds is True:
                    finer.add((left.name, right.name))
    nodes, key_node = _nodes(specs, finer, support)
    edges = {(key_node[coarse], key_node[fine]) for fine, coarse in finer}
    edges = {(a, b) for a, b in edges if a != b}
    placements = _place(names, specs, components, truth, tests, key_node, edges, nodes)
    return {
        "status": "computed" if mask.any() else "empty",
        "evaluated_rows": int(mask.sum()),
        "missing_excluded_rows": size - int(mask.sum()),
        "missingness": "complete_candidate_keys" if dropna else "missing_as_level",
        "nodes": nodes,
        "edges": [
            {"source": a, "target": b, "relation": "finer_grouping"} for a, b in _reduce(edges)
        ],
        "key_relationships": relations,
        "tests": tests,
        **placements,
    }


def _support(encoded: Encoded, spec: KeySpec, mask: np.ndarray) -> dict[str, int]:
    table = pd.DataFrame({i: encoded.codes[c][mask] for i, c in enumerate(spec.columns)})
    sizes = table.groupby(list(table.columns), sort=False, observed=True).size()
    return {
        "evaluated_rows": int(mask.sum()),
        "evaluated_groups": len(sizes),
        "singleton_groups": int((sizes == 1).sum()),
        "repeated_groups": int((sizes > 1).sum()),
    }


_COUNTS = ("evaluated_rows", "evaluated_groups", "violating_groups", "affected_rows")


def _tests(size, names, specs, encoded, known, mask, dropna, support) -> list[dict[str, Any]]:
    """Every key x column test on the graph population, with its counts."""
    from .grain import record_on

    tests = []
    with phase("grain graph evidence", len(specs) * len(names), "tests") as progress:
        for spec in specs:
            for target in names:
                if target in spec.columns:
                    # A key determines its own components wherever it is observed.
                    counts = {**support[spec.name], "violating_groups": 0, "affected_rows": 0}
                    holds, compatible = (True if mask.any() else None), True
                else:
                    record, evaluated = record_on(
                        size, spec, target, mask, known=known, dropna=dropna, encoded=encoded
                    )
                    counts, holds = record, record["holds"]
                    compatible = same_mask(evaluated, mask)
                test = {"key": spec.name, "target": target, "holds": holds}
                test["compatible"] = compatible
                test.update({k: counts[k] for k in _COUNTS})
                test["singleton_groups"] = counts["singleton_groups"]
                test["repeated_groups"] = counts["repeated_groups"]
                tests.append(test)
                progress.advance(detail=f"{spec.name} → {target}")
    return tests


def _nodes(specs, finer, support) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """One node per class of mutually determining keys."""
    nodes, key_node = [], {}
    for spec in specs:
        if spec.name in key_node:
            continue
        keys = [
            other.name
            for other in specs
            if other.name == spec.name
            or ((spec.name, other.name) in finer and (other.name, spec.name) in finer)
        ]
        node_id = f"g{len(nodes)}"
        key_node.update({name: node_id for name in keys})
        nodes.append({"id": node_id, "keys": keys, **support[spec.name], "attributes": []})
    return nodes, key_node


def _reduce(edges: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop edges implied by a path through an intermediate node (transitive reduction)."""
    middles = {node for edge in edges for node in edge}
    return sorted(
        (a, b) for a, b in edges if not any((a, m) in edges and (m, b) in edges for m in middles)
    )


def _place(names, specs, components, truth, tests, key_node, edges, nodes) -> dict[str, Any]:
    """Assign each column to its coarsest determining nodes, or record why not."""
    assignments, unplaced = [], []
    by_id = {node["id"]: node for node in nodes}
    for target in names:
        candidates = {key_node[s.name] for s in specs if truth[(s.name, target)] is True}
        coarsest = sorted(
            node for node in candidates if not any((other, node) in edges for other in candidates)
        )
        reason = None
        if not coarsest:
            mismatch = any(not t["compatible"] for t in tests if t["target"] == target)
            reason = "different_target_population" if mismatch else "no_supported_key"
            if target not in components:
                unplaced.append({"target": target, "reason": reason})
        assignments.append(
            {
                "target": target,
                "nodes": coarsest,
                "reason": reason,
                "key_component": target in components,
            }
        )
        if target not in components:
            for node in coarsest:
                by_id[node]["attributes"].append(target)
    return {"assignments": assignments, "unplaced": unplaced}
