"""A common-population quotient graph of observed candidate-key partitions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._runtime import checkpoint, phase
from ._kernels import same_mask
from .census import _scope
from .encoding import missing_code
from .result import KeySpec


def build_grain_graph(
    df: pd.DataFrame,
    specs: tuple[KeySpec, ...],
    encoded: dict,
    known: dict,
    *,
    dropna: bool,
    scope_metadata: dict | None,
) -> dict[str, Any]:
    from .grain import _record_on

    components = {column for spec in specs for column in spec.columns}
    mask = np.ones(len(df), dtype=bool)
    if dropna:
        for column in components:
            values, codes = encoded[column]
            absent = missing_code(values)
            if absent is not None:
                mask &= codes != absent
    scope_id = "s3:graph"
    metadata = scope_metadata or {}
    scope = _scope(
        scope_id,
        len(df),
        int((~mask).sum()),
        0,
        bool(metadata.get("conditional")),
        parent_scope=metadata.get("scope"),
    )
    evidence = []
    truth = {}
    support = {}
    with phase("grain graph evidence", len(specs) * len(df.columns), "tests") as progress:
        for spec in specs:
            key_table = pd.DataFrame({i: encoded[c][1][mask] for i, c in enumerate(spec.columns)})
            sizes = key_table.groupby(list(key_table.columns), sort=False, observed=True).size()
            support[spec.name] = {
                "evaluated_rows": int(mask.sum()),
                "evaluated_groups": len(sizes),
                "singleton_groups": int((sizes == 1).sum()),
                "repeated_groups": int((sizes > 1).sum()),
            }
            for target in df.columns:
                token = target
                if target in spec.columns:
                    record = {
                        "key_name": spec.name,
                        "target": token,
                        "holds": True if mask.any() else None,
                        "undefined_reason": None if mask.any() else "no_evaluated_groups",
                        **support[spec.name],
                        "violating_groups": 0,
                        "affected_rows": 0,
                        "group_rate": 0.0 if mask.any() else None,
                        "row_rate": 0.0 if mask.any() else None,
                    }
                    evaluated = mask
                else:
                    record, evaluated = _record_on(
                        df,
                        spec,
                        target,
                        mask,
                        known=known,
                        dropna=dropna,
                        encoded=encoded,
                        scope_prefix=scope_id,
                    )
                compatible = same_mask(evaluated, mask)
                record["scope_id"] = (
                    scope_id if compatible else f"{scope_id}:target:{len(evidence)}"
                )
                record["scope_compatible"] = compatible
                record["scope"] = _scope(
                    record["scope_id"],
                    len(df),
                    int((~evaluated).sum()),
                    0,
                    bool(metadata.get("conditional")),
                    parent_scope=metadata.get("scope"),
                )
                evidence.append(record)
                truth[(spec.name, target)] = record["holds"] if compatible else None

                progress.advance(detail=f"{spec.name} → {target}")

    def determines(left: KeySpec, right: KeySpec) -> bool | None:
        outcomes = [truth[(left.name, column)] for column in right.columns]
        return None if any(value is None for value in outcomes) else all(outcomes)

    relations = []
    finer = set()
    for left in specs:
        checkpoint()
        for right in specs:
            if left == right:
                continue
            holds = determines(left, right)
            relations.append({"determinant": left.name, "dependent": right.name, "holds": holds})
            if holds is True:
                finer.add((left.name, right.name))
    nodes = []
    key_node = {}
    for spec in specs:
        if spec.name in key_node:
            continue
        names = [
            other.name
            for other in specs
            if other.name == spec.name
            or ((spec.name, other.name) in finer and (other.name, spec.name) in finer)
        ]
        node_id = f"g{len(nodes)}"
        key_node.update({name: node_id for name in names})
        nodes.append({"id": node_id, "keys": names, **support[spec.name], "attributes": []})
    edges = {
        (key_node[coarse], key_node[fine])
        for fine, coarse in finer
        if key_node[fine] != key_node[coarse]
    }
    # The relation is transitively closed. An intermediate node makes an edge redundant.
    reduced = sorted(
        (a, b)
        for a, b in edges
        if not any((a, middle["id"]) in edges and (middle["id"], b) in edges for middle in nodes)
    )
    assignments = []
    unplaced = []
    for target in df.columns:
        token = target
        candidates = {key_node[s.name] for s in specs if truth[(s.name, target)] is True}
        coarsest = sorted(
            node for node in candidates if not any((other, node) in edges for other in candidates)
        )
        reason = None
        if not coarsest:
            reason = (
                "different_target_population"
                if any(not r["scope_compatible"] for r in evidence if r["target"] == token)
                else "no_supported_key"
            )
            if target not in components:
                unplaced.append({"target": token, "reason": reason})
        assignments.append(
            {
                "target": token,
                "nodes": coarsest,
                "reason": reason,
                "key_component": target in components,
            }
        )
        if target not in components:
            for node in nodes:
                if node["id"] in coarsest:
                    node["attributes"].append(token)
    return {
        "status": "computed" if mask.any() else "empty",
        "scope": scope,
        "missingness": "complete_candidate_keys" if dropna else "missing_as_level",
        "nodes": nodes,
        "edges": [{"source": a, "target": b, "relation": "finer_grouping"} for a, b in reduced],
        "key_relationships": relations,
        "dependencies": evidence,
        "assignments": assignments,
        "unplaced": unplaced,
    }
