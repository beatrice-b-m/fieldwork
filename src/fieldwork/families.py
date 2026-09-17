"""Typed, evidence-linked feature relationships for single-table browsing."""

from __future__ import annotations

from .evidence import qualitative_analysis_unit

RELATIONS = {
    "availability_family": "identical_availability",
    "similar_availability": "similar_availability",
    "presence_implication": "presence_implication",
    "mutually_exclusive": "mutually_exclusive",
    "indexed_family": "indexed_name",
    "value_alias": "equivalent_value_partitions",
    "exact_dependency": "exact_dependency",
    "approximate_dependency": "approximate_dependency",
}


def feature_network(base):
    """Connect features without treating connectedness as equivalence or composing FDs."""
    relationships = []
    nodes = {}
    adjacency = {}
    for record in base["findings"]:
        kind = RELATIONS.get(record["pattern"])
        if kind is None:
            continue
        selector = record["selector"]
        section = selector["analysis_section"]
        metrics = record["measurements"]
        relation = {
            "id": f"r{len(relationships)}",
            "kind": kind,
            "features": record["features"],
            "structure": record.get("structure", {}),
            "analysis_unit": qualitative_analysis_unit(base, record),
            "evidence": {
                "section": section,
                "finding_id": selector["finding_id"],
                "overview_finding_id": record["id"],
                "population_ref": f"sections.{section}.scope",
                "counting_unit": record["counting_unit"],
            },
        }
        if kind in {"exact_dependency", "approximate_dependency"}:
            relation["determinant"] = metrics["determinant"]
            relation["target"] = metrics["target"]
        if kind == "presence_implication":
            relation["source"] = selector["source"]
            relation["target"] = selector["target"]
        relationships.append(relation)
        columns = [f["column"] for f in record["features"]]
        for feature in record["features"]:
            c = feature["column"]
            nodes[c] = feature
            adjacency.setdefault(c, set()).update(set(columns) - {c})
    components = []
    remaining = set(nodes)
    while remaining:
        pending, component = [min(remaining)], set()
        while pending:
            c = pending.pop()
            if c in component:
                continue
            component.add(c)
            pending.extend(adjacency[c] - component)
        remaining -= component
        components.append(sorted(component))
    return {
        "nodes": [nodes[c] for c in sorted(nodes)],
        "relationships": relationships,
        "components": components,
    }
