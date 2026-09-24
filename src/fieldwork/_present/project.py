"""Allowlisted projections of saved evidence: the only input of every renderer.

A projection is fresh JSON data built field by field from a saved result, never
a copy of it. With ``detail="topology"`` it keeps structure (labels, relations,
states, omissions) and drops every quantity, position and statistic, and it
orders frequency-ranked content canonically, so topology depends on structure
only. Each kind has one projector function.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .._explore.encoding import json_order
from ..evidence import context_statement, qualitative_analysis_unit
from ..result import overview_findings
from .common import controls, label, predicate

FINDING_KINDS = {"missingness", "dependencies", "paths", "value_patterns", "overview", "comparison"}
COMPOSITES = {"overview", "profile"}


@dataclass(frozen=True)
class Context:
    """What every projector needs besides the data: disclosure and scope."""

    detail: str
    scope: Mapping[str, Any]
    missing_label: str = "<NA>"

    @property
    def full(self) -> bool:
        return self.detail == "full"

    def value(self, value: Any) -> str:
        return label(value, missing_label=self.missing_label)

    def predicate(self, column: str, value: Any) -> str:
        return predicate(column, value, self.missing_label)

    def population(self, record: Mapping[str, Any]) -> str:
        """A record's population within the analysis scope, relative to the source."""
        restricted = record.get("restriction_excluded_rows", 0)
        restricted += self.scope.get("restriction_excluded_rows", 0)
        text = "Conditional cohort" if restricted else "Input population"
        if self.full:
            text += (
                f" · {record['evaluated_rows']} evaluated / "
                f"{self.scope.get('input_rows', record['evaluated_rows'])} input rows"
                f" · {record.get('missing_excluded_rows', 0)} missing, "
                f"{restricted} restricted exclusions"
            )
        return text


def resolve(data: Mapping[str, Any], section: str | None) -> Mapping[str, Any]:
    """The requested part of a saved result (a section of a composite)."""
    kind = data.get("kind")
    if section is None or (section == kind and kind not in COMPOSITES):
        return data
    if kind not in COMPOSITES:
        raise ValueError("section does not match result kind")
    if section not in data.get("sections", {}):
        raise ValueError(f"Unknown section: {section!r}")
    part = data["sections"][section]
    if part.get("status") == "not_requested":
        raise ValueError(f"Section {section!r} was not computed: it was not requested")
    return part


def project(
    data: Mapping[str, Any],
    *,
    section: str | None = None,
    detail: str = "full",
    missing_label: str = "<NA>",
) -> dict[str, Any]:
    """Project a saved result (or one of its sections) for presentation."""
    if detail not in {"full", "topology"}:
        raise ValueError("detail must be 'full' or 'topology'")
    data = resolve(data, section)
    if data.get("status") == "not_requested":
        raise ValueError("Requested visualization section was not computed")
    kind = data.get("kind")
    context = Context(detail, data.get("scope") or {}, missing_label)
    if kind in FINDING_KINDS:
        return _findings_kind(data, context)
    if kind not in _PROJECTORS:
        raise ValueError(f"No presentation for result kind {kind!r}")
    output = {"kind": kind, "detail": detail, "conditional": _conditional(data)}
    output.update(_PROJECTORS[kind](data, context))
    return output


def _conditional(data: Mapping[str, Any]) -> bool:
    """Whether a scope, context or pre-selection restricted the analyzed rows."""
    records = [data.get("scope") or {}, data, data.get("tree") or {}, *data.get("pairs", [])]
    return any(record.get("restriction_excluded_rows") for record in records)


def _warnings(data: Mapping[str, Any], context: Context) -> list[dict[str, Any]]:
    output = []
    for warning in data.get("warnings", []):
        if not context.full and warning["code"] == "LOW_RETAINED_FRACTION":
            continue
        record = {"code": warning["code"]}
        if warning.get("column") is not None:
            record["column"] = label(warning["column"], column=True)
        if context.full:
            record["detail"] = ", ".join(
                f"{key}={value!r}"
                for key, value in warning.items()
                if key not in {"code", "column"}
            )
        output.append(record)
    return output


def _levels(data: Mapping[str, Any], context: Context) -> dict[str, Any]:
    features = []
    for feature in data["per_feature"]:
        rows = feature["levels"]
        if not context.full:
            rows = sorted(rows, key=lambda r: json_order(r["value"]))
        projected: dict[str, Any] = {
            "label": label(feature["column"], column=True),
            "scope": context.population(feature),
            "excluded": bool(
                feature["missing_excluded_rows"] or context.scope.get("restriction_excluded_rows")
            ),
            "rows": [],
        }
        for row in rows:
            item: dict[str, Any] = {"label": context.value(row["value"])}
            if context.full:
                item.update(count=row["count"], share=row["share_of_feature"])
            projected["rows"].append(item)
        if feature["omitted_levels"]:
            item = {"label": "Omitted levels", "omitted": True}
            if context.full:
                total = feature["evaluated_rows"]
                item.update(
                    count=feature["unreported_rows"],
                    share=feature["unreported_rows"] / total if total else None,
                    levels=feature["omitted_levels"],
                )
            projected["rows"].append(item)
        if context.full:
            projected["summary"] = {
                k: feature[k]
                for k in ("levels_reported", "levels_total", "reported_rows", "evaluated_rows")
            }
        features.append(projected)
    return {"status": data["status"], "features": features, "warnings": _warnings(data, context)}


def _census(data: Mapping[str, Any], context: Context) -> dict[str, Any]:
    tree = data["tree"]
    children = defaultdict(list)
    for node in tree["nodes"]:
        children[node["parent_id"]].append(node)
    if not context.full:
        for siblings in children.values():
            siblings.sort(key=lambda n: json_order(n["value"]))
    rows: list[dict[str, Any]] = []
    total = tree["root"]["count"]
    # Explicit parentage, not stored depth, defines the tree.
    stack = [(tree["root"], None, 0)]
    while stack:
        node, parent, depth = stack.pop()
        item_id = f"c{len(rows)}"
        item: dict[str, Any] = {
            "id": item_id,
            "parent": parent,
            "depth": depth,
            "label": "All evaluated rows"
            if parent is None
            else context.predicate(node["column"], node["value"]),
        }
        if context.full:
            item.update(
                count=node["count"],
                share=node["share_of_total"],
                parent_share=node["share_of_parent"],
            )
        rows.append(item)
        if node["omitted_child_levels"]:
            rows.append(_omitted_branches(node, item_id, depth + 1, total, context))
        stack.extend((child, item_id, depth + 1) for child in reversed(children[node["node_id"]]))
    labels = {row["id"]: row["label"] for row in rows}
    for row in rows:
        row["parent_label"] = labels.get(row["parent"])
    return {
        "status": data["status"],
        "scope": context.population(tree),
        "caption": "Observed paths; omitted branches retain their original mass.",
        "dimensions": [label(c, column=True) for c in tree["dimensions"]],
        "rows": rows,
        "warnings": _warnings(data, context),
    }


def _omitted_branches(node, parent_id, depth, total, context) -> dict[str, Any]:
    """An omission as a synthetic child keeping its own mass."""
    omitted = {
        "id": f"o{parent_id[1:]}",
        "parent": parent_id,
        "depth": depth,
        "omitted": True,
        "reasons": list(node["stop_reasons"]),
        "label": "Omitted branches (" + ", ".join(node["stop_reasons"]) + ")",
    }
    if context.full:
        omitted.update(
            count=node["omitted_child_rows"],
            levels=node["omitted_child_levels"],
            share=node["omitted_child_rows"] / total if total else None,
            parent_share=node["omitted_child_rows"] / node["count"] if node["count"] else None,
        )
    return omitted


_STATE = {None: "undefined", True: "constant", False: "varying"}
_TEST_COUNTS = (
    "evaluated_rows",
    "evaluated_groups",
    "violating_groups",
    "singleton_groups",
    "repeated_groups",
    "affected_rows",
)


def _grain(data: Mapping[str, Any], context: Context) -> dict[str, Any]:
    if "graph" not in data:
        raise ValueError("Grain graph evidence is missing; recompute grain()")
    graph = data["graph"]
    names = {key["name"]: key for key in data["keys"]}
    feature_ids = {a["target"]: f"f{i}" for i, a in enumerate(graph["assignments"])}
    features = [
        {
            "id": feature_ids[a["target"]],
            "label": label(a["target"], column=True),
            "nodes": list(a["nodes"]),
            "key_component": a["key_component"],
            "reason": a["reason"],
        }
        for a in graph["assignments"]
    ]
    nodes = [_grain_node(node, names, feature_ids, context) for node in graph["nodes"]]
    evidence = []
    for test in graph["tests"]:
        record = {
            "feature": feature_ids[test["target"]],
            "key": test["key"],
            "state": _STATE[test["holds"]],
            "repeated_support": bool(test["repeated_groups"]),
            "compatible": test["compatible"],
            "scope": context.population(test),
        }
        if context.full:
            record.update({k: test[k] for k in _TEST_COUNTS})
        evidence.append(record)
    feature_labels = {f["id"]: f["label"] for f in features}
    node_labels = {n["id"]: " / ".join(n["titles"]) for n in nodes}
    for feature in features:
        feature["node_labels"] = [node_labels[n] for n in feature["nodes"]]
    for node in nodes:
        node["attribute_labels"] = [feature_labels[f] for f in node["attributes"]]
    edges = [
        {
            "source": e["source"],
            "target": e["target"],
            "source_label": node_labels[e["source"]],
            "target_label": node_labels[e["target"]],
        }
        for e in graph["edges"]
    ]
    for record in evidence:
        record["feature_label"] = feature_labels[record["feature"]]
    return {
        "status": data["status"],
        "caption": "Observed groupings among tested keys.",
        "scope": context.population(graph),
        "missingness": graph["missingness"].replace("_", " "),
        "features": features,
        "nodes": nodes,
        "edges": edges,
        "evidence": evidence,
        "dependencies": [_grain_test(d, context) for d in data["dependencies"]],
        "targets": [_grain_target(t) for t in data["targets"]],
    }


def _grain_node(node, names, feature_ids, context) -> dict[str, Any]:
    titles = []
    for name in node["keys"]:
        columns = ", ".join(label(c, column=True) for c in names[name]["columns"])
        titles.append(controls(name if name == columns else f"{name} = ({columns})"))
    projected: dict[str, Any] = {
        "id": node["id"],
        "titles": titles,
        "role": support_role(
            node["evaluated_rows"], node["evaluated_groups"], node["repeated_groups"]
        ),
        "attributes": [feature_ids[a] for a in node["attributes"]],
        "key_names": list(node["keys"]),
    }
    if context.full:
        projected["support"] = {
            k: node[k]
            for k in ("evaluated_rows", "evaluated_groups", "singleton_groups", "repeated_groups")
        }
    return projected


def _grain_test(record: Mapping[str, Any], context: Context) -> dict[str, Any]:
    state = {None: "undefined", True: "holds", False: "fails"}[record["holds"]]
    projected = {
        "key": controls(record["key_name"]),
        "columns": [label(c, column=True) for c in record["key_columns"]],
        "target": label(record["target"], column=True),
        "state": state,
        # Exactness without a repeated key group holds trivially.
        "repeated_support": bool(record["repeated_groups"]),
        "reason": (record.get("undefined_reason") or "").replace("_", " ") or None,
        "scope": context.population(record),
        "excluded": bool(record["missing_excluded_rows"]),
    }
    if context.full:
        projected.update({k: record[k] for k in _TEST_COUNTS})
    return projected


def _grain_target(target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": label(target["target"], column=True),
        "comparable": target["cross_key_comparison"] == "comparable",
        "determined": bool(target["determining_keys"]),
        "coarsest": [controls(k) for k in target["coarsest_candidates"]],
        "equivalent": [[controls(k) for k in group] for group in target["equivalent_determinants"]],
        "incomparable": [
            [controls(k) for k in group] for group in target["incomparable_candidates"]
        ],
    }


def _pairs(data: Mapping[str, Any], context: Context) -> dict[str, Any]:
    ordered = sorted(
        {*data.get("features", []), *(c for record in data["pairs"] for c in record["columns"])}
    )
    indexes = {c: i for i, c in enumerate(ordered)}
    contexts: dict[str, dict[str, Any]] = {}

    def context_label(predicates) -> str:
        return ", ".join(context.predicate(p["column"], p["value"]) for p in predicates) or "Global"

    for predicates in data.get("contexts", []):
        contexts.setdefault(
            context_label(predicates), {"label": context_label(predicates), "cells": []}
        )
    records = []
    for record in data["pairs"]:
        name = context_label(record["context"])
        cell = {
            "a": indexes[record["columns"][0]],
            "b": indexes[record["columns"][1]],
            "a_label": label(record["columns"][0], column=True),
            "b_label": label(record["columns"][1], column=True),
            "relation": record["relation"] or "undefined",
            "scope": context.population(record),
        }
        if context.full:
            cell.update(
                association=record["cramers_v"], association_reason=record["cramers_v_reason"]
            )
        contexts.setdefault(name, {"label": name, "cells": []})["cells"].append(cell)
        records.append(_pair_record(record, name, cell, context))
    output = {
        "status": data["status"],
        "features": [label(c, column=True) for c in ordered],
        "contexts": list(contexts.values()),
        "records": records,
        "omitted": bool(data["omitted_pairs"] or data["omitted_contexts"]),
        "caption": "Row grouping → column grouping. Mapping and association are separate.",
    }
    if context.full:
        output["omitted_counts"] = {
            "pairs": data["omitted_pairs"],
            "contexts": data["omitted_contexts"],
        }
    return output


def _pair_record(record, context_name, cell, context) -> dict[str, Any]:
    projected = {
        "columns": [cell["a_label"], cell["b_label"]],
        "context": context_name,
        "relation": record["relation"],
        "relation_reason": (record["relation_reason"] or "").replace("_", " ") or None,
        "scope": cell["scope"],
    }
    if context.full:
        projected.update(
            association=record["cramers_v"],
            association_reason=(record["cramers_v_reason"] or "").replace("_", " ") or None,
        )
    absence = record.get("absence")
    if absence:
        domains = record["domains"]
        projected["absence"] = {
            "sources": [
                domains["a_source"].replace("_", " "),
                domains["b_source"].replace("_", " "),
            ],
            "examples": [
                [context.value(example["a"]), context.value(example["b"])]
                for example in absence["examples"]
            ],
            "examples_omitted": bool(absence["examples_omitted"]),
        }
        if context.full:
            projected["absence"].update(
                sizes=[domains["a_size"], domains["b_size"]],
                absent_cells=absence["absent_cells"],
                total_cells=absence["total_cells"],
                classes=dict(absence["classes"]),
                examples_omitted_count=absence["examples_omitted"],
            )
    return projected


def _joint_counts(data: Mapping[str, Any], context: Context) -> dict[str, Any]:
    a = [context.value(v) for v in data["a"]]
    b = [context.value(v) for v in data["b"]]
    predicates = [context.predicate(p["column"], p["value"]) for p in data.get("context", [])]
    omitted = data.get("omitted_cells", 0)
    caption = (
        "Observed joint cells; blank cells are unobserved or below min_count in this scope."
        if omitted
        else "Observed joint cells; blank cells are unobserved in this scope."
    )
    if predicates:
        caption += " Context: " + ", ".join(predicates)
    output = {
        "status": data["status"],
        "scope": context.population(data),
        "columns": [label(c, column=True) for c in data["columns"]],
        "context": ", ".join(predicates) or "global",
        "a": a,
        "b": b,
        "cells": [
            {
                "a": c["a"],
                "b": c["b"],
                "a_label": a[c["a"]],
                "b_label": b[c["b"]],
                **({"count": c["count"]} if context.full else {}),
            }
            for c in data["cells"]
        ],
        "omitted": bool(omitted),
        "caption": caption,
    }
    if context.full:
        output["omitted_counts"] = {"cells": omitted, "rows": data.get("omitted_rows", 0)}
    return output


def _schema_proposal(data: Mapping[str, Any], context: Context) -> dict[str, Any]:
    proposals = []
    for proposal in data["proposals"]:
        reasons = [
            {"code": r["code"], "value": r["value"]}
            for r in proposal["reasons"]
            if context.full or r["code"] in {"DTYPE", "NAME_HINT_ID"}
        ]
        evidence = proposal["fd_evidence"]
        projected: dict[str, Any] = {
            "label": label(proposal["column"], column=True),
            "role": proposal["proposed_role"],
            "reasons": reasons,
            "fd_evidence": evidence.replace("_", " ")
            if isinstance(evidence, str)
            else evidence["status"],
        }
        if isinstance(evidence, Mapping):
            projected["keys"] = [
                {
                    "name": controls(key["name"]),
                    "holds": key["holds"],
                    **({"groups": key["evaluated_groups"]} if context.full else {}),
                }
                for key in evidence["keys"]
            ]
        proposals.append(projected)
    output: dict[str, Any] = {"status": data["status"], "proposals": proposals}
    if context.full and "scope" in data:
        output["scope"] = data["scope"]
    return output


def _profile(data: Mapping[str, Any], context: Context) -> dict[str, Any]:
    sections = {}
    for name, section in data["sections"].items():
        if section.get("status") == "not_requested":
            sections[name] = {"kind": name, "status": "not_requested"}
        else:
            sections[name] = project(
                section, detail=context.detail, missing_label=context.missing_label
            )
    return {"status": data["status"], "sections": sections, "warnings": _warnings(data, context)}


_PROJECTORS: dict[str, Callable[[Mapping[str, Any], Context], dict[str, Any]]] = {
    "levels": _levels,
    "census": _census,
    "grain": _grain,
    "pairs": _pairs,
    "joint_counts": _joint_counts,
    "schema_proposal": _schema_proposal,
    "profile": _profile,
}


# Findings-based kinds: missingness, dependencies, paths, value patterns,
# comparisons and overviews.


def structural_evidence(structure: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: structure[k]
        for k in (
            "context",
            "presence",
            "present",
            "absent",
            "entity_keys",
            "presence_pattern",
            "relation",
            "strength",
            "repeated_support",
        )
        if k in structure
    }


def qualitative_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: unit[k] for k in ("counting_unit", "entity_keys", "presence_aggregation") if k in unit
    }


_ROLES = {"repeated grouping": 0, "unique identifier": 1, "constant": 2, "no evaluated support": 3}


def support_role(evaluated_rows: int, groups: int, repeated_groups: int) -> str:
    """A grouping's structural role from its support; the counts stay private."""
    if not evaluated_rows or not groups:
        return "no evaluated support"
    if groups == 1:
        return "constant"
    return "repeated grouping" if repeated_groups else "unique identifier"


def candidate_role(candidate: Mapping[str, Any]) -> str:
    return support_role(
        candidate["evaluated_rows"], candidate["groups"], candidate["repeated_groups"]
    )


def candidate_priority(candidate: Mapping[str, Any]) -> tuple:
    return (
        _ROLES[candidate_role(candidate)],
        -len(candidate["determines_with_repeated_support"]),
        -candidate["repeated_rows"],
        len(candidate["columns"]),
        tuple(candidate["columns"]),
    )


def candidate_summaries(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One comparable ranking for the whole saved candidate collection."""
    candidates = [{**c, "role": candidate_role(c)} for c in data.get("candidates", [])]
    return sorted(candidates, key=candidate_priority)


def collapse_equivalent(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge single-column candidates that partition the same rows identically.

    Mutual determination only implies the same partition on the same evaluated
    population, so group counts and evaluated rows must also match.
    """
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        columns = candidate["columns"]
        twin = next(
            (
                k
                for k in kept
                if len(columns) == 1
                and len(k["columns"]) == 1
                and k["groups"] == candidate["groups"]
                and k["evaluated_rows"] == candidate["evaluated_rows"]
                and columns[0] in k.get("determines", [])
                and k["columns"][0] in candidate.get("determines", [])
            ),
            None,
        )
        if twin is None:
            kept.append({**candidate, "equivalent": []})
        else:
            twin["equivalent"].append(columns[0])
    return kept


def grain_title(candidate: Mapping[str, Any]) -> str:
    title = ", ".join(candidate["columns"])
    if candidate.get("equivalent"):
        title += " (equivalent: " + ", ".join(candidate["equivalent"]) + ")"
    return title


def signature_label(signature: Mapping[str, Any]) -> str:
    present, absent = signature["present"], signature["absent"]
    if not absent:
        return "All populated"
    if len(absent) <= len(present):
        return "Missing: " + ", ".join(absent)
    return "Only: " + (", ".join(present) or "none")


def candidate_explanation(candidate: Mapping[str, Any]) -> str:
    supported = candidate["determines_with_repeated_support"]
    tested, possible = candidate["global_targets_tested"], candidate["global_targets_possible"]
    text = (
        f"{candidate['groups']} groups, {candidate['repeated_groups']} repeated; "
        f"{candidate['repeated_rows']} determinant repeated rows; "
        f"{len(candidate['determines'])} exact targets, {len(supported)} with repeated support; "
        f"global targets tested {tested}/{possible}"
    )
    if tested < possible:
        text += " (incomplete; untested targets are unknown)"
    return text


def dependency_explanation(row: Mapping[str, Any], dropna: bool | None) -> str:
    n, repeated = row["evaluated_rows"], row["repeated_rows"]
    exactness = "Exact" if row["exact"] else "Approximate"
    if row["exact"] is None:
        exactness = "Consistency unsupported"
    text = f"{exactness} on {n} evaluated rows"
    q, observed = row["determinant_evaluated_rows"], row["target_observed_rows"]
    text += f"; target observed on {observed}/{q} determinant-eligible rows"
    if row["target_coverage"] is not None:
        text += f" (observed target coverage {row['target_coverage']:.3g})"
    if repeated == 0:
        text += "; no repeated groups after exclusions; repeat-only consistency not assessable"
    else:
        text += f"; {repeated}/{n} rows in repeated groups"
        if row["repeat_coverage"] is not None:
            text += f" (repeat coverage {row['repeat_coverage']:.3g})"
        accuracy = row["repeat_modal_accuracy"]
        text += (
            f"; repeat-only consistency {accuracy:.3g}"
            if accuracy is not None
            else "; repeat-only consistency unavailable"
        )
    if dropna is False:
        text += "; missing values participated in consistency measurements"
    return text


TRIVIAL_EXACTNESS = "without repeated support (every determinant group is one row)"


def support_note(row: Mapping[str, Any]) -> str | None:
    """A finding's trivial-exactness caveat, readable without its counts."""
    return (
        "Holds " + TRIVIAL_EXACTNESS if row["structure"].get("repeated_support") is False else None
    )


def dependency_label(row: Mapping[str, Any]) -> str:
    text = ", ".join(row["determinant"]) + " → " + row["target"]
    if row.get("context"):
        text += " within " + context_statement(row["context"])
    return text


def findings_of(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    return overview_findings(data) if data.get("kind") == "overview" else data.get("findings", [])


def _findings_kind(data: Mapping[str, Any], context: Context) -> dict[str, Any]:
    output: dict[str, Any] = {"kind": data["kind"], "detail": context.detail}
    findings = findings_of(data)
    output["findings"] = [_finding_row(data, record, context) for record in findings]
    unit = _analysis_unit(data)
    if unit is not None:
        output["analysis_unit"] = qualitative_unit(unit)
    if data.get("skipped_features"):
        output["skipped_features"] = data["skipped_features"]
    output.update(_comparison_sides(data, context))
    if context.full:
        if unit is not None:
            output["analysis_unit"] = unit
        for key in (
            "scope",
            "coverage",
            "availability",
            "signatures",
            "entities",
            "contexts",
            "changes",
        ):
            if key in data:
                output[key] = data[key]
    if "feature_network" in data:
        output["feature_network"] = _network(data, findings, context)
    if data["kind"] == "dependencies" and context.full:
        dropna = data.get("parameters", {}).get("dropna")
        output["candidates"] = candidate_summaries(data)
        output["dependencies"] = [
            {**d, "explanation": dependency_explanation(d, dropna)}
            for d in data.get("dependencies", [])
        ]
    elif data["kind"] == "dependencies":
        # Roles, in search order, tell a unique row ID from a repeated grouping.
        output["candidates"] = [
            {"columns": c["columns"], "role": candidate_role(c)} for c in data.get("candidates", [])
        ]
    if data["kind"] == "overview":
        output.update(_overview(data, context))
    if not context.full:
        # Frequency-ranked evidence is canonicalized; no support or selectors are exported.
        output["findings"].sort(key=lambda r: (r["pattern"], r["statement"]))
    return output


def _analysis_unit(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if data.get("kind") == "overview":
        return data["sections"].get("missingness", {}).get("analysis_unit")
    return data.get("analysis_unit")


def _finding_row(data, record, context) -> dict[str, Any]:
    row = {
        "pattern": record["pattern"],
        "statement": record["statement"],
        "features": record["features"],
        "structure": structural_evidence(record.get("structure", {})),
        "counting_unit": record["counting_unit"],
        "analysis_unit": qualitative_analysis_unit(data, record),
    }
    if context.full:
        row.update(
            id=record["id"],
            measurements=record["measurements"],
            examples=record["examples"],
            exceptions=record["exceptions"],
        )
        if "lead" in record:
            row["lead"] = record["lead"]
        if record["pattern"] in {"exact_dependency", "approximate_dependency"}:
            owner = data.get("sections", {}).get("dependencies", data)
            row["explanation"] = dependency_explanation(
                record["measurements"], owner.get("parameters", {}).get("dropna")
            )
    return row


def _comparison_sides(data, context) -> dict[str, Any]:
    output = {}
    for side in ("before", "after"):
        scope, unit = data.get(f"{side}_scope"), data.get(f"{side}_analysis_unit")
        if scope is not None:
            output[f"{side}_scope"] = (
                scope if context.full else {k: scope[k] for k in ("name", "parent") if k in scope}
            )
        if unit is not None:
            output[f"{side}_analysis_unit"] = unit if context.full else qualitative_unit(unit)
    return output


_NETWORK_FIELDS = (
    "kind",
    "analysis_unit",
    "features",
    "structure",
    "determinant",
    "source",
    "target",
)


def _network(data, findings, context) -> dict[str, Any]:
    network = dict(data["feature_network"])
    by_id = {record["id"]: record for record in findings}
    network["relationships"] = []
    for edge in data["feature_network"]["relationships"]:
        record = by_id.get(edge.get("evidence", {}).get("overview_finding_id"))
        if "analysis_unit" not in edge and record is not None:
            edge = {**edge, "analysis_unit": qualitative_analysis_unit(data, record)}
        network["relationships"].append(edge)
    if context.full:
        return network
    relationships = [
        {
            k: structural_evidence(edge[k])
            if k == "structure"
            else qualitative_unit(edge[k])
            if k == "analysis_unit"
            else edge[k]
            for k in _NETWORK_FIELDS
            if k in edge
        }
        for edge in network["relationships"]
    ]
    return {
        "nodes": network["nodes"],
        "components": network["components"],
        "relationships": sorted(relationships, key=lambda e: json.dumps(e, sort_keys=True)),
    }


_GRAIN_FIELDS = (
    "groups",
    "repeated_groups",
    "repeated_rows",
    "determines",
    "determines_with_repeated_support",
    "global_targets_tested",
    "global_targets_possible",
)


def _overview(data, context) -> dict[str, Any]:
    sections = data["sections"]
    output: dict[str, Any] = {}
    if context.full:
        output["section_coverage"] = {
            name: section["coverage"] for name, section in sections.items() if "coverage" in section
        }
    omitted = [name for name, s in sections.items() if s.get("status") == "not_requested"]
    if omitted:
        output["section_selection"] = {
            "requested": [name for name in sections if name not in omitted],
            "omitted": omitted,
        }
    dependencies = sections["dependencies"]
    candidates = (
        candidate_summaries(dependencies) if context.full else dependencies.get("candidates", [])
    )
    summary = {
        "families": [f["features"] for f in sections["missingness"].get("families", [])],
        "paths": [path["dimensions"] for path in sections["paths"].get("paths", [])],
        "grains": [
            {
                "columns": c["columns"],
                "equivalent": c["equivalent"],
                "role": candidate_role(c),
                **({k: c[k] for k in _GRAIN_FIELDS} if context.full else {}),
            }
            for c in collapse_equivalent(candidates)
        ],
        "signatures": [
            {
                "present": s["present"],
                "absent": s["absent"],
                **({"count": s["count"]} if context.full else {}),
            }
            for s in sections["missingness"].get("signatures", [])
        ],
    }
    if not context.full:
        for rows in summary.values():
            rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
    output["overview"] = summary
    return output
