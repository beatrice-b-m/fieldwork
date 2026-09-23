"""Allowlisted presentation data; disclosure filtering precedes serialization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Literal

from ..result import Result
from .encoding import display, json_order
from .render import _CONTROL


def label(value: Any, *, column: bool = False) -> str:
    text = str(value) if column else display(value)
    return _CONTROL.sub(lambda match: f"\\u{ord(match.group()):04x}", text)


def _scope(record: Mapping, scope: Mapping, full: bool) -> str:
    """Population of a record within the analysis scope, relative to the source."""
    restricted = record.get("restriction_excluded_rows", 0) + scope["restriction_excluded_rows"]
    text = "Conditional cohort" if restricted else "Input population"
    if full:
        text += (
            f" · {record['evaluated_rows']} evaluated / {scope['input_rows']} input rows"
            f" · {record.get('missing_excluded_rows', 0)} missing, "
            f"{restricted} restricted exclusions"
        )
    return text


def visualization_data(
    result: Result | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Literal["full", "topology"] = "full",
) -> dict[str, Any]:
    """Return reviewed presentation data, including a compact grain structure.

    This is a fresh allowlisted projection, not a copy of the analytical result.
    Topology output contains structural identifiers and labels, never quantities.
    """
    if detail not in {"full", "topology"}:
        raise ValueError("detail must be 'full' or 'topology'")
    data = result.to_dict() if isinstance(result, Result) else result
    if data.get("kind") == "explore":
        section = section or "grain"
        if section not in data["sections"]:
            raise ValueError(f"Unknown explore section: {section!r}")
        data = data["sections"][section]
    elif section is not None and section != data.get("kind"):
        raise ValueError("section does not match result kind")
    if data.get("status") == "not_requested":
        raise ValueError("Requested visualization section was not computed")
    kind = data.get("kind")
    full = detail == "full"
    scope = data.get("scope") or {"restriction_excluded_rows": 0, "input_rows": 0}
    output = {"kind": kind, "detail": detail}
    if kind == "grain":
        if "graph" not in data:
            raise ValueError("Grain graph evidence is missing; recompute grain() with schema 0.3+")
        graph = data["graph"]
        output["caption"] = "Observed groupings among tested keys."
        output["scope"] = _scope(graph, scope, full)
        output["missingness"] = graph["missingness"].replace("_", " ")
        names = {key["name"]: key for key in data["keys"]}
        feature_ids = {a["target"]: f"f{i}" for i, a in enumerate(graph["assignments"])}
        output["features"] = [
            {
                "id": feature_ids[a["target"]],
                "label": label(a["target"], column=True),
                "nodes": list(a["nodes"]),
                "key_component": a["key_component"],
                "reason": a["reason"],
            }
            for a in graph["assignments"]
        ]
        output["nodes"] = []
        for node in graph["nodes"]:
            titles = []
            for name in node["keys"]:
                columns = ", ".join(label(c, column=True) for c in names[name]["columns"])
                title = name if name == columns else f"{name} = ({columns})"
                titles.append(_CONTROL.sub(lambda m: f"\\u{ord(m.group()):04x}", title))
            projected = {
                "id": node["id"],
                "titles": titles,
                "attributes": [feature_ids[a] for a in node["attributes"]],
                "key_names": list(node["keys"]),
            }
            if full:
                projected["support"] = {
                    k: node[k]
                    for k in (
                        "evaluated_rows",
                        "evaluated_groups",
                        "singleton_groups",
                        "repeated_groups",
                    )
                }
            output["nodes"].append(projected)
        output["edges"] = [{"source": e["source"], "target": e["target"]} for e in graph["edges"]]
        output["evidence"] = []
        for record in graph["tests"]:
            projected = {
                "feature": feature_ids[record["target"]],
                "key": record["key"],
                "state": (
                    "undefined"
                    if record["holds"] is None
                    else "constant"
                    if record["holds"]
                    else "varying"
                ),
                "compatible": record["compatible"],
                "scope": _scope(record, scope, full),
            }
            if full:
                projected.update(
                    {
                        k: record[k]
                        for k in (
                            "evaluated_rows",
                            "evaluated_groups",
                            "violating_groups",
                            "singleton_groups",
                            "repeated_groups",
                            "affected_rows",
                        )
                    }
                )
            output["evidence"].append(projected)
        feature_labels = {f["id"]: f["label"] for f in output["features"]}
        node_labels = {n["id"]: " / ".join(n["titles"]) for n in output["nodes"]}
        for feature in output["features"]:
            feature["node_labels"] = [node_labels[n] for n in feature["nodes"]]
        for node in output["nodes"]:
            node["attribute_labels"] = [feature_labels[f] for f in node["attributes"]]
        for edge in output["edges"]:
            edge.update(
                source_label=node_labels[edge["source"]], target_label=node_labels[edge["target"]]
            )
        for record in output["evidence"]:
            record["feature_label"] = feature_labels[record["feature"]]
    elif kind == "levels":
        output["features"] = []
        for feature in data["per_feature"]:
            rows = feature["levels"]
            if not full:
                rows = sorted(rows, key=lambda r: json_order(r["value"]))
            projected = {
                "label": label(feature["column"], column=True),
                "scope": _scope(feature, scope, full),
                "rows": [],
            }
            for row in rows:
                item = {"label": label(row["value"])}
                if full:
                    item.update(count=row["count"], share=row["share_of_feature"])
                projected["rows"].append(item)
            if feature["omitted_levels"]:
                item = {"label": "Omitted levels", "omitted": True}
                if full:
                    total = feature["evaluated_rows"]
                    item.update(
                        count=feature["unreported_rows"],
                        share=feature["unreported_rows"] / total if total else None,
                    )
                projected["rows"].append(item)
            output["features"].append(projected)
    elif kind == "census":
        output["scope"] = _scope(data["tree"], scope, full)
        output["caption"] = "Observed paths; omitted branches retain their original mass."
        children = defaultdict(list)
        for node in data["tree"]["nodes"]:
            children[node["parent_id"]].append(node)
        if not full:
            for siblings in children.values():
                siblings.sort(key=lambda n: json_order(n["value"]))
        output["rows"] = []
        stack = [(data["tree"]["root"], None)]
        total = data["tree"]["root"]["count"]
        while stack:
            node, parent = stack.pop()
            item_id = f"c{len(output['rows'])}"
            item = {
                "id": item_id,
                "parent": parent,
                "depth": node["depth"],
                "label": "All evaluated rows"
                if parent is None
                else f"{label(node['column'], column=True)}={label(node['value'])}",
            }
            if full:
                item.update(
                    count=node["count"],
                    share=node["share_of_total"],
                    parent_share=node["share_of_parent"],
                )
            output["rows"].append(item)
            # Emit an omission as a synthetic child, keeping its own mass.
            descendants = children[node["node_id"]]
            if node["omitted_child_levels"]:
                omitted = {
                    "id": f"o{len(output['rows'])}",
                    "parent": item_id,
                    "depth": node["depth"] + 1,
                    "omitted": True,
                    "label": "Omitted branches (" + ", ".join(node["stop_reasons"]) + ")",
                }
                if full:
                    omitted.update(
                        count=node["omitted_child_rows"],
                        share=node["omitted_child_rows"] / total if total else None,
                        parent_share=(
                            node["omitted_child_rows"] / node["count"] if node["count"] else None
                        ),
                    )
                output["rows"].append(omitted)
            stack.extend((child, item_id) for child in reversed(descendants))
        row_labels = {row["id"]: row["label"] for row in output["rows"]}
        for row in output["rows"]:
            row["parent_label"] = row_labels.get(row["parent"])
    elif kind == "pairs":
        output["contexts"] = []
        contexts = {}
        ordered = sorted(
            {*data.get("features", []), *(c for record in data["pairs"] for c in record["columns"])}
        )
        output["features"] = [label(c, column=True) for c in ordered]
        indexes = {c: i for i, c in enumerate(ordered)}
        for predicates in data.get("contexts", []):
            context = tuple(
                (label(p["column"], column=True), label(p["value"])) for p in predicates
            )
            if context not in contexts:
                projected = {
                    "label": ", ".join(f"{c}={v}" for c, v in context) or "Global",
                    "cells": [],
                }
                contexts[context] = projected
                output["contexts"].append(projected)
        for record in data["pairs"]:
            context = tuple(
                (label(p["column"], column=True), label(p["value"])) for p in record["context"]
            )
            if context not in contexts:
                projected = {
                    "label": ", ".join(f"{c}={v}" for c, v in context) or "Global",
                    "cells": [],
                }
                contexts[context] = projected
                output["contexts"].append(projected)
            item = {
                "a": indexes[record["columns"][0]],
                "b": indexes[record["columns"][1]],
                "a_label": label(record["columns"][0], column=True),
                "b_label": label(record["columns"][1], column=True),
                "relation": record["relation"] or "undefined",
                "scope": _scope(record, scope, full),
            }
            if full:
                item.update(
                    association=record["cramers_v"], association_reason=record["cramers_v_reason"]
                )
            contexts[context]["cells"].append(item)
        output["omitted"] = bool(data["omitted_pairs"] or data["omitted_contexts"])
        output["caption"] = "Row grouping → column grouping. Mapping and association are separate."
    elif kind == "joint_counts":
        output["scope"] = _scope(data, scope, full)
        output["columns"] = [label(c, column=True) for c in data["columns"]]
        output["a"] = [label(v) for v in data["a"]]
        output["b"] = [label(v) for v in data["b"]]
        output["cells"] = [
            {
                "a": c["a"],
                "b": c["b"],
                "a_label": output["a"][c["a"]],
                "b_label": output["b"][c["b"]],
                **({"count": c["count"]} if full else {}),
            }
            for c in data["cells"]
        ]
        output["caption"] = "Observed joint cells; blank cells are unobserved in this scope."
        predicates = [
            label(p["column"], column=True) + "=" + label(p["value"])
            for p in data.get("context", [])
        ]
        if predicates:
            output["caption"] += " Context: " + ", ".join(predicates)
    else:
        raise ValueError(f"No graphical renderer for result kind {kind!r}")
    return output
