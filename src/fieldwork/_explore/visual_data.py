"""Allowlisted presentation data; disclosure filtering precedes serialization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Literal

from .encoding import display_scalar
from .render import _CONTROL, _identity
from .result import ExplorerResult


def label(value: Mapping, *, column: bool = False) -> str:
    if column and value["type"] == "string" and value["value"].isidentifier():
        text = value["value"]
    else:
        text = display_scalar(_identity(value), "<NA>")
    return _CONTROL.sub(lambda match: f"\\u{ord(match.group()):04x}", text)


def _scope(scope: Mapping, full: bool) -> str:
    text = "Conditional cohort" if scope.get("conditional") else "Input population"
    if full:
        text += (
            f" · {scope['evaluated_rows']} evaluated / {scope['input_rows']} input rows"
            f" · {scope['missing_excluded_rows']} missing, "
            f"{scope['restriction_excluded_rows']} restricted exclusions"
        )
    return text


def visualization_data(
    result: ExplorerResult | Mapping[str, Any],
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
    data = result.to_dict() if isinstance(result, ExplorerResult) else result
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
    output = {"kind": kind, "detail": detail}
    if kind == "grain":
        if "graph" not in data:
            raise ValueError("Grain graph evidence is missing; recompute grain() with schema 0.3+")
        graph = data["graph"]
        output["caption"] = "Observed groupings among tested keys; not a semantic entity model."
        output["scope"] = _scope(graph["scope"], full)
        output["missingness"] = graph["missingness"].replace("_", " ")
        names = {key["name"]: key for key in data["keys"]}
        feature_ids = {str(a["target"]): f"f{i}" for i, a in enumerate(graph["assignments"])}
        output["features"] = [
            {
                "id": feature_ids[str(a["target"])],
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
                "attributes": [feature_ids[str(a)] for a in node["attributes"]],
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
        for record in graph["dependencies"]:
            projected = {
                "feature": feature_ids[str(record["target"])],
                "key": record["key_name"],
                "state": (
                    "undefined"
                    if record["holds"] is None
                    else "constant"
                    if record["holds"]
                    else "varying"
                ),
                "compatible": record["scope_compatible"],
                "scope": _scope(record["scope"], full),
            }
            if full:
                projected.update(
                    {
                        k: record[k]
                        for k in (
                            "evaluated_rows",
                            "evaluated_groups",
                            "violating_groups",
                            "group_rate",
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
        scopes = {s["scope_id"]: s for s in data["scopes"]}
        output["features"] = []
        for feature in data["per_feature"]:
            rows = feature["levels"]
            if not full:
                rows = sorted(rows, key=lambda r: _identity(r["value"]).sort_key())
            projected = {
                "label": label(feature["column"], column=True),
                "scope": _scope(scopes[feature["scope_id"]], full),
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
                    total = scopes[feature["scope_id"]]["evaluated_rows"]
                    item.update(
                        count=feature["unreported_rows"],
                        share=feature["unreported_rows"] / total if total else None,
                    )
                projected["rows"].append(item)
            output["features"].append(projected)
    elif kind == "census":
        output["scope"] = _scope(data["scopes"][0], full)
        output["caption"] = "Observed paths; omitted branches retain their original mass."
        columns = {f["feature_id"]: label(f["column"], column=True) for f in data["features"]}
        values = {v["level_id"]: v["value"] for v in data["level_dictionary"]}
        children = defaultdict(list)
        for node in data["tree"]["nodes"]:
            children[node["parent_id"]].append(node)
        if not full:
            for siblings in children.values():
                siblings.sort(key=lambda n: _identity(values[n["level_id"]]).sort_key())
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
                else (f"{columns[node['feature_id']]}={label(values[node['level_id']])}"),
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
        columns = {str(c): c for c in data.get("features", [])}
        for record in data["pairs"]:
            for column in record["columns"]:
                columns[str(column)] = column
        ordered = sorted(columns.values(), key=lambda c: _identity(c).sort_key())
        output["features"] = [label(c, column=True) for c in ordered]
        indexes = {str(c): i for i, c in enumerate(ordered)}
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
                "a": indexes[str(record["columns"][0])],
                "b": indexes[str(record["columns"][1])],
                "a_label": label(record["columns"][0], column=True),
                "b_label": label(record["columns"][1], column=True),
                "relation": record["relation"] or "undefined",
                "scope": _scope(record["scope"], full),
            }
            if full:
                item.update(
                    association=record["cramers_v"], association_reason=record["cramers_v_reason"]
                )
            contexts[context]["cells"].append(item)
        output["omitted"] = bool(data["omitted_pairs"] or data["omitted_contexts"])
        output["caption"] = "Row grouping → column grouping. Mapping and association are separate."
    elif kind == "joint_counts":
        output["scope"] = _scope(data["scopes"][0], full)
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
