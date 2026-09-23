"""Self-contained analytical exports without changing local reference identities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .visual_data import label


def _predicate(column: dict, value: dict) -> dict:
    return {
        "column": column,
        "value": value,
        "label": f"{label(column, column=True)}={label(value)}",
    }


def _resolve(data: dict) -> None:
    kind = data.get("kind")
    if data.get("status") == "not_requested":
        return
    for section in data.get("sections", {}).values():
        _resolve(section)

    features = data.get("per_feature", []) if kind == "levels" else data.get("features", [])
    columns = {
        f["feature_id"]: f["column"] for f in features if isinstance(f, dict) and "feature_id" in f
    }
    for warning in data.get("warnings", []):
        column = warning.get("column", columns.get(warning.get("feature_id")))
        if column is not None:
            warning["column"] = column
            warning["column_label"] = label(column, column=True)

    if kind == "levels":
        for feature in features:
            feature["column_label"] = label(feature["column"], column=True)
            for level in feature["levels"]:
                level.update(_predicate(feature["column"], level["value"]))
    elif kind == "census":
        values = {entry["level_id"]: entry["value"] for entry in data["level_dictionary"]}
        tree = data["tree"]
        tree["dimension_columns"] = [columns[f] for f in tree["dimensions"]]
        tree["dimension_labels"] = [label(c, column=True) for c in tree["dimension_columns"]]
        for feature in features:
            feature["column_label"] = label(feature["column"], column=True)
        for entry in data["level_dictionary"]:
            entry.update(_predicate(columns[entry["feature_id"]], entry["value"]))
        root = tree["root"]
        root.update(column=None, value=None, label="All evaluated rows", parent_label=None)
        nodes = {root["node_id"]: root}
        for node in tree["nodes"]:
            node.update(_predicate(columns[node["feature_id"]], values[node["level_id"]]))
            nodes[node["node_id"]] = node
        for node in tree["nodes"]:
            node["parent_label"] = nodes[node["parent_id"]]["label"]
        for retained in tree["retained_sets"]:
            depth = retained["depth"] - 1

            def predicate_at(index: int, code: int) -> dict:
                feature_id = tree["dimensions"][index]
                level_id = f"{feature_id}:l{code}"
                if level_id not in values:
                    raise ValueError(
                        "Retained census values are missing from this saved result; "
                        "recompute census() before resolving references"
                    )
                return _predicate(columns[feature_id], values[level_id])

            retained["column"] = tree["dimension_columns"][depth]
            retained["column_label"] = tree["dimension_labels"][depth]
            chosen = [predicate_at(depth, code) for code in retained["level_codes"]]
            retained["values"] = [p["value"] for p in chosen]
            retained["labels"] = [p["label"] for p in chosen]
            if "path" in retained:
                retained["path_values"] = [
                    predicate_at(index, code) for index, code in enumerate(retained["path"])
                ]
    elif kind == "grain":
        for key in data["keys"]:
            key["column_labels"] = [label(c, column=True) for c in key["columns"]]
        for record in [*data["dependencies"], *data["targets"]]:
            record["target_label"] = label(record["target"], column=True)
        graph = data.get("graph")
        if graph is not None:
            keys = {node["id"]: node["keys"] for node in graph["nodes"]}
            for edge in graph["edges"]:
                edge.update(source_keys=keys[edge["source"]], target_keys=keys[edge["target"]])
            for record in [*graph["dependencies"], *graph["assignments"], *graph["unplaced"]]:
                record["target_label"] = label(record["target"], column=True)
            for assignment in graph["assignments"]:
                assignment["node_keys"] = [keys[node] for node in assignment["nodes"]]
    elif kind == "pairs":
        for pair in data["pairs"]:
            pair["column_labels"] = [label(c, column=True) for c in pair["columns"]]
            for predicate in pair["context"]:
                predicate.update(_predicate(predicate["column"], predicate["value"]))
            for example in pair.get("absence", {}).get("examples", []):
                example["a_column"], example["b_column"] = pair["columns"]
                example["label"] = ", ".join(
                    _predicate(c, example[side])["label"]
                    for c, side in zip(pair["columns"], ("a", "b"))
                )
    elif kind == "joint_counts":
        data["column_labels"] = [label(c, column=True) for c in data["columns"]]
        for predicate in data["context"]:
            predicate.update(_predicate(predicate["column"], predicate["value"]))
        for cell in data["cells"]:
            cell["a_column"], cell["b_column"] = data["columns"]
            cell["a_value"], cell["b_value"] = data["a"][cell["a"]], data["b"][cell["b"]]
            cell["label"] = ", ".join(
                _predicate(c, cell[f"{side}_value"])["label"]
                for c, side in zip(data["columns"], ("a", "b"))
            )
    elif kind == "schema_proposal":
        for proposal in data["proposals"]:
            proposal["column_label"] = label(proposal["column"], column=True)


def resolve_result(data: dict[str, Any]) -> dict[str, Any]:
    """Return an independent, strict-JSON-compatible result with local reference values."""
    resolved = deepcopy(data)
    _resolve(resolved)
    return resolved
