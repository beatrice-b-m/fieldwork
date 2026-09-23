from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd
import pytest

from fieldwork import (
    ExplorerResult,
    KeySpec,
    census,
    explore,
    grain,
    infer_schema,
    joint_counts,
    levels,
    render_plaintext,
    visualization_data,
)


def test_resolved_census_names_parents_without_merging_repeated_levels():
    frame = pd.DataFrame({"site": ["North", "South"], "modality": ["CT", "CT"]})
    result = census(frame, ["site", "modality"])
    before = deepcopy(result.to_dict())
    resolved = result.to_dict(resolve_references=True)
    children = [n for n in resolved["tree"]["nodes"] if n["depth"] == 2]
    assert [n["label"] for n in children] == ["modality='CT'", "modality='CT'"]
    assert [n["parent_label"] for n in children] == ["site='North'", "site='South'"]
    assert len({n["parent_id"] for n in children}) == 2
    assert len({n["node_id"] for n in children}) == 2
    assert resolved["tree"]["dimension_labels"] == ["site", "modality"]
    assert render_plaintext(resolved) == render_plaintext(result)
    for raw, named in zip(before["tree"]["nodes"], resolved["tree"]["nodes"]):
        assert all(named[key] == value for key, value in raw.items())
    children[0]["value"]["value"] = "edited"
    resolved["source"]["dtypes"].clear()
    assert result.to_dict() == before == dict(result)


@pytest.mark.parametrize("make", [levels, lambda df: census(df, list(df.columns))])
def test_resolved_values_preserve_types_and_tuple_columns(make):
    column = ("finding", 1)
    values = [True, 1, 1.0, "1", None, "<NA>", float("inf"), pd.Timestamp("2026-01-01")]
    frame = pd.DataFrame({column: pd.Series(values, dtype=object)})
    result = make(frame)
    data = result.to_dict(resolve_references=True)
    records = data["per_feature"][0]["levels"] if result.kind == "levels" else data["tree"]["nodes"]
    assert {r["value"]["type"] for r in records} == {
        "boolean",
        "integer",
        "float",
        "string",
        "missing",
        "datetime_naive",
    }
    assert len({r["label"] for r in records}) == len(values)
    assert all(r["column"]["type"] == "tuple" for r in records)
    assert data == json.loads(json.dumps(data, allow_nan=False))
    assert data == make(frame.iloc[::-1]).to_dict(resolve_references=True)


@pytest.mark.parametrize("per_parent", [False, True])
@pytest.mark.parametrize("limits", [{"max_nodes": 0}, {"max_levels": 0}, {"min_count": 100}])
def test_pre_selection_metadata_resolves_without_emitted_nodes(per_parent, limits):
    frame = pd.DataFrame({"site": ["North", "North", "South"], "modality": ["CT", "MRI", "CT"]})
    result = census(
        frame,
        ["site", "modality"],
        top_n=1,
        top_n_mode="pre",
        top_n_per_parent=per_parent,
        **limits,
    )
    assert not result["tree"]["nodes"]
    assert len(result["level_dictionary"]) == 2
    retained = result.to_dict(resolve_references=True)["tree"]["retained_sets"]
    assert [r["column_label"] for r in retained] == ["site", "modality"]
    assert [r["values"] for r in retained] == [
        [{"type": "string", "value": "North"}],
        [{"type": "string", "value": "CT"}],
    ]
    if per_parent:
        assert retained[0]["path_values"] == []
        assert retained[1]["path_values"][0]["label"] == "site='North'"
    assert result["tree"]["root"]["count"] == 1


def test_global_pre_selection_keeps_values_excluded_by_other_dimensions():
    frame = pd.DataFrame({"a": ["x", "x", "y", "z"], "b": ["u", "v", "w", "u"]})
    result = census(frame, ["a", "b"], top_n=2, top_n_mode="pre")
    data = result.to_dict(resolve_references=True)
    assert all(n["label"] != "a='y'" for n in data["tree"]["nodes"])
    assert data["tree"]["retained_sets"][0]["labels"] == ["a='x'", "a='y'"]


def test_combined_warnings_use_section_columns_with_different_feature_orders():
    frame = pd.DataFrame({"a": pd.Series([1, "x"], dtype=object), "b": [1, 2]})
    result = explore(frame, ["b", "a"], features=["a", "b"], schema={"b": "id"})
    data = result.to_dict(resolve_references=True)
    assert [w["column_label"] for w in data["warnings"]] == ["a", "b", "b", "a"]
    for section in ("levels", "census"):
        assert all("column_label" in w for w in data["sections"][section]["warnings"])
    assert "feature_id=" not in render_plaintext(result)


def test_resolved_grain_graph_and_joint_cells_are_self_contained():
    frame = pd.DataFrame({"site": ["N", "N", "S"], "id": [1, 2, 3], "finding": ["x", "y", "x"]})
    result = grain(frame, ["site", KeySpec("exam", ("site", "id"))])
    data = result.to_dict(resolve_references=True)
    edge = data["graph"]["edges"][0]
    assert edge["source_keys"] == ["site"] and edge["target_keys"] == ["exam"]
    finding = next(a for a in data["graph"]["assignments"] if a["target_label"] == "finding")
    assert finding["node_keys"] == [["exam"]]
    joint = joint_counts(frame, ["id", "finding"], context={"site": "N"})
    cells = joint.to_dict(resolve_references=True)["cells"]
    assert [c["label"] for c in cells] == ["id=1, finding='x'", "id=2, finding='y'"]
    assert cells[0]["a_value"] == {"type": "integer", "value": "1"}
    assert cells[0]["a_column"] == {"type": "string", "value": "id"}
    assert "site='N'" in str(joint)


def test_resolved_combined_pairs_and_unrequested_sections():
    frame = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    result = explore(frame, ["a", "b"], include_absence=True)
    data = result.to_dict(resolve_references=True)
    assert data["sections"]["grain"] == {"status": "not_requested"}
    pair = data["sections"]["pairs"]["pairs"][0]
    assert pair["column_labels"] == ["a", "b"]
    assert [e["label"] for e in pair["absence"]["examples"]] == ["a='x', b=2", "a='y', b=1"]
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    proposal = infer_schema(frame).to_dict(resolve_references=True)
    assert proposal["proposals"][0]["column_label"] == "a"


def test_interactive_display_is_bounded_safe_and_does_not_serialize(monkeypatch):
    frame = pd.DataFrame({"finding": [f"value-{n}\x1b" for n in range(200)]})
    result = levels(frame, max_levels=None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Interactive display must not serialize the entire result")

    full = render_plaintext(result, max_lines=10_000)
    monkeypatch.setattr(ExplorerResult, "to_dict", forbidden)
    text = repr(result)
    assert str(result) == text
    assert "finding" in text and "feature_id" not in text
    # The interactive display is a bounded prefix of the full rendering.
    lines = text.splitlines()
    assert len(lines) < len(full.splitlines()) / 4
    assert lines[:-1] == full.splitlines()[: len(lines) - 1]
    assert "\x1b" not in text

    class Printer:
        def text(self, value):
            self.value = value

    printer = Printer()
    result._repr_pretty_(printer, False)
    assert printer.value == text
    result._repr_pretty_(printer, True)
    assert printer.value == "ExplorerResult(...)"


@pytest.mark.parametrize("detail", ["full", "topology"])
def test_projection_references_have_labels_including_omission_parents(detail):
    frame = pd.DataFrame({"site": ["N", "N", "S"], "id": [1, 2, 3]})
    tree = visualization_data(census(frame, ["site", "id"], max_nodes=3), detail=detail)
    labels = {r["id"]: r["label"] for r in tree["rows"]}
    assert tree["rows"][0]["parent_label"] is None
    assert any(r.get("omitted") for r in tree["rows"])
    assert all(r["parent_label"] == labels[r["parent"]] for r in tree["rows"][1:])
    graph = visualization_data(grain(frame, ["site", "id"]), detail=detail)
    assert graph["edges"][0]["source_label"] == "site"
    assert graph["edges"][0]["target_label"] == "id"
    assert all(r["feature_label"] in {"site", "id"} for r in graph["evidence"])
    joint = visualization_data(joint_counts(frame, ["site", "id"]), detail=detail)
    assert joint["cells"][0]["a_label"] == "'N'"
    pair = visualization_data(explore(frame, ["site", "id"]), section="pairs", detail=detail)
    cell = pair["contexts"][0]["cells"][0]
    assert cell["a_label"] == "site" and cell["b_label"] == "id"


def test_empty_results_have_readable_displays_and_resolved_exports():
    frame = pd.DataFrame({"a": [], "b": []})
    for result in (
        levels(frame),
        census(frame, ["a"]),
        grain(frame, ["a"]),
        joint_counts(frame, ["a", "b"]),
        infer_schema(frame),
        explore(frame, ["a"]),
    ):
        assert "Unsupported" not in repr(result)
        assert "empty" in repr(result)
        json.dumps(result.to_dict(resolve_references=True), allow_nan=False)
