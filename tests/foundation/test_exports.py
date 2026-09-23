from __future__ import annotations

import json

import pandas as pd
import pytest

from fieldwork import (
    KeySpec,
    Result,
    Scope,
    census,
    grain,
    infer_schema,
    joint_counts,
    levels,
    profile,
    render_plaintext,
    visualization_data,
)


def test_census_nodes_name_their_own_values_and_parents():
    frame = pd.DataFrame({"site": ["North", "South"], "modality": ["CT", "CT"]})
    result = census(frame, ["site", "modality"])
    nodes = {n["node_id"]: n for n in result["tree"]["nodes"]}
    children = [n for n in nodes.values() if n["depth"] == 2]
    # A repeated level under two parents stays two nodes.
    assert [(n["column"], n["value"]) for n in children] == [("modality", "CT")] * 2
    assert [nodes[n["parent_id"]]["value"] for n in children] == ["North", "South"]
    assert result["tree"]["dimensions"] == ["site", "modality"]


@pytest.mark.parametrize("make", [levels, lambda df: census(df, list(df.columns))])
def test_exported_values_keep_types_and_str_column_names(make):
    column = ("finding", 1)
    values = [True, 1, 1.0, "1", None, "<NA>", float("inf"), pd.Timestamp("2026-01-01")]
    frame = pd.DataFrame({column: pd.Series(values, dtype=object)})
    result = make(frame)
    data = result.to_dict()
    records = data["per_feature"][0]["levels"] if result.kind == "levels" else data["tree"]["nodes"]
    exported = [r["value"] for r in records]
    assert len(exported) == len(values)
    assert {type(v) for v in exported} == {bool, int, float, str, type(None)}
    rows = visualization_data(result)
    labels = [
        r["label"]
        for r in (rows["features"][0]["rows"] if "features" in rows else rows["rows"][1:])
    ]
    assert len(set(labels)) == len(values)
    column_name = (
        data["per_feature"][0]["column"]
        if result.kind == "levels"
        else data["tree"]["dimensions"][0]
    )
    assert column_name == str(column)
    assert data == json.loads(json.dumps(data, allow_nan=False))
    reversed_data = make(frame.iloc[::-1]).to_dict()
    # Row order changes the source identity, never the evidence.
    assert data.pop("source") != reversed_data.pop("source")
    assert data == reversed_data


@pytest.mark.parametrize("per_parent", [False, True])
@pytest.mark.parametrize("limits", [{"max_nodes": 0}, {"max_levels": 0}, {"min_count": 100}])
def test_pre_selection_reports_kept_values_without_emitted_nodes(per_parent, limits):
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
    retained = result["tree"]["retained_sets"]
    assert [(r["column"], r["values"]) for r in retained] == [
        ("site", ["North"]),
        ("modality", ["CT"]),
    ]
    if per_parent:
        assert [r["path"] for r in retained] == [[], ["North"]]
    assert result["tree"]["root"]["count"] == 1


def test_global_pre_selection_keeps_values_excluded_by_other_dimensions():
    frame = pd.DataFrame({"a": ["x", "x", "y", "z"], "b": ["u", "v", "w", "u"]})
    result = census(frame, ["a", "b"], top_n=2, top_n_mode="pre")
    assert all(n["value"] != "y" for n in result["tree"]["nodes"] if n["column"] == "a")
    assert result["tree"]["retained_sets"][0]["values"] == ["x", "y"]


def test_combined_warnings_name_section_columns_with_different_feature_orders():
    frame = pd.DataFrame({"a": pd.Series([1, "x"], dtype=object), "b": [1, 2]})
    result = profile(frame, ["b", "a"], features=["a", "b"], schema={"b": "id"})
    assert [w["column"] for w in result["warnings"]] == ["a", "b", "b", "a"]
    assert "feature_id=" not in render_plaintext(result)


def test_grain_graph_and_joint_cells_reference_their_records():
    frame = pd.DataFrame({"site": ["N", "N", "S"], "id": [1, 2, 3], "finding": ["x", "y", "x"]})
    graph = grain(frame, ["site", KeySpec("exam", ("site", "id"))])["graph"]
    keys = {node["id"]: node["keys"] for node in graph["nodes"]}
    edge = graph["edges"][0]
    assert (keys[edge["source"]], keys[edge["target"]]) == (["site"], ["exam"])
    finding = next(a for a in graph["assignments"] if a["target"] == "finding")
    assert [keys[node] for node in finding["nodes"]] == [["exam"]]
    joint = joint_counts(frame, ["id", "finding"], context={"site": "N"})
    cells = [(joint["a"][c["a"]], joint["b"][c["b"]]) for c in joint["cells"]]
    assert cells == [(1, "x"), (2, "y")]
    assert joint["context"] == [{"column": "site", "value": "N"}]
    assert "site=N" in str(joint)


def test_combined_pairs_absence_and_unrequested_sections():
    frame = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    result = profile(frame, ["a", "b"], include_absence=True)
    data = result.to_dict()
    assert data["sections"]["grain"] == {"status": "not_requested"}
    pair = data["sections"]["pairs"]["pairs"][0]
    assert pair["columns"] == ["a", "b"]
    assert pair["absence"]["examples"] == [{"a": "x", "b": 2}, {"a": "y", "b": 1}]
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert infer_schema(frame)["proposals"][0]["column"] == "a"


def test_interactive_display_is_bounded_safe_and_does_not_serialize(monkeypatch):
    frame = pd.DataFrame({"finding": [f"value-{n}\x1b" for n in range(200)]})
    result = levels(frame, max_levels=None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Interactive display must not serialize the entire result")

    full = render_plaintext(result, max_lines=10_000)
    monkeypatch.setattr(Result, "to_dict", forbidden)
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
    assert printer.value == "Result(...)"


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
    assert joint["cells"][0]["a_label"] == "N"
    pair = visualization_data(profile(frame, ["site", "id"]), section="pairs", detail=detail)
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
        profile(frame, ["a"]),
    ):
        assert "Unsupported" not in repr(result)
        assert "empty" in repr(result)
        json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "make",
    [
        lambda df, **context: levels(df, ["site"], dropna=True, **context),
        lambda df, **context: census(df, ["site", "id"], top_n=1, top_n_mode="pre", **context),
        lambda df, **context: grain(df, ["site", KeySpec("both", ("site", "id"))], **context),
        lambda df, **context: joint_counts(df, ["site", "finding"], context={"id": 1}, **context),
        lambda df, **context: infer_schema(df, candidate_keys=["id"], **context),
    ],
    ids=["levels", "census", "grain", "joint_counts", "infer_schema"],
)
def test_saved_foundation_results_recompute_with_their_source_context(make):
    frame = pd.DataFrame(
        {"site": ["N", "N", "S", "-"], "id": [1, 2, 1, 2], "finding": ["x", "y", "x", "x"]}
    )
    context = {
        "scope": Scope.from_positions(frame, [0, 1, 2], name="kept"),
        "missing": {"site": ["-"]},
        "table_id": "delivery",
    }
    result = make(frame, **context)
    saved = Result.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
    assert saved.recompute(frame).to_dict() == result.to_dict()
