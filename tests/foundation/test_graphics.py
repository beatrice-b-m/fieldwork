from __future__ import annotations

from xml.etree import ElementTree as ET

import pandas as pd
import pytest

from fieldwork import (
    KeySpec,
    census,
    grain,
    joint_counts,
    levels,
    profile,
    render_html,
    render_svg,
    visualization_data,
)

# Validity, non-mutation, escaping, HTML IDs and topology suppression for every
# result kind are covered once in tests/test_rendering_contracts.py.


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "exam_id": [1, 1, 2, 2, 3, 3],
            "side": ["L", "R", "L", "R", "L", "L"],
            "finding": ["clear", "scar", "clear", "clear", "scar", "scar"],
            "site": ["north", "south", "north", "south", "north", "south"],
        }
    )


def test_grain_projection_reduces_graph_and_places_features(frame):
    data = visualization_data(grain(frame, ["exam_id", KeySpec("both", ("exam_id", "side"))]))
    nodes = {node["key_names"][0]: node for node in data["nodes"]}
    assert [(e["source"], e["target"]) for e in data["edges"]] == [
        (nodes["exam_id"]["id"], nodes["both"]["id"])
    ]
    assert nodes["both"]["support"] == {
        "evaluated_rows": 6,
        "evaluated_groups": 5,
        "singleton_groups": 4,
        "repeated_groups": 1,
    }
    # finding varies within one of three exams, so it is placed at the finer key.
    assert nodes["both"]["attribute_labels"] == ["finding"]
    evidence = {(e["key"], e["feature_label"]): e for e in data["evidence"]}
    assert evidence["exam_id", "finding"]["state"] == "varying"
    assert (
        evidence["exam_id", "finding"]["violating_groups"],
        evidence["exam_id", "finding"]["evaluated_groups"],
    ) == (1, 3)
    assert evidence["both", "finding"]["state"] == "constant"
    ET.fromstring(render_svg(grain(frame, ["exam_id"]), view="matrix"))


def test_missing_target_population_is_not_used_in_graph():
    frame = pd.DataFrame({"key": [1, 1, 2, 2], "target": ["x", None, "y", None]})
    data = visualization_data(grain(frame, ["key"], dropna=True))
    target = next(f for f in data["features"] if f["label"] == "target")
    assert target["nodes"] == []
    assert target["reason"] == "different_target_population"
    evidence = next(e for e in data["evidence"] if e["feature_label"] == "target")
    assert evidence["compatible"] is False
    assert evidence["evaluated_rows"] == 2


def test_omitted_mass_keeps_original_denominator_and_missingness():
    frame = pd.DataFrame({"a": ["x"] * 3 + ["y", None], "b": ["z"] * 5})
    projected = visualization_data(levels(frame, ["a"], top_n=1))
    rows = projected["features"][0]["rows"]
    assert [(r["count"], r["share"]) for r in rows] == [(3, 0.6), (2, 0.4)]
    tree = visualization_data(census(frame, ["a", "b"], max_nodes=1))
    assert tree["rows"][0]["count"] == 5
    omitted = [r for r in tree["rows"] if r.get("omitted")]
    assert omitted and omitted[0]["share"] == 0.4


def test_pairs_directions_contexts_and_separate_association(frame):
    result = profile(frame, ["exam_id", "finding"], pairs={"pair_contexts": [{"site": "north"}]})
    data = visualization_data(result, section="pairs")
    assert len(data["contexts"]) == 2
    assert data["contexts"][1]["label"] == "site=north"
    ET.fromstring(render_svg(result, section="pairs", view="association"))
    with pytest.raises(ValueError, match="requires detail"):
        render_svg(result, section="pairs", detail="topology", view="association")


def test_typed_labels_stay_distinct():
    frame = pd.DataFrame({"v": pd.Series([True, 1, "1", None], dtype=object)})
    data = visualization_data(levels(frame), detail="topology")
    assert {r["label"] for r in data["features"][0]["rows"]} == {"True", "1", "'1'", "<NA>"}


def test_empty_inputs_and_actionable_errors():
    frame = pd.DataFrame({"a": [], "b": []})
    for result in [
        grain(frame, ["a", "b"]),
        levels(frame),
        census(frame, ["a"]),
        joint_counts(frame, ["a", "b"]),
    ]:
        ET.fromstring(render_svg(result))
        assert render_html(result)
    with pytest.raises(ValueError, match="was not computed"):
        render_svg(profile(frame, ["a"]))
    with pytest.raises(ValueError, match="recompute"):
        render_svg({"kind": "grain"})
    with pytest.raises(ValueError, match="detail"):
        render_html(levels(frame), detail="private")
    with pytest.raises(ValueError, match="Invalid view"):
        render_svg(levels(frame), view="map")


def test_pair_budget_preserves_untested_matrix_features_and_contexts(frame):
    result = profile(
        frame,
        ["exam_id", "side", "finding"],
        pairs={"pair_contexts": [{"site": "north"}], "limits": {"max_pairs": 0}},
    )
    data = visualization_data(result, section="pairs")
    assert data["features"] == ["exam_id", "finding", "side"]
    assert len(data["contexts"]) == 2
    ET.fromstring(render_svg(result, section="pairs", detail="topology"))


def test_joint_context_is_visible_in_figure(frame):
    result = joint_counts(frame, ["side", "finding"], context={"site": "north"})
    assert "site=north" in visualization_data(result)["caption"]
