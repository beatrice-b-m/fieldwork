from __future__ import annotations

import json
from copy import deepcopy
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

import pandas as pd
import pytest

from fieldwork import (
    KeySpec,
    census,
    explore,
    grain,
    joint_counts,
    levels,
    render_html,
    render_svg,
    visualization_data,
)


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


def test_every_surface_is_valid_svg_and_standalone_html(frame):
    results = [
        grain(frame, ["exam_id", KeySpec("exam_side", ("exam_id", "side"))]),
        levels(frame, ["side"], top_n=1),
        census(frame, ["side", "finding"], max_nodes=3),
        explore(frame, ["side", "finding"])["sections"]["pairs"],
        joint_counts(frame, ["side", "finding"]),
    ]
    for result in results:
        before = deepcopy(dict(result))
        for detail in ("full", "topology"):
            svg = render_svg(result, detail=detail)
            assert ET.fromstring(svg).tag.endswith("svg")
            document = render_html(result, detail=detail)
            assert document.startswith("<!doctype html>")
            assert document.count("<script>") == 1
            assert "<script src=" not in document and "<link " not in document
            json.dumps(visualization_data(result, detail=detail), allow_nan=False)
        assert dict(result) == before


def test_topology_exports_are_invariant_to_quantities(frame):
    # Duplication changes every support count and prevalence; qualitative structure is unchanged.
    changed = pd.concat([frame, frame.iloc[[0] * 23]], ignore_index=True)
    for make in [
        lambda df: grain(df, ["exam_id", KeySpec("both", ("exam_id", "side"))]),
        lambda df: levels(df, ["side", "finding"]),
        lambda df: census(df, ["side", "finding"]),
        lambda df: explore(df, ["side", "finding"])["sections"]["pairs"],
        lambda df: joint_counts(df, ["side", "finding"]),
    ]:
        first, second = make(frame), make(changed)
        assert visualization_data(first, detail="topology") == visualization_data(
            second, detail="topology"
        )
        assert render_svg(first, detail="topology") == render_svg(second, detail="topology")
        assert render_html(first, detail="topology") == render_html(second, detail="topology")
        assert render_svg(first) != render_svg(second)


def test_topology_allowlist_filters_even_hidden_payloads(frame):
    result = grain(frame, ["exam_id"]).to_dict()
    for node in result["graph"]["nodes"]:
        node["evaluated_groups"] = 987654321
    for evidence in result["graph"]["dependencies"]:
        evidence["singleton_groups"] = 987654321
    result["sensitive_unrecognized_field"] = "secret-value"
    for output in (
        render_svg(result, detail="topology", show_exceptions=True),
        render_html(result, detail="topology"),
        json.dumps(visualization_data(result, detail="topology")),
    ):
        assert "987654321" not in output
        assert "secret-value" not in output
        assert '"singleton_groups"' not in output
        assert '"group_rate"' not in output
    assert "987654321" in render_html(result)


def test_main_graph_reduction_shared_features_and_matrix(frame):
    graph = grain(frame, ["exam_id", KeySpec("both", ("exam_id", "side"))])
    svg = render_svg(graph, show_exceptions=True)
    text = " ".join(ET.fromstring(svg).itertext())
    assert "Arrows mean finer grouping" in text
    assert "both = (exam_id, side)" in text
    assert "5 groups · 6 rows" in text
    assert "Varies: finding · 1/3 groups" in text
    matrix = render_svg(graph, view="matrix")
    assert "varying" in matrix and "constant" in matrix
    document = render_html(graph)
    assert 'data-evidence="f2"' in document
    assert 'id="focus"' in document and 'id="exceptions"' in document


def test_missing_population_is_not_silently_used_in_graph():
    frame = pd.DataFrame({"key": [1, 1, 2, 2], "target": ["x", None, "y", None]})
    result = grain(frame, ["key"], dropna=True)
    assert "Not placed by tested keys" in render_svg(result)
    assert "different target population" in render_svg(result)
    assert "Different target population" in render_html(result)
    assert "constant *" in render_svg(result, view="matrix")


def test_omitted_mass_keeps_original_denominator_and_missingness():
    frame = pd.DataFrame({"a": ["x"] * 3 + ["y", None], "b": ["z"] * 5})
    projected = visualization_data(levels(frame, ["a"], top_n=1))
    rows = projected["features"][0]["rows"]
    assert [(r["count"], r["share"]) for r in rows] == [(3, 0.6), (2, 0.4)]
    tree = visualization_data(census(frame, ["a", "b"], max_nodes=1))
    assert tree["rows"][0]["count"] == 5
    omitted = [r for r in tree["rows"] if r.get("omitted")]
    assert omitted and omitted[0]["share"] == 0.4
    assert "Omitted branches" in render_html(census(frame, ["a"], max_nodes=0), detail="topology")
    assert "1 missing" in render_svg(levels(frame, ["a"], dropna=True))


def test_pairs_directions_contexts_and_separate_association(frame):
    result = explore(frame, ["exam_id", "finding"], pair_contexts=[{"site": "north"}])
    data = visualization_data(result, section="pairs")
    assert len(data["contexts"]) == 2
    assert data["contexts"][1]["label"] == "site='north'"
    assert "n:1" in render_svg(result, section="pairs")
    assert "Pair association" in render_svg(result, section="pairs", view="association")
    assert 'id="context"' in render_html(result, section="pairs")
    with pytest.raises(ValueError, match="requires detail"):
        render_svg(result, section="pairs", detail="topology", view="association")


def test_typed_labels_escaping_and_no_injected_markup():
    name = '</text><script>alert("oops")</script>\x00'
    values = pd.Series([True, 1, "1", None], dtype=object)
    frame = pd.DataFrame({name: values})
    result = levels(frame)
    data = visualization_data(result, detail="topology")
    assert {r["label"] for r in data["features"][0]["rows"]} == {"True", "1", "'1'", "<NA>"}
    for result in [levels(frame), grain(frame, [KeySpec(name, (name,))])]:
        for view in [None, "matrix"] if result.kind == "grain" else [None]:
            svg = render_svg(result, view=view)
            root = ET.fromstring(svg)
            assert not list(root.iter("{http://www.w3.org/2000/svg}script"))
        document = render_html(result)
        assert document.count("<script>") == 1
        assert "\x00" not in document


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
        render_svg(explore(frame, ["a"]))
    with pytest.raises(ValueError, match="recompute"):
        render_svg({"kind": "grain"})
    with pytest.raises(ValueError, match="detail"):
        render_html(levels(frame), detail="private")
    with pytest.raises(ValueError, match="Invalid view"):
        render_svg(levels(frame), view="map")


def test_html_ids_unique(frame):
    class IDs(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []

        def handle_starttag(self, tag, attrs):
            self.ids.extend(value for key, value in attrs if key == "id")

    parser = IDs()
    parser.feed(render_html(grain(frame, ["exam_id"])))
    assert len(parser.ids) == len(set(parser.ids))


def test_pair_budget_preserves_untested_matrix_features_and_contexts(frame):
    result = explore(
        frame, ["exam_id", "side", "finding"], max_pairs=0, pair_contexts=[{"site": "north"}]
    )
    data = visualization_data(result, section="pairs")
    assert data["features"] == ["exam_id", "finding", "side"]
    assert len(data["contexts"]) == 2
    svg = render_svg(result, section="pairs", detail="topology")
    assert "untested" in svg and "omitted" in svg


def test_joint_context_is_visible_in_figure(frame):
    result = joint_counts(frame, ["side", "finding"], context={"site": "north"})
    assert "site='north'" in visualization_data(result)["caption"]


def test_multiple_pair_svgs_do_not_repeat_document_ids(frame):
    document = render_html(
        explore(frame, ["side", "finding"], pair_contexts=[{"site": "north"}]), section="pairs"
    )

    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []

        def handle_starttag(self, tag, attrs):
            self.ids.extend(value for key, value in attrs if key == "id")

    parser = Parser()
    parser.feed(document)
    assert len(parser.ids) == len(set(parser.ids))
