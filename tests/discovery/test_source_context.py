"""Source context (scope, sentinels, table_id) applies alike to every analysis."""

import json
from xml.etree import ElementTree

import pandas as pd
import pytest

import fieldwork as fw


@pytest.mark.parametrize("label", [1, ("visit", (2, "code"))])
@pytest.mark.parametrize("context_kind", ["identity", "scope", "missing"])
@pytest.mark.parametrize("operation", [fw.census, fw.explore])
def test_source_context_applies_to_foundation_analyses(label, context_kind, operation):
    df = pd.DataFrame({label: [1, -999, 2], "value": [4, 5, 6]})
    original = df.copy(deep=True)
    context = {"table_id": "delivery"}
    expected = df.copy()
    if context_kind == "scope":
        context["scope"] = fw.Scope.from_positions(df, [0, 2], name="selected")
        expected = expected.iloc[[0, 2]]
    elif context_kind == "missing":
        context["missing"] = {label: [-999]}
        expected[label] = expected[label].astype(object).where(expected[label] != -999, None)
    result = operation(df, [label], **context)
    baseline = operation(expected, [label])
    census = result if operation == fw.census else result["sections"]["census"]
    expected_census = baseline if operation == fw.census else baseline["sections"]["census"]
    assert census["tree"] == expected_census["tree"]
    assert census["scope"]["input_rows"] == 3
    assert census["scope"]["restriction_excluded_rows"] == int(context_kind == "scope")
    saved = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert saved["source"]["table_id"] == "delivery"
    # Columns are named by str(label) everywhere, including sentinel conventions.
    sentinels = saved["missing_convention"]["sentinels"]
    assert sentinels.get(str(label), []) == ([-999] if context_kind == "missing" else [])
    assert fw.render_plaintext(saved)
    ElementTree.fromstring(fw.render_svg(saved, section="census"))
    pd.testing.assert_frame_equal(df, original)
    availability = fw.missingness(df, **context)["availability"]
    assert [row["feature"] for row in availability] == [str(label), "value"]


@pytest.mark.parametrize("operation", ["discovered", "scoped_discovered", "scoped_explicit"])
def test_contextualized_grain_edges_render_and_resolve(operation):
    df = pd.DataFrame({"site": ["A", "A", "B", "B"], "exam": [1, 2, 3, 4]})
    context = {"table_id": "delivery"}
    if operation.startswith("scoped"):
        context["scope"] = fw.Scope.from_positions(df, [0, 1, 2], name="selected")
    if operation == "scoped_explicit":
        result = fw.explore(df, ["site", "exam"], candidate_keys=["site", "exam"], **context)
        grain = result["sections"]["grain"]
        assert result["sections"]["grain"]["graph"]["edges"]
        for section in result["sections"].values():
            assert section["source"]["table_id"] == "delivery"
            assert section["source"]["rows"] == len(df)
    else:
        result = fw.discover_dependencies(df, max_key_size=1, **context)
        grain = result["grain_views"][0]["grain"]
    assert grain["source"]["table_id"] == "delivery"
    assert grain["source"]["rows"] == len(df)
    assert len(grain["graph"]["edges"]) == 1
    nodes = {node["id"]: node for node in grain["graph"]["nodes"]}
    edge = grain["graph"]["edges"][0]
    assert edge["relation"] == "finer_grouping"
    assert isinstance(edge["source"], str) and edge["source"] in nodes
    assert isinstance(edge["target"], str) and edge["target"] in nodes
    for saved in (grain, json.loads(json.dumps(grain, allow_nan=False))):
        keys = {node["id"]: node["keys"] for node in saved["graph"]["nodes"]}
        saved_edge = saved["graph"]["edges"][0]
        assert keys[saved_edge["source"]] == nodes[edge["source"]]["keys"]
        assert keys[saved_edge["target"]] == nodes[edge["target"]]["keys"]
        for detail in ("full", "topology"):
            ElementTree.fromstring(fw.render_svg(saved, detail=detail))
            assert "<html" in fw.render_html(saved, detail=detail)
            assert fw.render_plaintext(saved, detail=detail)


@pytest.mark.parametrize("applies_to", ["census", "both"])
@pytest.mark.parametrize("dropna", [False, True])
def test_scoped_pre_selection_cohort_counts_rows_once(applies_to, dropna):
    df = pd.DataFrame({"site": ["A", "B", None, "C"], "exam": [1, 2, 3, 4]})
    scope = fw.Scope.from_positions(df, [0, 1, 2], name="selected")
    result = fw.explore(
        df,
        ["site", "exam"],
        scope=scope,
        candidate_keys=["site", "exam"],
        top_n=1,
        top_n_mode="pre",
        top_n_applies_to=applies_to,
        dropna=dropna,
    )
    census = result["sections"]["census"]
    pairs = result["sections"]["pairs"]
    grain = result["sections"]["grain"]
    # The census analyzes the selected scope; its pre-selection keeps one row.
    assert census["scope"]["name"] == "selected"
    assert census["scope"]["restriction_excluded_rows"] == 1
    assert census["tree"]["evaluated_rows"] == 1
    assert census["tree"]["missing_excluded_rows"] == int(dropna)
    assert census["tree"]["restriction_excluded_rows"] == 2 - int(dropna)
    # Pairs (and grain, when the cohort applies to both) analyze that one row,
    # as a scope refining the selection.
    cohorts = [pairs, grain] if applies_to == "both" else [pairs]
    for section in cohorts:
        assert section["scope"]["name"] == "census top_n cohort"
        assert section["scope"]["parent"] == "selected"
        assert (section["scope"]["input_rows"], section["scope"]["evaluated_rows"]) == (4, 1)
    if applies_to == "census":
        assert grain["scope"]["name"] == "selected"
    assert pairs["pairs"][0]["evaluated_rows"] == 1
    assert grain["graph"]["evaluated_rows"] == (1 if applies_to == "both" else 3 - int(dropna))

    def check_partitions(section):
        # Every record counts evaluated and excluded rows within its section's scope.
        records = [*section.get("pairs", []), *section.get("dependencies", [])]
        records += [section["graph"]] if "graph" in section else []
        for record in records:
            assert section["scope"]["evaluated_rows"] == (
                record["evaluated_rows"]
                + record["missing_excluded_rows"]
                + record.get("restriction_excluded_rows", 0)
            )

    for data in (result.to_dict(), json.loads(json.dumps(result.to_dict(), allow_nan=False))):
        for name in ("pairs", "grain"):
            check_partitions(data["sections"][name])
