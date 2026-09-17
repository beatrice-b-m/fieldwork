"""Context adaptation preserves graph references and shared population records."""

import json
from xml.etree import ElementTree

import pandas as pd
import pytest

import fieldwork as fw
from fieldwork._explore.resolved import resolve_result


@pytest.mark.parametrize("operation", ["discovered", "scoped_discovered", "scoped_explicit"])
def test_contextualized_grain_edges_render_and_resolve(operation):
    df = pd.DataFrame({"site": ["A", "A", "B", "B"], "exam": [1, 2, 3, 4]})
    context = {"table_id": "delivery"}
    if operation.startswith("scoped"):
        context["scope"] = fw.Scope.from_positions(df, [0, 1, 2], name="selected")
    if operation == "scoped_explicit":
        result = fw.explore(df, ["site", "exam"], candidate_keys=["site", "exam"], **context)
        grain = result["sections"]["grain"]
        assert result.to_dict(resolve_references=True)["sections"]["grain"]["graph"]["edges"]
        for section in result["sections"].values():
            assert section["source"]["table_id"] == "delivery"
            assert section["source"]["rows"] == len(df)
    else:
        result = fw.discover_dependencies(df, max_key_size=1, **context)
        grain = result["exact_grain"]
    assert grain["source"]["table_id"] == "delivery"
    assert grain["source"]["rows"] == len(df)
    assert len(grain["graph"]["edges"]) == 1
    nodes = {node["id"]: node for node in grain["graph"]["nodes"]}
    edge = grain["graph"]["edges"][0]
    assert edge["relation"] == "finer_grouping"
    assert isinstance(edge["source"], str) and edge["source"] in nodes
    assert isinstance(edge["target"], str) and edge["target"] in nodes
    for saved in (grain, json.loads(json.dumps(grain, allow_nan=False))):
        resolved = resolve_result(saved)
        resolved_edge = resolved["graph"]["edges"][0]
        assert resolved_edge["source_keys"] == nodes[edge["source"]]["keys"]
        assert resolved_edge["target_keys"] == nodes[edge["target"]]["keys"]
        for detail in ("full", "topology"):
            ElementTree.fromstring(fw.render_svg(saved, detail=detail))
            assert "<html" in fw.render_html(saved, detail=detail)
            assert fw.render_plaintext(saved, detail=detail)


@pytest.mark.parametrize("applies_to", ["census", "both"])
@pytest.mark.parametrize("dropna", [False, True])
def test_scoped_pre_filter_rebases_shared_populations_once(applies_to, dropna):
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
    census = result["sections"]["census"]["scopes"][0]
    pairs = result["sections"]["pairs"]
    grain = result["sections"]["grain"]
    # These are aliases in the composed result, not independent scope copies.
    assert pairs["scope_metadata"]["scope"] is census
    if applies_to == "both":
        assert grain["scope_metadata"]["scope"] is census
    assert census["input_rows"] == 4
    assert census["evaluated_rows"] == census["retained_rows"] == 1
    assert census["missing_excluded_rows"] == int(dropna)
    assert census["restriction_excluded_rows"] == 3 - int(dropna)
    assert census["lineage"] == [
        "selected",
        "input",
        "dropna" if dropna else "include_missing",
        "pre",
    ]
    pair = pairs["pairs"][0]["scope"]
    assert pair["evaluated_rows"] == 1
    assert pair["missing_excluded_rows"] == census["missing_excluded_rows"]
    assert pair["restriction_excluded_rows"] == census["restriction_excluded_rows"]
    graph_scope = grain["graph"]["scope"]
    assert graph_scope["evaluated_rows"] == (1 if applies_to == "both" else 3 - int(dropna))
    assert graph_scope["missing_excluded_rows"] == int(dropna)
    assert graph_scope["restriction_excluded_rows"] == (
        3 - int(dropna) if applies_to == "both" else 1
    )

    def check_populations(value):
        if isinstance(value, dict):
            if "scope_id" in value and "input_rows" in value:
                assert value["input_rows"] == len(df)
                assert value["input_rows"] == sum(
                    value[k]
                    for k in (
                        "evaluated_rows",
                        "missing_excluded_rows",
                        "restriction_excluded_rows",
                    )
                )
                assert value["lineage"].count("selected") == 1
            for child in value.values():
                check_populations(child)
        elif isinstance(value, list):
            for child in value:
                check_populations(child)

    # Check both shared in-memory records and the independently saved copies.
    check_populations(result.to_dict())
    check_populations(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
