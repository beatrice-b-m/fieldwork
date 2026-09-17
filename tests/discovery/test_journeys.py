"""Regression journeys across discovery, inspection, scope and saved presentation."""

import json

import pandas as pd
import pytest

import fieldwork as fw


def test_recommendation_census_preserves_context_and_original_population():
    df = pd.DataFrame({"site": ["A", "A", "B", "B"], "value": [1, -999, 2, 3]}, index=[0] * 4)
    scope = fw.Scope.from_positions(df, [0, 1, 2], name="selected")
    paths = fw.suggest_paths(df, scope=scope, missing={"value": [-999]}, start_with=["value"])
    paths = fw.InvestigationResult.from_dict(json.loads(json.dumps(paths.to_dict())))
    tree = paths.best.census(df, dropna=True)
    assert tree["source"]["rows"] == 4
    assert tree["scopes"][0]["input_rows"] == 4
    assert tree["scopes"][0]["evaluated_rows"] == 2
    assert tree["scopes"][0]["missing_excluded_rows"] == 1
    assert tree["scopes"][0]["restriction_excluded_rows"] == 1
    preview = paths["paths"][0]["preview"]
    assert preview["scopes"][0]["input_rows"] == 4
    assert preview["scopes"][0]["evaluated_rows"] == 3
    assert {"type": "missing"} in [v["value"] for v in preview["level_dictionary"]]
    assert paths.best.census(df)["tree"] == preview["tree"]
    with pytest.raises(ValueError, match="differs"):
        paths.best.census(df.iloc[::-1])
    with pytest.raises(ValueError, match="context"):
        paths.best.census(df, missing={})
    explicit = fw.explore(
        df, ["value"], discovery={"scope": scope, "missing": {"value": [-999]}}, dropna=True
    )
    for name in ("levels", "census", "pairs"):
        for population in explicit["sections"][name]["scopes"]:
            assert population["input_rows"] == 4
            assert population["restriction_excluded_rows"] == 1
    with pytest.raises(ValueError, match="search options"):
        fw.explore(df, ["site"], discovery={"objective": "compact"})


def test_sparse_candidates_keep_compatible_grain_views():
    df = pd.DataFrame(
        {
            "entity": [1, 1, 2, 2],
            "site": ["A", "A", "B", "B"],
            "left": [1, 1, None, None],
            "right": [None, None, 2, 2],
            "never": [None] * 4,
        }
    )
    analysis = fw.discover_dependencies(df, max_key_size=1)
    assert analysis["exact_grain"]["graph"]["scope"]["evaluated_rows"] == 4
    assert analysis["graph_selection"]["excluded"] == [
        {
            "candidate_id": "key4",
            "columns": ["never"],
            "reason": "no_evaluated_support",
        }
    ]
    assert len(analysis["grain_views"]) == 3
    for view in analysis["grain_views"]:
        assert (
            view["grain"]["graph"]["scope"]["evaluated_rows"]
            == view["population"]["evaluated_rows"]
        )
    assert any(
        d["exact"]
        for d in analysis["dependencies"]
        if d["determinant"] == ["entity"] and d["target"] == "site"
    )
    assert all(c["graph_views"] for c in analysis["candidates"] if c["columns"] != ["never"])
    scoped = fw.discover_dependencies(df, scope=fw.Scope.from_positions(df, [0, 1]), max_key_size=1)
    assert scoped["exact_grain"]["graph"]["scope"]["input_rows"] == 4
    assert scoped["exact_grain"]["graph"]["scope"]["restriction_excluded_rows"] == 2


def test_typed_contexts_survive_saved_presentations():
    df = pd.DataFrame(
        {
            "site": pd.Series([1, 1, "1", "1"], dtype=object),
            "key": [1] * 4,
            "value": ["a", "a", "b", "b"],
        }
    )
    analysis = fw.discover_dependencies(df, by=["site"], max_key_size=1)
    saved = json.loads(json.dumps(analysis.to_dict(), allow_nan=False))
    for rendered in (
        fw.render_plaintext(saved, width=200, max_lines=1000),
        fw.render_html(saved),
        fw.render_html(saved, detail="topology"),
    ):
        assert "(integer)" in rendered and "(string)" in rendered
    rows = [
        r
        for r in fw.visualization_data(saved, detail="topology")["findings"]
        if r["structure"].get("context")
    ]
    assert {r["structure"]["context"]["site"]["type"] for r in rows} == {"integer", "string"}
    assert all("measurements" not in r and "examples" not in r for r in rows)


def test_candidate_roles_and_priority():
    df = pd.DataFrame(
        {
            "never": [None] * 4,
            "constant": [1] * 4,
            "id": range(4),
            "entity": [1, 1, 2, 2],
            "label": ["a", "a", "b", "b"],
        }
    )
    overview = fw.visualization_data(fw.explore(df))["overview"]
    roles = {tuple(c["columns"]): c["role"] for c in overview["grains"]}
    assert roles[("never",)] == "no evaluated support"
    assert roles[("constant",)] == "constant"
    assert roles[("id",)] == "unique identifier"
    assert roles[("entity",)] == "repeated grouping"
    assert overview["grains"][0]["role"] == "repeated grouping"
    assert len(overview["grains"]) == 5
