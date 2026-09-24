import json

import numpy as np
import pandas as pd
import pytest

import fieldwork as fw


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "site": ["N", "N", "N", "S", "S", "S"],
            "exam": [1, 1, 2, 3, 3, 3],
            "a": [1, 2, None, 3, 4, 5],
            "b": [1, 2, None, 3, 4, None],
            "c": [None, None, 1, None, None, None],
            "always": [1] * 6,
            "never": [None] * 6,
        },
        index=[0] * 6,
    )


def test_missingness_counts_and_duplicate_index_inspection(frame):
    r = fw.missingness(frame, by=["site"], entity="exam", min_implication=0.8)
    edge = next(
        f
        for f in r["findings"]
        if f["pattern"] == "presence_implication"
        and [c["column"] for c in f["features"]] == ["a", "b"]
    )
    assert edge["measurements"]["conditional_presence"] == 0.8
    assert edge["measurements"]["presence_jaccard"] == 0.8
    assert r.inspect(frame, edge["id"], exceptions=True).equals(frame.iloc[[5]])
    entities = {x["feature"]: x for x in r["entities"]}
    assert entities["a"] == {
        "feature": "a",
        "denominator": 3,
        "any": 2,
        "all": 2,
        "one": 0,
        "some": 0,
        "none": 1,
    }
    assert entities["b"]["some"] == 1
    assert r["contexts"][0]["rows"] == 3
    with pytest.raises(ValueError, match="differs"):
        r.inspect(frame.iloc[::-1], edge["id"])


def test_sentinels_scope_and_signatures(frame):
    original = frame.copy(deep=True)
    scope = fw.Scope.from_positions(frame, [0, 2, 5], name="subset")
    r = fw.missingness(
        frame,
        features=["a", "b"],
        missing={"a": [5]},
        scope=scope,
        limits={"max_signatures": 1, "example_limit": 1},
    )
    assert r["availability"][0]["populated"] == 1
    assert r["coverage"]["signature_omitted_rows"] == 1
    assert r["scope"]["restriction_excluded_rows"] == 3
    pd.testing.assert_frame_equal(frame, original)
    assert scope.refine(frame, [2]).parent == "subset"
    with pytest.raises(ValueError):
        scope.refine(frame, [1])
    with pytest.raises(ValueError):
        fw.Scope.from_positions(frame, [-1])
    loaded = fw.Result.from_dict(json.loads(json.dumps(r.to_dict(), allow_nan=False)))
    assert loaded.inspect(frame, 0).equals(frame.iloc[[0]])


def test_coabsence_does_not_manufacture_similarity():
    df = pd.DataFrame({"a": [1] + [None] * 99, "b": [None, 1] + [None] * 98, "zero": [None] * 100})
    r = fw.missingness(df)
    assert not any(f["pattern"] == "similar_availability" for f in r["findings"])
    assert any(f["pattern"] == "mutually_exclusive" for f in r["findings"])
    assert any(
        f["pattern"] == "availability" and f["measurements"]["populated"] == 0
        for f in r["findings"]
    )


def test_approximate_conditional_and_composite_dependencies():
    df = pd.DataFrame(
        {
            "key": [1, 1, 1, 2, 2],
            "value": ["a", "a", "b", "c", "c"],
            "site": ["N", "N", "S", "N", "N"],
        },
        index=[0] * 5,
    )
    r = fw.discover_dependencies(df, by=["site"], min_accuracy=0.8)
    dep = next(
        f
        for f in r["findings"]
        if f["pattern"] == "approximate_dependency"
        and f["measurements"]["determinant"] == ["key"]
        and f["measurements"]["target"] == "value"
    )
    assert dep["measurements"]["modal_accuracy"] == 0.8
    assert dep["measurements"]["violating_groups"] == 1
    assert dep["measurements"]["repeated_groups"] == 2
    assert r.inspect(df, dep["id"], exceptions=True).equals(df.iloc[[2]])
    assert any(
        d["exact"]
        for d in r["dependencies"]
        if d["determinant"] == ["key"] and d["target"] == "value" and d["context"]
    )
    assert any(
        d["exact"]
        for d in r["dependencies"]
        if d["determinant"] == ["key", "site"] and d["target"] == "value"
    )
    bounded = fw.discover_dependencies(df, limits={"max_candidates": 1})
    assert bounded["coverage"]["candidates_evaluated"] == 1
    assert bounded["coverage"]["candidate_space"] == 6


def test_dependency_population_and_empty():
    df = pd.DataFrame({"key": [1, 1, None], "value": ["a", "missing", "z"]})
    r = fw.discover_dependencies(df, missing={"value": ["missing"]})
    d = r["dependencies"][0]
    assert d["evaluated_rows"] == 1 and d["missing_excluded_rows"] == 2
    assert d["exact"]
    assert fw.discover_dependencies(df.iloc[:0])["dependencies"][0]["exact"] is None
    assert fw.missingness(df.iloc[:0])["availability"][0]["populated_fraction"] is None


def test_nesting_and_constraints(frame):
    paths = fw.suggest_paths(frame, features=["exam", "site"], max_dimensions=2)
    assert paths.best.dimensions == ("site", "exam")
    assert (
        paths.to_dict()
        == fw.suggest_paths(frame, features=["exam", "site"], max_dimensions=2).to_dict()
    )
    constrained = fw.suggest_paths(
        frame,
        features=["site", "exam", "a"],
        start_with=["site"],
        before=[("exam", "a")],
        max_dimensions=3,
    )
    assert constrained.best.dimensions == ("site", "exam", "a")
    with pytest.raises(ValueError, match="cycle"):
        fw.suggest_paths(frame, before=[("site", "exam"), ("exam", "site")])
    with pytest.raises(ValueError):
        fw.suggest_paths(frame, before=[("site", "exam")], exclude=["site"])
    with pytest.raises(ValueError):
        fw.suggest_paths(frame, objective="target")
    r = fw.suggest_paths(frame, limits={"max_candidates": 1})
    assert r["coverage"]["paths_evaluated"] == 1


@pytest.mark.parametrize(
    "operation",
    [fw.missingness, fw.discover_dependencies, fw.suggest_paths, fw.value_patterns, fw.explore],
)
def test_saved_results_restore_and_tabulate(frame, operation):
    r = operation(frame)
    saved = json.loads(json.dumps(r.to_dict(), allow_nan=False))
    restored = fw.Result.from_dict(saved)
    assert restored.to_dict() == saved
    assert not r.to_frame().empty
    assert restored.to_frame().equals(r.to_frame())


def test_value_patterns_recipe_compare(tmp_path):
    df = pd.DataFrame(
        {
            "code_1": ["AB-01", "AB-02", None],
            "code_2": ["X1", "X2", None],
            "n": [1, 2, 3],
            "m": [3, 4, 5],
        }
    )
    r = fw.value_patterns(df, by=["code_1"])
    assert r["families"][0]["evidence"] == ["indexed_name", "identical_availability"]
    assert any(
        f["pattern"] == "numeric_offset" and f["measurements"]["value"] == 2 for f in r["findings"]
    )
    recipe = fw.Recipe("missingness", {"features": ["n"]}, notes="delivery check")
    path = tmp_path / "recipe.json"
    recipe.save(path)
    first = fw.Recipe.load(path).run(df)
    df.loc[0, "n"] = np.nan
    second = recipe.run(df)
    change = fw.compare(first, second)["changes"][0]
    assert change["populated_fraction_delta"] == pytest.approx(-1 / 3)


def test_no_optional_runtime_imports():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import fieldwork,sys; assert not {'pydicom','pulp','matplotlib','resvg_py'} & set(sys.modules)",
        ],
        check=True,
    )


def test_target_and_availability_objectives():
    df = pd.DataFrame(
        {
            "irrelevant": [0, 1, 0, 1] * 3,
            "signal": [0] * 6 + [1] * 6,
            "target": ["a"] * 6 + ["b"] * 6,
            "optional": [None] * 6 + [1] * 6,
        }
    )
    target = fw.suggest_paths(
        df,
        objective="target",
        target="target",
        features=["irrelevant", "signal", "target"],
        max_dimensions=1,
    )
    assert target.best.dimensions == ("signal",)
    availability = fw.suggest_paths(
        df,
        objective="availability",
        features=["irrelevant", "signal", "optional"],
        max_dimensions=1,
    )
    assert availability.best.dimensions in [("signal",), ("optional",)]
    context = fw.suggest_paths(df, objective="context", start_with=["irrelevant"])
    assert context.best.dimensions[0] == "irrelevant"


def test_numeric_sentinel_equality_and_multiindex():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, None]}, index=pd.MultiIndex.from_tuples([("a", 0), ("a", 0), ("b", 1)])
    )
    r = fw.missingness(df, missing={"a": [1]})
    assert r["availability"][0]["populated"] == 1
    assert r.inspect(df, 0).equals(df.iloc[[1]])


@pytest.mark.parametrize(
    "level", [[1.5, 2.5, 1.5, 2.5], pd.date_range("2024-01-01", periods=4)], ids=["float", "date"]
)
def test_groupby_style_multiindex_supports_discovery(level):
    index = pd.MultiIndex.from_arrays([["x", "x", "y", "y"], level])
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": ["p", "q", "p", "q"]}, index=index)
    overview = fw.explore(df)
    finding = next(f for f in overview.findings if f["pattern"] == "availability")
    assert overview.inspect(df, finding["id"], all_matches=True).equals(df.iloc[[0, 1, 3]])
    assert fw.discover_dependencies(df)["candidates"]


def test_scope_and_convention_propagate_to_overview(frame):
    scope = fw.Scope.from_positions(frame, [0, 2, 5])
    overview = fw.explore(frame, scope=scope, missing={"a": [5]}, features=["a", "b"])
    for name in ("missingness", "dependencies", "paths"):
        assert overview["sections"][name]["scope"]["evaluated_rows"] == 3
    assert overview["sections"]["missingness"]["availability"][0]["populated"] == 1


def test_availability_matches_boolean_oracle():
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @given(st.lists(st.tuples(st.booleans(), st.booleans()), max_size=40))
    @settings(max_examples=30)
    def check(rows):
        df = pd.DataFrame(
            {"a": [1 if a else None for a, b in rows], "b": [1 if b else None for a, b in rows]}
        )
        r = fw.missingness(df, min_implication=0)
        assert r["availability"][0]["populated"] == sum(a for a, b in rows)
        for f in r["findings"]:
            if f["pattern"] == "presence_implication" and f["features"][0]["column"] == "a":
                m = f["measurements"]
                assert m["conditional_presence"] == sum(a and b for a, b in rows) / sum(
                    a for a, b in rows
                )
                assert f["exceptions"]["total"] == sum(a and not b for a, b in rows)

    check()


def test_saved_recomputation_restores_scope_and_sentinels(frame):
    scope = fw.Scope.from_positions(frame, [0, 2, 5])
    r = fw.missingness(
        frame, features=["a"], scope=scope, missing={"a": [5]}, limits={"example_limit": 0}
    )
    restored = fw.Result.from_dict(json.loads(json.dumps(r.to_dict())))
    replay = restored.recompute(frame, limits={"example_limit": len(frame)})
    assert replay["availability"] == r["availability"]
    assert replay.inspect(frame, 0, exceptions=True).equals(frame.iloc[[2, 5]])
    with pytest.raises(ValueError):
        restored.recompute(frame.iloc[::-1])


def test_scope_constructor_and_missing_contexts():
    with pytest.raises(ValueError):
        fw.Scope("identity", (-1,))
    df = pd.DataFrame({"site": [None, "unknown", "A"], "value": [1, 2, 3]})
    r = fw.missingness(df, by=["site"], missing={"site": ["unknown"]})
    assert r["coverage"]["contexts_total"] == 2
    assert r["contexts"][0]["values"]["site"] is None
    assert r["contexts"][0]["rows"] == 2


def test_overview_displays_all_orientation_sections(frame):
    overview = fw.explore(frame)
    text = str(overview)
    for section in [
        "Availability families",
        "Major availability signatures",
        "Candidate grains",
        "Suggested census paths",
    ]:
        assert section in text


@pytest.mark.parametrize("unit", ["rows", "entities"])
def test_saved_comparison_preserves_both_scopes_and_units(unit):
    df = pd.DataFrame({"e": [1, 1, 2, 3, 3, None], "a": [1, None, 1, None, None, 1]})
    parent = fw.Scope.from_positions(df, range(len(df)), name="delivery")
    before_scope = parent.refine(df, [0, 1, 2], name="baseline cohort")
    after_scope = parent.refine(df, [3, 4, 5], name="followup cohort")
    config = {"features": ["a"], "unit": unit, "entity": "e"}
    before = fw.missingness(df, scope=before_scope, **config)
    after = fw.missingness(df, scope=after_scope, **config)
    saved = json.loads(json.dumps(fw.compare(before, after).to_dict(), allow_nan=False))
    restored = fw.Result.from_dict(saved)
    assert saved["before_source"] == saved["after_source"]
    for side, analysis in (("before", before), ("after", after)):
        assert restored[f"{side}_scope"] == analysis["scope"]
        assert restored[f"{side}_analysis_unit"] == analysis["analysis_unit"]
    assert saved["scope"] == saved["after_scope"]
    assert saved["analysis_unit"] == saved["after_analysis_unit"]
    if unit == "entities":
        assert saved["before_analysis_unit"]["denominator"] == 2
        assert saved["after_analysis_unit"]["denominator"] == 1
        assert saved["after_analysis_unit"]["missing_key_excluded_rows"] == 1
    full = fw.visualization_data(restored)
    assert full["before_scope"]["selection_positions"] == [0, 1, 2]
    assert full["after_scope"]["selection_positions"] == [3, 4, 5]
    for detail in ("full", "topology"):
        # Both populations stay named in every medium.
        for render in (fw.render_plaintext, fw.render_svg, fw.render_html):
            rendered = render(saved, detail=detail)
            assert "baseline cohort" in rendered and "followup cohort" in rendered
        projected = fw.visualization_data(saved, detail=detail)
        assert projected["before_scope"]["parent"] == "delivery"


def test_overview_recipe_reapplies_configuration_to_selected_population(tmp_path):
    df = pd.DataFrame({"e": [1, 1, 2, 3], "site": ["A", "B", "B", "B"], "a": [1, -999, 2, 3]})
    scope = fw.Scope.from_positions(df, [0, 1], name="selected cohort")
    recipe = fw.Recipe(
        "explore",
        {
            "features": ["site", "a"],
            "missing": {"a": [-999]},
            "table_id": "configured",
            "entity": "e",
            "unit": "entities",
            "entity_presence": "all",
            "by": ["site"],
            "options": {
                "paths": {
                    "limits": {"max_candidates": 1},
                    "max_dimensions": 2,
                    "start_with": ["site"],
                }
            },
        },
    )
    path = tmp_path / "overview.json"
    recipe.save(path)
    configured = json.loads(path.read_text())
    loaded = fw.Recipe.load(path)
    baseline = loaded.run(df)
    overview = loaded.run(df, scope=scope, table_id="selected delivery")
    assert loaded.to_dict() == configured
    unit = overview["sections"]["missingness"]["analysis_unit"]
    assert (unit["denominator"], unit["presence_aggregation"]) == (1, "all")
    assert baseline["scope"]["evaluated_rows"] == 4
    for name in ("missingness", "dependencies", "paths", "value_patterns"):
        section = overview["sections"][name]
        assert section["scope"]["selection_positions"] == [0, 1]
        assert section["scope"]["name"] == "selected cohort"
        assert section["source"]["table_id"] == "selected delivery"
        assert section["parameters"]["features"] == ["site", "a"]
        assert section["missing_convention"] == baseline["sections"][name]["missing_convention"]
    missingness = overview["sections"]["missingness"]
    assert missingness["parameters"]["by"] == ["site"]
    assert missingness["availability"][1]["populated"] == 0
    paths = overview["sections"]["paths"]
    assert paths["parameters"]["limits"]["max_candidates"] == 1
    assert paths["parameters"]["start_with"] == ["site"]
    assert paths["coverage"]["paths_evaluated"] == 1
    assert paths["paths"][0]["preview"]["scope"]["selection_positions"] == [0, 1]
    overridden = loaded.run(df, scope=scope, missing={}, features=["site"])
    assert overridden["sections"]["missingness"]["parameters"]["features"] == ["site"]
    assert overridden["missing_convention"]["sentinels"].get("a", []) == []
    assert loaded.to_dict() == configured
    assert fw.Recipe("explore").run(df, scope=scope)["scope"]["selection_positions"] == [0, 1]
    with pytest.raises(TypeError, match="unexpected keyword"):
        loaded.run(df, max_candidates=2)


def test_unsupported_cells_skip_automatic_columns_but_reject_explicit_ones():
    from decimal import Decimal

    df = pd.DataFrame(
        {
            "site": ["A", "A", "B"],
            "tags": [["x"], ["y"], None],
            "meta": [{"k": 1}, None, {"k": 2}],
            "amount": [Decimal("1.5"), Decimal(2), None],
        }
    )
    overview = fw.explore(df)
    assert [r["feature"] for r in overview["skipped_features"]] == ["tags", "meta", "amount"]
    assert {r["value_type"] for r in overview["skipped_features"]} == {"list", "dict", "Decimal"}
    analyzed = {c["column"] for f in overview.findings for c in f["features"]}
    assert analyzed == {"site"}
    assert any(
        "tags" in line and "list" in line for line in fw.render_plaintext(overview).splitlines()
    )
    for analysis in (fw.missingness, fw.discover_dependencies, fw.value_patterns):
        assert analysis(df)["parameters"]["features"] == ["site"]
        with pytest.raises(TypeError, match="'tags'"):
            analysis(df, features=["site", "tags"])


def test_availability_omits_vacuous_findings_but_keeps_measurements():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "complete": ["a", "b", "c", "d"],
            "left": [1, 2, None, None],
            "twin": [5, 6, None, None],
            "mostly": [1, 2, 3, None],
        }
    )
    result = fw.missingness(df)
    assert [r["feature"] for r in result["availability"]] == list(df.columns)
    found = {(f["pattern"], tuple(c["column"] for c in f["features"])) for f in result["findings"]}
    availability = {cols[0] for pattern, cols in found if pattern == "availability"}
    assert availability == {"left", "twin", "mostly"}
    implications = {cols for pattern, cols in found if pattern == "presence_implication"}
    assert implications == {("left", "mostly"), ("twin", "mostly")}
    assert [f["features"] for f in result["families"]] == [["left", "twin"]]


def test_context_and_entity_availability_keep_their_state_in_topology():
    df = pd.DataFrame(
        {
            "acc": [1, 1, 2, 2, 3, 4, 5],
            "b": [1, None, 2, None, None, None, 3],
            "ctx": ["M", "M", "U", "U", "M", "U", "V"],
        }
    )
    result = fw.missingness(df, features=["b"], by=["ctx"], entity="acc")
    topology = fw.visualization_data(result, detail="topology")["findings"]
    contexts = {
        f["structure"]["context"]["ctx"]: f["structure"]["presence"]
        for f in topology
        if f["pattern"] == "context_availability"
    }
    assert contexts == {"M": "some", "U": "some", "V": "all"}
    empty = fw.missingness(df.assign(b=None), features=["b"], by=["ctx"])
    assert {
        f["structure"]["presence"]
        for f in empty["findings"]
        if f["pattern"] == "context_availability"
    } == {"none"}
    # Entities 1 and 2 have one of two rows, 3 and 4 none, 5 its only row: no
    # entity other than 5 has all rows populated, and every pattern matches one.
    summary = result["entities"][0]
    assert {k: summary[k] for k in ("any", "all", "one", "some", "none")} == {
        "any": 3,
        "all": 1,
        "one": 3,
        "some": 2,
        "none": 2,
    }
    unmatched = fw.missingness(df.iloc[:6], features=["b"], entity="acc")
    assert unmatched["entities"][0]["all"] == 0
    for analysis in (result, unmatched):
        counts = analysis["entities"][0]
        patterns = {
            f["structure"]["presence_pattern"]
            for f in fw.visualization_data(analysis, detail="topology")["findings"]
            if f["pattern"] == "entity_availability"
        }
        assert patterns == {p for p in ("any", "all", "one", "some", "none") if counts[p]}


def test_topology_distinguishes_exact_from_approximate_availability_relations():
    df = pd.DataFrame({"a": [1] * 10 + [None], "c": [1] * 9 + [None, None]})
    topology = fw.visualization_data(fw.missingness(df), detail="topology")["findings"]
    strengths = {
        (f["pattern"], tuple(c["column"] for c in f["features"])): f["structure"]["strength"]
        for f in topology
        if f["pattern"] in {"presence_implication", "similar_availability"}
    }
    # c implies a on every unit; a implies c on 9 of 10.
    assert strengths == {
        ("presence_implication", ("a", "c")): "approximate",
        ("presence_implication", ("c", "a")): "exact",
        ("similar_availability", ("a", "c")): "approximate",
    }
    # Plain-text topology shows statements only, so they carry the strength too:
    # "a implies c" reads differently once it holds exactly.
    exact = pd.DataFrame({"a": [1] * 9 + [None, None], "c": [1] * 10 + [None]})
    texts = [fw.render_plaintext(fw.missingness(frame), detail="topology") for frame in (df, exact)]
    assert texts[0] != texts[1]


def test_overview_ranks_leads_and_keeps_trivial_rules_out_of_network():
    rows = 40
    df = pd.DataFrame(
        {
            "row_id": range(rows),
            "site": ["A", "B"] * (rows // 2),
            "region": ["north", "south"] * (rows // 2),
            "constant": ["same"] * rows,
        }
    )
    df.loc[2, "region"] = "south"  # one exception to site -> region
    overview = fw.explore(df)
    top = overview.findings[0]
    assert top["id"] == "f0" and top["pattern"] == "approximate_dependency"
    assert top["measurements"]["determinant"] == ["site"]
    assert top["lead"]["reason"] == "near-rule with exceptions"
    scores = [f["lead"]["score"] for f in overview.findings]
    assert scores == sorted(scores, reverse=True)
    trivial = [
        f
        for f in overview.findings
        if f["lead"]["reason"] in {"target is constant", "determinant is unique here"}
    ]
    assert trivial
    network = overview["feature_network"]["relationships"]
    linked = {e["evidence"]["overview_finding_id"] for e in network}
    assert linked.isdisjoint(f["id"] for f in trivial)
    assert overview.inspect(df, "f0", exceptions=True, all_matches=True).index.tolist() == [2]


def test_similarity_with_an_always_present_feature_is_not_reported():
    df = pd.DataFrame({"complete": range(10), "nearly": [1] * 9 + [None]})
    result = fw.missingness(df, min_similarity=0.8)
    assert not [f for f in result["findings"] if f["pattern"] == "similar_availability"]


def test_overview_summary_leads_with_ranked_findings_and_merged_grains():
    df = pd.DataFrame(
        {
            "site": ["A", "A", "B", "B", "C", "C"],
            "site_name": ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"],
            "note": ["x", None, None, None, None, None],
        }
    )
    overview = fw.explore(df)
    data = fw.visualization_data(overview)["overview"]
    first = data["grains"][0]
    assert {first["columns"][0], *first["equivalent"]} == {"site", "site_name"}
    assert {s["absent"] == ["note"] for s in data["signatures"]} == {True, False}
    # The text summary leads with the top-ranked finding, by ID and statement.
    first = next(line for line in str(overview).splitlines() if "[f" in line)
    assert "[f0]" in first and overview.findings[0]["statement"] in first


def test_saved_timestamp_sentinel_reapplies_after_json_round_trip():
    df = pd.DataFrame({"when": pd.to_datetime(["2020-01-01", "1900-01-01", "2020-01-02"])})
    result = fw.missingness(df, missing={"when": [pd.Timestamp("1900-01-01")]})
    assert result["availability"][0]["missing"] == 1
    saved = fw.Result.from_dict(json.loads(json.dumps(result.to_dict())))
    assert saved.recompute(df)["availability"] == result["availability"]
