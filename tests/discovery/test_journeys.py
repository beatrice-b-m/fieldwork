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
    with pytest.raises(TypeError, match="unexpected keyword argument.*missing"):
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


def test_signature_to_complete_scope_and_saved_overview_inspection():
    df = pd.DataFrame({"a": [1, 2, 3, None], "b": [None, None, None, 4]}, index=[0] * 4)
    analysis = fw.missingness(df, example_limit=1)
    signature = next(s for s in analysis["signatures"] if s["present"] == ["a"])
    assert len(analysis.inspect(df, signature["finding_id"])) == 1
    saved = fw.InvestigationResult.from_dict(json.loads(json.dumps(analysis.to_dict())))
    scope = saved.select(df, signature["finding_id"], name="a only")
    assert scope.positions == (0, 1, 2)
    assert scope.parent == "input"
    assert saved.inspect(df, signature["finding_id"], all_matches=True).equals(df.iloc[:3])
    assert fw.missingness(df, scope=scope)["availability"][0]["populated"] == 3
    with pytest.raises(ValueError, match="differs"):
        saved.select(df.iloc[::-1], signature["finding_id"])
    overview = fw.explore(df)
    record = next(f for f in overview["findings"] if f["pattern"] == "availability_signature")
    assert len(overview.select(df, record["id"]).positions) == 3
    for rendered in (fw.render_plaintext(saved), fw.render_html(saved)):
        assert "Present: a; absent: b" in rendered
    assert "<pre>" not in fw.render_html(saved)


def test_context_and_entity_findings_select_full_source_rows():
    df = pd.DataFrame(
        {"site": ["A", "A", "B"], "entity": [1, 1, 2], "x": [1, None, None]}, index=[0] * 3
    )
    analysis = fw.missingness(df, features=["x"], by=["site"], entity="entity", example_limit=0)
    context = next(
        f
        for f in analysis["findings"]
        if f["pattern"] == "context_availability"
        and f["structure"]["context"]["site"]["value"] == "A"
    )
    assert analysis.select(df, context["id"], exceptions=True).positions == (1,)
    entity = next(
        f
        for f in analysis["findings"]
        if f["pattern"] == "entity_availability" and f["structure"]["presence_pattern"] == "some"
    )
    assert analysis.select(df, entity["id"]).positions == (0, 1)
    assert "some populated rows" in fw.render_plaintext(analysis)


def test_equal_entity_weights_and_aggregation():
    df = pd.DataFrame(
        {"entity": ["big"] * 100 + ["small", None], "a": [1] * 102, "b": [1] * 100 + [None, None]},
        index=[0] * 102,
    )
    args = {"features": ["a", "b"], "entity": "entity", "min_implication": 0, "min_similarity": 0}
    rows = fw.missingness(df.iloc[:101], **args)
    entities = fw.missingness(df, **args, unit="entities")

    def implication(analysis):
        return next(
            f
            for f in analysis["findings"]
            if f["pattern"] == "presence_implication" and f["features"][0]["column"] == "a"
        )

    assert implication(rows)["measurements"]["conditional_presence"] == pytest.approx(100 / 101)
    assert implication(entities)["measurements"]["conditional_presence"] == 0.5
    assert implication(entities)["measurements"]["presence_jaccard"] == 0.5
    assert implication(entities)["counting_unit"] == "entities"
    assert sorted(s["count"] for s in entities["signatures"]) == [1, 1]
    assert entities["analysis_unit"]["missing_key_excluded_rows"] == 1
    assert entities.select(df, implication(entities)["id"]).positions == tuple(range(100))
    assert entities.select(df, implication(entities)["id"], exceptions=True).positions == (100,)
    # A separate mixed entity demonstrates any/all presence without index-label selection.
    mixed = pd.DataFrame({"e": [1, 1, 2], "x": [1, None, None]})
    any_present = fw.missingness(mixed, entity="e", features=["x"], unit="entities")
    all_present = fw.missingness(
        mixed, entity="e", features=["x"], unit="entities", entity_presence="all"
    )
    assert any_present["availability"][0]["populated"] == 1
    assert all_present["availability"][0]["populated"] == 0
    with pytest.raises(ValueError, match="same analysis unit"):
        fw.compare(any_present, all_present)
    with pytest.raises(ValueError, match="requires entity"):
        fw.missingness(mixed, unit="entities")
    assert "2 entities" in fw.render_plaintext(any_present)


def test_recommendations_explain_evidence_and_diversify_feature_choices():
    df = pd.DataFrame(
        {
            "site": ["A"] * 4 + ["B"] * 4,
            "exam": [1, 1, 2, 2, 3, 3, 4, 4],
            "mode": ["x", "y"] * 4,
            "optional": [None] * 4 + [1, 2, 3, 4],
        }
    )
    paths = fw.suggest_paths(df, max_dimensions=2, n_paths=3)
    assert len(paths["paths"]) == 3
    assert len({frozenset(p["dimensions"]) for p in paths["paths"]}) == 3
    for path in paths["paths"]:
        assert "Observed prefix groups" in path["explanation"]
        assert {r["kind"] for r in path["reasons"]} >= {
            "branching",
            "nesting",
            "redundancy",
            "availability_separation",
        }
    nested = fw.suggest_paths(df, features=["site", "exam"], max_dimensions=2)
    assert nested.best.dimensions == ("site", "exam")
    assert len(nested["paths"]) == 1  # The reverse permutation is not a new investigation.
    assert "site → exam" in nested["paths"][0]["explanation"]
    target = fw.suggest_paths(df, objective="target", target="site", max_dimensions=1)
    assert any(r["kind"] == "target_separation" for r in target["paths"][0]["reasons"])
    constrained = fw.suggest_paths(
        df, start_with=["mode"], before=[("site", "exam")], max_dimensions=3
    )
    assert all(p["dimensions"] == ["mode", "site", "exam"] for p in constrained["paths"])
    assert "Observed prefix groups" in fw.render_plaintext(paths)


def test_connected_feature_relationships_preserve_evidence_types():
    df = pd.DataFrame(
        {
            "a_1": [1, 1, 2, 2, 3, None],
            "a_2": [10, 10, 20, 20, None, None],
            "mirror": [1, 1, 2, 2, 3, None],
        }
    )
    overview = fw.explore(df)
    network = overview["feature_network"]
    assert {r["kind"] for r in network["relationships"]} >= {
        "identical_availability",
        "similar_availability",
        "indexed_name",
        "equivalent_value_partitions",
        "exact_dependency",
    }
    connections = overview.relationships("a_1", kinds=["indexed_name"])
    assert len(connections) == 1
    finding_id = connections.iloc[0]["evidence.overview_finding_id"]
    assert overview.inspect(df, finding_id).equals(df.iloc[:5])
    assert overview.select(df, finding_id).positions == tuple(range(6))
    for relationship in network["relationships"]:
        evidence = relationship["evidence"]
        assert any(
            f["id"] == evidence["finding_id"]
            for f in overview["sections"][evidence["section"]]["findings"]
        )
    saved = json.loads(json.dumps(overview.to_dict(), allow_nan=False))
    assert "Browse feature connections" in fw.render_html(saved)
    assert "inspect evidence</a>" in fw.render_html(saved)
    topology = fw.visualization_data(saved, detail="topology")["feature_network"]
    serialized = json.dumps(topology)
    for key in ("population_ref", "overview_finding_id", "denominator", "measurements"):
        assert key not in serialized
    assert "indexed_name" in serialized and "exact_dependency" in serialized


def test_overview_entity_context_configuration_and_selection():
    df = pd.DataFrame({"site": ["A", "A", "B"], "entity": [1, 1, 2], "x": [1, None, None]})
    overview = fw.explore(df, discovery={"by": ["site"], "entity": "entity", "unit": "entities"})
    assert overview["sections"]["missingness"]["analysis_unit"]["denominator"] == 2
    assert overview["sections"]["dependencies"]["coverage"]["contexts_evaluated"] == 2
    signature = next(
        f
        for f in overview["findings"]
        if f["pattern"] == "availability_signature" and "x" in f["structure"]["present"]
    )
    assert overview.select(df, signature["id"]).positions == (0, 1)


def test_path_empty_exception_selection_and_entity_presentation_units():
    df = pd.DataFrame({"e": [1, 1, 2], "x": [1, 2, None]})
    paths = fw.suggest_paths(df)
    path = next(f for f in paths["findings"] if f["pattern"] == "census_path")
    assert paths.select(df, path["id"], exceptions=True).positions == ()
    assert paths.inspect(df, path["id"], exceptions=True, all_matches=True).empty
    overview = fw.explore(df, discovery={"entity": "e", "unit": "entities"})
    text = fw.render_plaintext(overview)
    assert "(1 entities)" in text
    assert "(1 rows)" not in text
    assert "2 entities" in fw.render_svg(overview)


def test_whole_context_and_entity_summaries_are_selectable_after_save():
    df = pd.DataFrame(
        {"site": ["A", "A", "B"], "e": [1, 1, None], "x": [1, None, 2]}, index=[0] * 3
    )
    analysis = fw.missingness(df, by=["site"], entity="e", features=["x"], example_limit=0)
    saved = fw.InvestigationResult.from_dict(
        json.loads(json.dumps(analysis.to_dict(), allow_nan=False))
    )
    context_id = saved["contexts"][0]["finding_id"]
    assert saved.select(df, context_id).positions == (0, 1)
    entity = next(f for f in saved["findings"] if f["pattern"] == "entity_summary")
    assert saved.select(df, entity["id"]).positions == (0, 1)
    assert "x: 1/2 populated rows" in fw.render_plaintext(saved)


@pytest.mark.parametrize("aggregation", ["any", "all"])
def test_topology_retains_entity_relationship_meaning(aggregation):
    df = pd.DataFrame(
        {"e": [1, 1, 2, 2, 3], "a": [1, None, 1, None, None], "b": [None, 1, None, 1, None]}
    )
    config = {
        "features": ["a", "b", "e"],
        "entity": "e",
        "unit": "entities",
        "entity_presence": aggregation,
    }
    rows = fw.visualization_data(fw.missingness(df, features=["a", "b"]), detail="topology")
    assert any(f["pattern"] == "mutually_exclusive" for f in rows["findings"])
    expected = {
        "counting_unit": "entities",
        "entity_keys": ["e"],
        "presence_aggregation": aggregation,
    }
    for result in (fw.missingness(df, **config), fw.explore(df, discovery=config)):
        saved = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        topology = fw.visualization_data(saved, detail="topology")
        family = next(f for f in topology["findings"] if f["pattern"] == "availability_family")
        assert family["analysis_unit"] == expected
        assert family["counting_unit"] == "entities"
        assert topology["analysis_unit"] == expected
        if result.kind == "overview":
            edge = next(
                e
                for e in topology["feature_network"]["relationships"]
                if e["kind"] == "identical_availability"
            )
            assert edge["analysis_unit"] == expected
            row_evidence = [f for f in topology["findings"] if f["pattern"] == "exact_dependency"]
            assert row_evidence
            assert all(f["analysis_unit"]["counting_unit"] == "rows" for f in row_evidence)
            for edge in saved["feature_network"]["relationships"]:
                edge.pop("analysis_unit")
            # Older exports can recover connection context from their linked findings.
            assert fw.visualization_data(saved, detail="topology") == topology
        for render in (fw.render_plaintext, fw.render_svg, fw.render_html):
            text = render(saved, detail="topology")
            assert "entities; keys=e; presence=" + aggregation in text
        serialized = json.dumps(topology)
        for key in (
            "denominator",
            "positions",
            "dataset_id",
            "measurements",
            "source_rows",
            "evaluated_rows",
        ):
            assert key not in serialized


def test_dependency_support_survives_overview_network_and_graph_handoffs():
    df = pd.DataFrame({"X": [1, 1, 2, 2], "Y": ["a", None, "b", None], "Z": range(4)})
    overview = fw.InvestigationResult.from_dict(
        json.loads(json.dumps(fw.explore(df).to_dict(compact=True), allow_nan=False))
    )
    dependencies = fw.InvestigationResult.from_dict(overview["sections"]["dependencies"])
    records = {(tuple(d["determinant"]), d["target"]): d for d in dependencies["dependencies"]}
    assert records[(("X",), "Y")]["exact"]
    assert records[(("Y",), "Z")]["exact"]
    assert not records[(("X",), "Z")]["exact"]
    assert records[(("X",), "Y")]["repeated_rows"] == 0
    broad = dependencies["grain_views"][0]
    assert broad["population"]["evaluated_rows"] == 4
    assert broad["candidate_ids"] == ["key0", "key2"]
    assignment = next(
        a for a in broad["grain"]["graph"]["assignments"] if a["target"]["value"] == "Y"
    )
    assert assignment["nodes"] == []
    assert assignment["reason"] == "different_target_population"
    narrow = dependencies["grain_views"][1]
    assert narrow["population"]["positions"] == [0, 2]
    assert len(narrow["grain"]["graph"]["nodes"]) == 1
    # Within Y's observed rows X is unique, so the rule is trivial: it stays a
    # finding but does not connect features in the network.
    assert not any(
        e.get("determinant") == ["X"] and e.get("target") == "Y"
        for e in overview["feature_network"]["relationships"]
    )
    finding_id = next(
        f["id"]
        for f in overview["findings"]
        if f["pattern"] == "exact_dependency"
        and f["measurements"]["determinant"] == ["X"]
        and f["measurements"]["target"] == "Y"
    )
    assert overview.select(df, finding_id).positions == (0, 2)
    finding = next(f for f in overview["findings"] if f["id"] == finding_id)
    assert finding["measurements"]["target_coverage"] == 0.5
    assert "0 with repeated support" in fw.render_plaintext(overview)
    assert "target observed on 2/4" in fw.render_html(overview, section="dependencies")
    assert "repeat-only consistency not assessable" in fw.render_html(overview)
    assert "target_coverage" not in json.dumps(fw.visualization_data(overview, detail="topology"))


def test_singleton_inflation_and_incompatible_views_remain_visible():
    frame = pd.DataFrame({"X": [*range(99), 98], "Y": ["a"] * 99 + ["b"]})
    overview = fw.explore(frame)
    dep = next(
        d for d in overview["sections"]["dependencies"]["dependencies"] if d["determinant"] == ["X"]
    )
    assert dep["modal_accuracy"] == 0.99
    assert dep["repeat_modal_accuracy"] == 0.5
    assert "repeat-only consistency 0.5" in fw.render_html(overview)
    assert "repeat coverage 0.02" in fw.render_html(overview, section="dependencies")
    overlapping = fw.discover_dependencies(
        pd.DataFrame({"X": [1, 1, 2, 2], "left": [1, 1, None, None], "right": [None, None, 2, 2]}),
        max_key_size=1,
    )
    for view in overlapping["grain_views"]:
        assert not {"key1", "key2"}.issubset(view["candidate_ids"])
        assert (
            view["population"]["evaluated_rows"]
            == view["grain"]["graph"]["scope"]["evaluated_rows"]
        )
