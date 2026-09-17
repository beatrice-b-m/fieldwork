import json
from xml.etree import ElementTree

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
        max_signatures=1,
        example_limit=1,
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
    loaded = fw.InvestigationResult.from_dict(json.loads(json.dumps(r.to_dict(), allow_nan=False)))
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
    bounded = fw.discover_dependencies(df, max_candidates=1)
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
    r = fw.suggest_paths(frame, max_candidates=1)
    assert r["coverage"]["paths_evaluated"] == 1


@pytest.mark.parametrize(
    "operation",
    [fw.missingness, fw.discover_dependencies, fw.suggest_paths, fw.value_patterns, fw.explore],
)
def test_strict_saved_rendering_and_topology(frame, operation):
    r = operation(frame)
    saved = json.loads(json.dumps(r.to_dict(), allow_nan=False))
    assert fw.render_plaintext(saved)
    ElementTree.fromstring(fw.render_svg(saved))
    assert "<html" in fw.render_html(saved)
    topology = fw.visualization_data(saved, detail="topology")
    serialized = json.dumps(topology)
    for key in ["measurements", "positions", "dataset_id", "populated_fraction", "evaluated_rows"]:
        assert key not in serialized
    assert "measurements" not in fw.render_html(saved, detail="topology")
    assert "Population" not in fw.render_plaintext(saved, detail="topology")
    assert not r.to_frame().empty


def test_escaping_and_export_limits():
    df = pd.DataFrame({"<script>alert(1)</script>\x1b": [1, None]})
    r = fw.missingness(df)
    assert "<script>alert" not in fw.render_html(r)
    assert "\x1b" not in fw.render_plaintext(r)
    ElementTree.fromstring(fw.render_svg(r))
    assert "more" in fw.render_plaintext(r, max_lines=1)


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


def test_scope_and_convention_propagate_to_overview(frame):
    scope = fw.Scope.from_positions(frame, [0, 2, 5])
    overview = fw.explore(
        frame, discovery={"scope": scope, "missing": {"a": [5]}, "features": ["a", "b"]}
    )
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
    r = fw.missingness(frame, features=["a"], scope=scope, missing={"a": [5]}, example_limit=0)
    restored = fw.InvestigationResult.from_dict(json.loads(json.dumps(r.to_dict())))
    replay = restored.recompute(frame, example_limit=len(frame))
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
    assert r["contexts"][0]["values"]["site"] == {"type": "missing"}
    assert r["contexts"][0]["rows"] == 2
