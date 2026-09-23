"""Target populations, repeated support, and candidate ranking."""

import json

import pandas as pd

import fieldwork as fw


def sparse_frame():
    return pd.DataFrame({"X": [1, 1, 2, 2], "Y": ["a", None, "b", None]}, index=[0] * 4)


EXCEPTION_FIELDS = {"exception_groups", "omitted_exception_groups"}


def measured(record):
    """A dependency test as its finding measures it: exception groups stay in the test."""
    return {k: v for k, v in record.items() if k not in EXCEPTION_FIELDS}


def dependency(result, key=("X",), target="Y", context=None):
    return next(
        d
        for d in result["dependencies"]
        if d["determinant"] == list(key) and d["target"] == target and d["context"] == context
    )


def test_sparse_target_characterization():
    result = fw.discover_dependencies(sparse_frame(), max_key_size=1)
    d = dependency(result)
    assert d["exact"] is True
    assert d["evaluated_rows"] == 2
    assert d["repeated_groups"] == 0
    assert result["candidates"][0]["repeated_rows"] == 4
    assert result["candidates"][0]["determines"] == ["Y"]
    finding = next(f for f in result["findings"] if f["measurements"]["determinant"] == ["X"])
    assert result.select(sparse_frame(), finding["id"]).positions == (0, 2)


def test_sparse_support_and_finding_measurements():
    result = fw.discover_dependencies(sparse_frame(), max_key_size=1)
    d = dependency(result)
    expected = {
        "determinant_evaluated_rows": 4,
        "target_observed_rows": 2,
        "target_coverage": 0.5,
        "target_missing_excluded_rows": 2,
        "repeated_rows": 0,
        "repeat_coverage": 0,
        "repeat_modal_accuracy": None,
    }
    assert {k: d[k] for k in expected} == expected
    assert result["candidates"][0]["determines_with_repeated_support"] == []
    assert result["candidates"][0]["global_targets_tested"] == 1
    assert result["candidates"][0]["global_targets_possible"] == 1
    f = next(f for f in result["findings"] if f["measurements"]["determinant"] == ["X"])
    assert all(f["measurements"][k] == v for k, v in expected.items())
    assert result.to_frame("dependencies").iloc[0]["target_coverage"] == 0.5


def test_singletons_do_not_inflate_repeat_consistency():
    df = pd.DataFrame({"X": [*range(99), 98], "Y": ["a"] * 99 + ["b"]})
    result = fw.discover_dependencies(df, max_key_size=1)
    d = dependency(result)
    assert d["modal_accuracy"] == 0.99
    assert d["repeated_rows"] == 2
    assert d["repeat_coverage"] == 0.02
    assert d["repeat_modal_accuracy"] == 0.5
    assert any(f["measurements"] == measured(d) for f in result["findings"])


def test_empty_denominators_and_missing_category():
    for df, q, coverage in [
        (sparse_frame().iloc[:0], 0, None),
        (sparse_frame().assign(Y=None), 4, 0.0),
    ]:
        result = fw.discover_dependencies(df, max_key_size=1)
        d = dependency(result)
        assert d["determinant_evaluated_rows"] == q
        assert d["target_coverage"] == coverage
        assert d["exact"] is None
        assert d["modal_accuracy"] is None
        assert d["repeat_coverage"] is None
        assert d["repeat_modal_accuracy"] is None
        assert result["candidates"][0]["determines"] == []
        assert result["candidates"][0]["global_targets_tested"] == 1
    df = pd.DataFrame({"X": [None, None, 2, 2], "Y": ["a", None, "b", "NA"]})
    result = fw.discover_dependencies(df, dropna=False, missing={"Y": ["NA"]})
    d = dependency(result)
    assert d["evaluated_rows"] == d["determinant_evaluated_rows"] == 4
    assert d["target_missing_excluded_rows"] == d["missing_excluded_rows"] == 0
    assert d["target_coverage"] == 0.5
    assert d["repeated_rows"] == 4
    assert d["repeat_modal_accuracy"] == 0.5


def test_scoped_composite_contexts_and_global_counts():
    df = pd.DataFrame(
        {
            "X": [1, 1, 1, None, 2],
            "Z": [1, 1, 2, 2, 2],
            "Y": ["a", "a", None, "a", "b"],
            "C": [None, None, "other", "other", "other"],
        },
        index=[0] * 5,
    )
    result = fw.discover_dependencies(
        df,
        features=["X", "Z", "Y"],
        by=["C"],
        scope=fw.Scope.from_positions(df, [0, 1, 2, 3]),
        max_key_size=2,
    )
    local = dependency(result, ("X", "Z"), context={"C": None})
    assert (
        local["target_coverage"] == local["repeat_coverage"] == local["repeat_modal_accuracy"] == 1
    )
    d = dependency(result, ("X", "Z"))
    assert d["determinant_evaluated_rows"] == 3
    assert d["target_observed_rows"] == 2
    assert d["target_missing_excluded_rows"] == 1
    assert d["missing_excluded_rows"] == 2
    assert d["repeated_rows"] == 2
    assert all(
        c["global_targets_tested"] == c["global_targets_possible"] for c in result["candidates"]
    )
    f = next(f for f in result["findings"] if f["measurements"] == measured(local))
    assert result.select(df, f["id"]).positions == (0, 1)
    # Conditional exactness alone must never enter the global exact lists.
    conditional = fw.discover_dependencies(
        pd.DataFrame(
            {"X": [1] * 4, "Y": ["a", "a", "b", "b"], "C": ["rare", "rare", "other", "other"]}
        ),
        features=["X", "Y"],
        by=["C"],
    )
    assert conditional["candidates"][0]["determines"] == []
    assert conditional["candidates"][0]["determines_with_repeated_support"] == []


def test_budgets_do_not_count_graph_or_conditional_tests():
    df = sparse_frame().assign(Z=[1, 1, 2, 2])
    for budget, expected in [(0, [0, 0, 0]), (1, [1, 0, 0]), (3, [2, 0, 0])]:
        result = fw.discover_dependencies(
            df, max_key_size=1, by=["Z"], limits={"max_dependency_tests": budget}
        )
        assert [c["global_targets_tested"] for c in result["candidates"]] == expected
        assert all(c["global_targets_possible"] == 2 for c in result["candidates"])
        assert result["grain_views"]
    result = fw.discover_dependencies(df, max_key_size=1, limits={"max_grain_views": 0})
    assert all(c["global_targets_tested"] == 2 for c in result["candidates"])
    assert not result["grain_views"]


def ranking_frame():
    return pd.DataFrame(
        {
            "X": [1, 1, 2, 2, 3, 3, 4, 4],
            "Z": ["a", "b", "a", "b", "c", "d", "c", "d"],
            "W": ["a", "b", "a", "b", "c", "d", "c", "d"],
            **{f"Y{i}": [10, None, 20, None, 30, None, 40, None] for i in range(3)},
        }
    )


def overview_of(dependencies):
    return {
        "kind": "overview",
        "sections": {"dependencies": dependencies, "missingness": {}, "paths": {}},
    }


def test_supported_ranking_reverses_singleton_advantage():
    result = fw.discover_dependencies(ranking_frame(), max_key_size=1).to_dict()
    original_order = [c["columns"] for c in result["candidates"]]
    x, z = result["candidates"][:2]
    assert len(x["determines"]) > len(z["determines"])
    assert x["repeated_rows"] == z["repeated_rows"]
    assert x["determines_with_repeated_support"] == []
    assert z["determines_with_repeated_support"] == ["W"]
    projected = fw.visualization_data(overview_of(result))["overview"]["grains"]
    order = [c["columns"] + c["equivalent"] for c in projected]
    position = {c: i for i, columns in enumerate(order) for c in columns}
    assert position["Z"] < position["X"]
    assert original_order == [c["columns"] for c in result["candidates"]]


def test_saved_details_below_threshold_and_disclosure():
    df = sparse_frame()
    for dropna in (True, False):
        result = fw.discover_dependencies(df, dropna=dropna, min_accuracy=1, max_key_size=1)
        saved = fw.Result.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        assert saved.to_dict() == result.to_dict()
        # The full projection keeps every completed test and its support fields,
        # including tests below the finding threshold.
        projection = fw.visualization_data(saved)
        assert len(projection["dependencies"]) == 2
        d = dependency(projection)
        assert (d["target_observed_rows"], d["determinant_evaluated_rows"]) == (2, 4)
        assert projection["candidates"][0]["global_targets_tested"] == 1
        if not dropna:
            # Missing Y participates as a category, so X no longer determines Y.
            assert (d["exact"], d["evaluated_rows"], d["repeat_modal_accuracy"]) == (False, 4, 0.5)
            assert not any(f["measurements"]["determinant"] == ["X"] for f in saved["findings"])
