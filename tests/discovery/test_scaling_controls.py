"""Coverage, direct selection, and exports for bounded large-data workflows."""

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import fieldwork as fw
from fieldwork._explore._kernels import group_ids, modal_groups


@settings(max_examples=60, deadline=None, derandomize=True)
@given(st.lists(st.tuples(st.integers(-2, 5), st.integers(-2, 5)), max_size=80))
def test_modal_groups_matches_independent_counter(pairs):
    keys = np.array([p[0] for p in pairs], dtype=np.int64)
    targets = np.array([p[1] for p in pairs], dtype=np.int64)
    ids, sizes, modes, maxima, distinct = modal_groups(keys, targets)
    observed = list(dict.fromkeys(keys))
    assert ids.tolist() == [observed.index(k) for k in keys]
    for i, key in enumerate(observed):
        counts = Counter(t for k, t in pairs if k == key)
        assert sizes[i] == sum(counts.values())
        assert maxima[i] == max(counts.values())
        assert modes[i] == min(k for k, n in counts.items() if n == maxima[i])
        assert distinct[i] == len(counts)
    grouped = group_ids([keys, targets])
    unique_pairs = list(dict.fromkeys(pairs))
    assert grouped.tolist() == [unique_pairs.index(p) for p in pairs]


def test_key_comparisons_on_equal_size_populations_are_not_shared():
    # t1 and t2 are observed on different four-row populations. A and B partition
    # t1's rows identically but not t2's, so a comparison cached by population
    # size alone would report the same relationship for both targets.
    frame = pd.DataFrame(
        {
            "A": [1, 1, 2, 2, 3, 3],
            "B": ["x", "x", "y", "y", "y", "w"],
            "t1": [5, 5, 6, 6, None, None],
            "t2": [None, None, 7, 7, 7, 7],
        }
    )
    targets = {t["target"]: t for t in fw.grain(frame, ["A", "B"], dropna=True)["targets"]}
    assert targets["t1"]["determining_keys"] == targets["t2"]["determining_keys"] == ["A", "B"]
    assert targets["t1"]["equivalent_determinants"] == [["A", "B"]]
    assert targets["t2"]["equivalent_determinants"] == []
    assert targets["t2"]["incomparable_candidates"] == [["A", "B"]]


def fixture():
    return pd.DataFrame(
        {
            "key": [1, 1, 2, 2, None, 3, 3, 3],
            "a": [10, None, 20, 21, 30, 30, 30, 30],
            "b": [20, None, 40, 42, 60, 60, 60, 60],
            "s": ["x1", None, "y2", "y2", "z3", "x1", "x1", "x1"],
            "context": ["a", "a", "b", "b", "c", "c", None, "c"],
        },
        index=[4] * 8,
    )


@pytest.mark.parametrize("cap", [0, 1, 5])
def test_explicit_dependency_budgets_report_omissions(cap):
    frame = fixture()
    full = fw.discover_dependencies(frame, max_candidates=4, max_key_size=1)
    bounded = fw.discover_dependencies(
        frame, max_candidates=4, max_key_size=1, max_dependency_tests=cap, include_grain=False
    )
    assert bounded["dependencies"] == full["dependencies"][:cap]
    assert bounded["coverage"]["dependency_tests"] == cap
    assert bounded["coverage"]["dependency_tests_omitted"] == 16 - cap
    assert bounded["graph_selection"]["status"] == "not_requested"
    assert bounded["grain_views"] == []
    assert {x["reason"] for x in bounded["graph_selection"]["excluded"]} == {"graph_not_requested"}
    limited = fw.discover_dependencies(frame, max_candidates=4, max_grain_views=cap)
    assert len(limited["grain_views"]) <= cap
    assert limited["graph_selection"]["views_omitted"] == limited["graph_selection"][
        "views_possible"
    ] - len(limited["grain_views"])


def test_overview_sections_options_and_rendered_omissions():
    frame = fixture()
    overview = fw.explore(
        frame,
        sections=["missingness", "dependencies"],
        section_options={
            "missingness": {"features": ["a"]},
            "dependencies": {"max_candidates": 1, "include_grain": False},
        },
    )
    assert overview["sections"]["paths"] == {"status": "not_requested"}
    assert overview["sections"]["dependencies"]["coverage"]["candidates_evaluated"] == 1
    assert len(overview["sections"]["missingness"]["availability"]) == 1
    for render in (fw.render_plaintext, fw.render_html, fw.render_svg):
        assert "Not requested" in render(overview)
    assert overview["section_selection"]["omitted"] == ["paths", "value_patterns"]
    with pytest.raises(ValueError, match="section_options"):
        fw.explore(frame, sections=["paths"], section_options={"dependencies": {}})
    with pytest.raises(ValueError, match="source context"):
        fw.explore(frame, section_options={"dependencies": {"scope": None}})


@pytest.mark.parametrize(
    "kind", ["rows", "entities_any", "entities_all", "dependencies", "dependencies_na", "patterns"]
)
def test_direct_selection_matches_complete_analysis_without_replay(kind, monkeypatch):
    frame = fixture()
    context = {
        "scope": fw.Scope.from_positions(frame, [0, 1, 2, 3, 5, 6, 7]),
        "missing": {"a": [21]},
    }
    if kind.startswith("entities"):
        analysis = fw.missingness(
            frame,
            entity="key",
            unit="entities",
            entity_presence=kind.split("_")[1],
            by=["context"],
            example_limit=len(frame),
            **context,
        )
    elif kind == "rows":
        analysis = fw.missingness(
            frame, entity="key", by=["context"], example_limit=len(frame), **context
        )
    elif kind.startswith("dependencies"):
        analysis = fw.discover_dependencies(
            frame,
            min_accuracy=0,
            max_candidates=8,
            by=["context"],
            dropna=kind != "dependencies_na",
            example_limit=len(frame),
            **context,
        )
    else:
        analysis = fw.value_patterns(frame, by=["context"], example_limit=len(frame), **context)
    expected = analysis["findings"]
    saved = fw.InvestigationResult.from_dict(json.loads(json.dumps(analysis.to_dict())))

    def no_replay(*args, **kwargs):
        pytest.fail("selection replayed the entire analysis")

    monkeypatch.setattr(fw.InvestigationResult, "recompute", no_replay)
    for record in expected:
        for exceptions in (False, True):
            positions = record["exceptions" if exceptions else "examples"]["positions"]
            assert saved.select(frame, record["id"], exceptions=exceptions).positions == tuple(
                positions
            ), (kind, record["pattern"], exceptions)


def test_saved_export_roundtrip_supports_selection():
    frame = pd.concat([fixture()] * 30, ignore_index=True)
    analysis = fw.explore(frame)
    ordinary = json.loads(json.dumps(analysis.to_dict(), allow_nan=False))
    restored = fw.InvestigationResult.from_dict(ordinary)
    assert restored.to_dict() == ordinary
    record = next(f for f in analysis["findings"] if f["pattern"] == "availability")
    assert restored.select(frame, record["id"]) == analysis.select(frame, record["id"])
    foundation = fw.grain(frame, ["key"])
    saved = json.loads(json.dumps(foundation.to_dict(), allow_nan=False))
    assert fw.ExplorerResult.from_dict(saved).to_dict() == foundation.to_dict()


def test_grain_view_populations_store_bounded_examples():
    frame = pd.DataFrame({"key": np.arange(200) % 20, "value": np.arange(200) % 20 * 2})
    result = fw.discover_dependencies(frame, max_key_size=1, example_limit=3)
    for view in result["grain_views"]:
        examples = view["population"]["examples"]
        assert examples["positions"] == [0, 1, 2]
        assert examples["total"] == view["population"]["evaluated_rows"] == 200
        assert "positions" not in view["population"]
