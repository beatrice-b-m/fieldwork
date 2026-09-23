"""Known answers for pair relation classes, Cramer's V and absence classes."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fieldwork import pairs


def only_pair(frame: pd.DataFrame, **options):
    result = pairs(frame, ["a", "b"], **options)
    assert result["processed_pairs"] == 1
    return result["pairs"]


@pytest.mark.parametrize(
    ("a", "b", "relation", "cramers_v"),
    [
        # Each a has one b and each b has one a.
        (["x", "x", "y", "y", "z"], [1, 1, 2, 2, 3], "1:1", 1.0),
        # One a (x) spans two b values; every b has a single a. V is 1 because
        # b determines a: chi2 = 3, n = 3, min(k) - 1 = 1.
        (["x", "x", "y"], [1, 2, 3], "1:n", 1.0),
        (["x", "y", "z"], [1, 1, 2], "n:1", 1.0),
        # Independent 2x2 table: every cell is at its expected count.
        (["x", "x", "y", "y"], [1, 2, 1, 2], "n:m", 0.0),
        # Table x:{1: 2, 2: 1}, y:{2: 1}. Expected counts 1.5, 1.5, 0.5, 0.5
        # give chi2 = 1/6 + 1/6 + 1/2 + 1/2 = 4/3, so V = sqrt(4/3 / 4).
        (["x", "x", "x", "y"], [1, 1, 2, 2], "n:m", math.sqrt(1 / 3)),
        # 3x2 table: a = 1, 2, 3 (twice each) against b = 1, 1, 1, 2, 2, 2.
        # chi2 = 4, n = 6, min(k) - 1 = 1, V = sqrt(4 / 6).
        ([1, 1, 2, 2, 3, 3], [1, 1, 1, 2, 2, 2], "n:m", math.sqrt(2 / 3)),
    ],
)
def test_relation_classes_and_cramers_v_match_hand_computation(a, b, relation, cramers_v):
    (record,) = only_pair(pd.DataFrame({"a": a, "b": b}))
    assert record["relation"] == relation
    assert record["relation_reason"] is None
    assert record["cramers_v"] == pytest.approx(cramers_v)
    assert record["cramers_v_reason"] is None
    assert record["evaluated_rows"] == len(a)
    assert record["observed_cells"] == len(set(zip(a, b)))
    assert record["marginals"]["a_supported_levels"] == len(set(a))
    assert record["marginals"]["b_supported_levels"] == len(set(b))
    assert sum(item["count"] for item in record["marginals"]["a"]) == len(a)
    assert sum(item["count"] for item in record["marginals"]["b"]) == len(b)


def test_cramers_v_is_symmetric_and_relation_direction_flips():
    frame = pd.DataFrame({"a": ["x", "x", "y", "z"], "b": [1, 2, 3, 3]})
    forward = pairs(frame, ["a", "b"])["pairs"][0]
    backward = pairs(frame, ["b", "a"])["pairs"][0]
    assert forward["relation"] == "n:m"
    assert backward["cramers_v"] == pytest.approx(forward["cramers_v"])
    one_to_many = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 3]})
    assert pairs(one_to_many, ["a", "b"])["pairs"][0]["relation"] == "1:n"
    assert pairs(one_to_many, ["b", "a"])["pairs"][0]["relation"] == "n:1"


@pytest.mark.parametrize(
    ("frame", "options", "relation", "relation_reason", "v_reason", "evaluated"),
    [
        # A constant dimension has no association denominator (min(k) - 1 = 0).
        (pd.DataFrame({"a": ["x", "x"], "b": [1, 2]}), {}, "1:n", None, "constant_dimension", 2),
        (pd.DataFrame({"a": ["x"], "b": [1]}), {}, "1:1", None, "constant_dimension", 1),
        # Excluding missing values can leave no population at all.
        (
            pd.DataFrame({"a": [None, "x"], "b": [1.0, None]}),
            {"dropna": True},
            None,
            "empty_population",
            "empty_population",
            0,
        ),
        (pd.DataFrame({"a": [], "b": []}), {}, None, "empty_population", "empty_population", 0),
    ],
)
def test_degenerate_tables_are_undefined_with_a_reason(
    frame, options, relation, relation_reason, v_reason, evaluated
):
    (record,) = only_pair(frame, **options)
    assert record["relation"] == relation
    assert record["relation_reason"] == relation_reason
    assert record["cramers_v"] is None
    assert record["cramers_v_reason"] == v_reason
    assert record["evaluated_rows"] == evaluated


def test_missing_values_are_a_category_unless_dropped():
    frame = pd.DataFrame({"a": ["x", None, None, "y"], "b": [1, 2, 2, 1]})
    (kept,) = only_pair(frame)
    # Missing a is its own level: {x, missing, y} against {1, 2}.
    assert kept["evaluated_rows"] == 4
    assert kept["marginals"]["a_supported_levels"] == 3
    assert kept["relation"] == "n:1"
    (dropped,) = only_pair(frame, dropna=True)
    assert dropped["evaluated_rows"] == 2
    assert dropped["missing_excluded_rows"] == 2
    # x -> 1 and y -> 1 remain: b is constant, so V has no denominator.
    assert dropped["relation"] == "n:1"
    assert dropped["cramers_v_reason"] == "constant_dimension"


def test_contexts_restrict_rows_and_keep_global_evidence():
    frame = pd.DataFrame({"a": ["x", "x", "x", "y"], "b": [1, 1, 2, 2], "g": ["p", "p", "q", "q"]})
    global_pair, local = only_pair(frame, pair_contexts=[{"g": "p"}])
    assert global_pair["context"] == []
    assert global_pair["cramers_v"] == pytest.approx(math.sqrt(1 / 3))
    assert local["evaluated_rows"] == 2
    assert local["restriction_excluded_rows"] == 2
    assert local["relation"] == "1:1"
    assert local["cramers_v_reason"] == "constant_dimension"
    (unmatched_global, unmatched) = only_pair(frame, pair_contexts=[{"g": "absent"}])
    assert unmatched["evaluated_rows"] == 0
    assert unmatched["relation_reason"] == "empty_population"
    assert unmatched_global == global_pair


def test_absence_classes_match_hand_counts():
    frame = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 2], "g": ["p", "p", "q"]})
    domains = {"a": ["x", "y", "z"], "b": [1, 2, 3]}
    global_pair, local = only_pair(
        frame, include_absence=True, reference_domains=domains, pair_contexts=[{"g": "p"}]
    )
    # 3x3 declared domain, observed (x,1), (x,2), (y,2). z and 3 have no support
    # anywhere: 3 + 3 - 1 = 5 cells. (y,1) is unobserved within supported margins.
    assert global_pair["domains"] == {
        "a_size": 3,
        "b_size": 3,
        "a_source": "caller_declared",
        "b_source": "caller_declared",
    }
    assert global_pair["absence"]["absent_cells"] == 6
    assert global_pair["absence"]["classes"] == {
        "unobserved_zero_support": 5,
        "level_absent_under_parent": 0,
        "unobserved_within_supported_margins": 1,
    }
    # Within g = p only x occurs, so y's cells (y,1) and (y,2) are absent under it.
    assert local["absence"]["absent_cells"] == 7
    assert local["absence"]["classes"] == {
        "unobserved_zero_support": 5,
        "level_absent_under_parent": 2,
        "unobserved_within_supported_margins": 0,
    }
    for record in (global_pair, local):
        absence = record["absence"]
        assert absence["total_cells"] == absence["observed_cells"] + absence["absent_cells"]
        assert len(absence["examples"]) + absence["examples_omitted"] == absence["absent_cells"]
    with pytest.raises(ValueError, match="omits observed levels"):
        pairs(frame, ["a", "b"], include_absence=True, reference_domains={"a": ["x"]})


def test_pair_and_context_budgets_are_reported():
    frame = pd.DataFrame({"a": [1, 2], "b": [1, 2], "c": [1, 2], "g": ["p", "q"]})
    result = pairs(frame, ["a", "b", "c"], max_pairs=2, pair_contexts=[{"g": "p"}, {"g": "q"}])
    assert (result["requested_pairs"], result["processed_pairs"], result["omitted_pairs"]) == (
        3,
        2,
        1,
    )
    assert result["processed_contexts"] == 3 and result["omitted_contexts"] == 0
    assert [r["columns"] for r in result["pairs"]] == [["a", "b"]] * 3 + [["a", "c"]] * 3
    limited = pairs(frame, ["a", "b", "c"], max_contexts=1, pair_contexts=[{"g": "p"}])
    assert limited["omitted_contexts"] == 1
    assert all(r["context"] == [] for r in limited["pairs"])
