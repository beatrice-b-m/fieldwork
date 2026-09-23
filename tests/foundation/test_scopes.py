from __future__ import annotations

import json

import pandas as pd
import pytest

from fieldwork import KeySpec, explore, grain


def test_contexts_do_not_change_global_pair_or_each_other() -> None:
    frame = pd.DataFrame(
        {"a": ["x", "x", "y"], "b": [1, 2, 2], "site": ["N", None, "S"], "batch": [None, "B", "B"]}
    )
    options = {"dropna": True, "include_absence": True}
    baseline = explore(frame, ["a", "b"], **options)["sections"]["pairs"]["pairs"][0]
    single = explore(frame, ["a", "b"], pair_contexts=[{"site": "N"}], **options)
    multiple = explore(frame, ["a", "b"], pair_contexts=[{"site": "N"}, {"batch": "B"}], **options)
    records = multiple["sections"]["pairs"]["pairs"]
    assert baseline == single["sections"]["pairs"]["pairs"][0] == records[0]
    assert baseline["relation"] == "n:m"
    assert baseline["cramers_v"] == pytest.approx(0.5)
    north = next(p for p in records if p["context"] and p["context"][0]["column"] == "site")
    single_north = single["sections"]["pairs"]["pairs"][1]
    assert north["evaluated_rows"] == single_north["evaluated_rows"] == 1
    assert north["missing_excluded_rows"] == 1
    assert north["restriction_excluded_rows"] == 1
    assert north["absence"] == single_north["absence"]


@pytest.mark.parametrize("per_parent", [False, True])
def test_pre_cohort_preserves_original_scope_in_pairs_and_grain(per_parent: bool) -> None:
    frame = pd.DataFrame(
        {
            "id": list(range(6)),
            "a": ["x", "x", "x", "y", None, "x"],
            "b": [1, 1, 1, 2, 1, 2],
            "context": ["N", "S", None, "N", "N", "N"],
            "target": [0, None, 0, 1, 0, 0],
        }
    )
    result = explore(
        frame,
        ["a", "b"],
        candidate_keys=["id"],
        dropna=True,
        top_n=1,
        top_n_mode="pre",
        top_n_per_parent=per_parent,
        top_n_applies_to="both",
        pair_contexts=[{"context": "N"}],
    )
    census_tree = result["sections"]["census"]["tree"]
    pairs = result["sections"]["pairs"]
    grain_data = result["sections"]["grain"]
    # The census pre-selection (a=x, b=1 after dropping the row missing a) is the
    # population both pairs and grain analyze.
    assert census_tree["evaluated_rows"] == 3
    assert census_tree["missing_excluded_rows"] == 1
    assert census_tree["restriction_excluded_rows"] == 2
    for section in (pairs, grain_data):
        assert section["scope"]["name"] == "census top_n cohort"
        assert section["scope"]["input_rows"] == 6
        assert section["scope"]["evaluated_rows"] == 3
        assert section["scope"]["restriction_excluded_rows"] == 3
    global_pair, local_pair = pairs["pairs"]
    assert global_pair["evaluated_rows"] == 3
    assert global_pair["missing_excluded_rows"] == global_pair["restriction_excluded_rows"] == 0
    assert local_pair["missing_excluded_rows"] == 1
    assert local_pair["restriction_excluded_rows"] == 1
    assert local_pair["evaluated_rows"] == 1
    target = next(d for d in grain_data["dependencies"] if d["target"] == "target")
    assert target["evaluated_rows"] == 2
    records = [*pairs["pairs"], *grain_data["dependencies"]]
    for record in records:
        assert pairs["scope"]["evaluated_rows"] == (
            record["missing_excluded_rows"]
            + record.get("restriction_excluded_rows", 0)
            + record["evaluated_rows"]
        )
    json.dumps(result.to_dict(), allow_nan=False)
    full_grain = explore(
        frame, ["a", "b"], candidate_keys=["id"], dropna=True, top_n=1, top_n_mode="pre"
    )["sections"]["grain"]
    assert full_grain["scope"]["name"] == "input"
    assert full_grain["scope"]["restriction_excluded_rows"] == 0


@pytest.mark.parametrize("composite", [False, True])
def test_grain_key_comparison_uses_target_population(composite: bool) -> None:
    frame = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "y"], "target": [0, 0, None]})
    keys = ["a", "b"]
    if composite:
        frame["constant"] = "k"
        keys = [KeySpec("a", ("a", "constant")), KeySpec("b", ("b", "constant"))]
    full = grain(frame, keys, dropna=True)
    common = grain(frame.dropna(subset=["target"]), keys, dropna=True)
    summary = next(t for t in full["targets"] if t["target"] == "target")
    expected = next(t for t in common["targets"] if t["target"] == "target")
    assert summary == expected
    assert summary["equivalent_determinants"] == [["a", "b"]]
    assert summary["incomparable_candidates"] == []


def test_grain_different_target_populations_remain_not_comparable() -> None:
    frame = pd.DataFrame({"a": [1, None, 2], "b": ["x", "y", None], "target": [0, 0, 0]})
    summary = grain(frame, ["a", "b"], dropna=True)["targets"][-1]
    assert summary["cross_key_comparison"] == "not_comparable"
    assert summary["coarsest_candidates"] == []
    assert summary["equivalent_determinants"] == []
    assert summary["incomparable_candidates"] == []
