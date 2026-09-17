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
    north = next(
        p for p in records if p["context"] and p["context"][0]["column"]["value"] == "site"
    )
    single_north = single["sections"]["pairs"]["pairs"][1]
    assert north["evaluated_rows"] == single_north["evaluated_rows"] == 1
    assert north["scope"]["missing_excluded_rows"] == 1
    assert north["scope"]["restriction_excluded_rows"] == 1
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
    pairs = result["sections"]["pairs"]
    global_pair, local_pair = pairs["pairs"]
    assert global_pair["scope"]["input_rows"] == 6
    assert global_pair["scope"]["evaluated_rows"] == 3
    assert global_pair["scope"]["missing_excluded_rows"] == 1
    assert global_pair["scope"]["restriction_excluded_rows"] == 2
    assert local_pair["scope"]["missing_excluded_rows"] == 2
    assert local_pair["scope"]["restriction_excluded_rows"] == 3
    assert local_pair["scope"]["evaluated_rows"] == 1
    assert pairs["scope_metadata"]["source_scope"] == "s2"
    grain_data = result["sections"]["grain"]
    target = next(d for d in grain_data["dependencies"] if d["target"]["value"] == "target")
    target_scope = next(s for s in grain_data["scopes"] if s["scope_id"] == target["scope_id"])
    assert target_scope["evaluated_rows"] == 2
    scopes = [p["scope"] for p in pairs["pairs"]] + grain_data["scopes"]
    for scope in scopes:
        assert scope["conditional"] is True
        assert "s2" in scope["lineage"]
        assert scope["input_rows"] == (
            scope["missing_excluded_rows"]
            + scope["restriction_excluded_rows"]
            + scope["evaluated_rows"]
        )
    json.dumps(result.to_dict(), allow_nan=False)
    full_grain = explore(
        frame, ["a", "b"], candidate_keys=["id"], dropna=True, top_n=1, top_n_mode="pre"
    )["sections"]["grain"]
    assert full_grain["scope_metadata"] is None
    assert all(s["input_rows"] == 6 and not s["conditional"] for s in full_grain["scopes"])


@pytest.mark.parametrize("composite", [False, True])
def test_grain_key_comparison_uses_target_population(composite: bool) -> None:
    frame = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "y"], "target": [0, 0, None]})
    keys = ["a", "b"]
    if composite:
        frame["constant"] = "k"
        keys = [KeySpec("a", ("a", "constant")), KeySpec("b", ("b", "constant"))]
    full = grain(frame, keys, dropna=True)
    common = grain(frame.dropna(subset=["target"]), keys, dropna=True)
    summary = next(t for t in full["targets"] if t["target"]["value"] == "target")
    expected = next(t for t in common["targets"] if t["target"]["value"] == "target")
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
