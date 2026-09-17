from __future__ import annotations

import math

import pandas as pd

from fieldwork import KeySpec, explore, grain, infer_schema


def test_fd_support_and_violations() -> None:
    frame = pd.DataFrame({"id": [1, 1, 2], "target": ["a", "b", "c"]})
    record = grain(frame, ["id"])["dependencies"][0]
    assert record["holds"] is False
    assert record["violating_groups"] == 1
    assert record["affected_rows"] == 2
    assert record["singleton_groups"] == 1
    assert record["repeated_groups"] == 1


def test_empty_fd_is_undefined() -> None:
    frame = pd.DataFrame({"id": [None], "target": [None]})
    record = grain(frame, ["id"], dropna=True)["dependencies"][0]
    assert record["holds"] is None
    assert record["undefined_reason"] == "no_evaluated_groups"


def test_schema_proposal_marks_requested_fd_probe_as_evaluated() -> None:
    frame = pd.DataFrame({"id": [1, 2], "target": ["a", "b"]})
    proposal = infer_schema(frame, candidate_keys=["id"])
    target = next(item for item in proposal["proposals"] if item["column"]["value"] == "target")
    assert target["fd_evidence"]["status"] == "evaluated"
    assert target["fd_evidence"]["keys"][0]["holds"] is True


def test_pairs_relation_cramers_and_absence() -> None:
    frame = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 2]})
    pair = explore(
        frame,
        ["a", "b"],
        include_absence=True,
        reference_domains={"a": ["x", "y", "z"], "b": [1, 2, 3]},
    )["sections"]["pairs"]["pairs"][0]
    assert pair["relation"] == "n:m"
    assert pair["cramers_v"] is not None and math.isfinite(pair["cramers_v"])
    absence = pair["absence"]
    assert sum(absence["classes"].values()) == absence["absent_cells"]
    assert absence["classes"]["unobserved_zero_support"] == 5


def test_top_n_both_requires_pre_and_records_conditional_grain() -> None:
    frame = pd.DataFrame({"id": [1, 2, 3], "a": ["x", "x", "y"], "b": [1, 2, 2]})
    result = explore(
        frame,
        ["a", "b"],
        candidate_keys=[KeySpec("id", ("id",))],
        top_n=1,
        top_n_mode="pre",
        top_n_applies_to="both",
    )
    assert result["sections"]["grain"]["scope_metadata"]["source_scope"] == "s2"


def test_equivalent_and_incomparable_determinants_are_distinguished() -> None:
    equivalent = pd.DataFrame(
        {"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"], "target": [0, 0, 1, 1]}
    )
    summary = grain(equivalent, ["a", "b"])["targets"][-1]
    assert summary["equivalent_determinants"] == [["a", "b"]]
    assert summary["incomparable_candidates"] == []

    incomparable = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": ["x", "y", "x", "y"],
            "target": [0, 0, 0, 0],
        }
    )
    summary = grain(incomparable, ["a", "b"])["targets"][-1]
    assert summary["incomparable_candidates"] == [["a", "b"]]
