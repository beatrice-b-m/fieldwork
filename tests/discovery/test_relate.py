"""relate: key coverage, cardinality, agreement and references across two tables."""

import json

import numpy as np
import oracle
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import fieldwork as fw

KEYS = [1, 2, 2.0, 3, "1", "2", True, None, float("nan")]
VALUES = ["a", "b", 1, None]


def frames(rows):
    return st.lists(
        st.fixed_dictionaries(
            {
                "k": st.sampled_from(KEYS),
                "j": st.sampled_from(["x", "y", None]),
                "v": st.sampled_from(VALUES),
            }
        ),
        min_size=0,
        max_size=rows,
    ).map(lambda records: pd.DataFrame(records, columns=["k", "j", "v"], dtype=object))


def finding(result, part, **selector):
    matches = [
        f
        for f in result["findings"]
        if f["selector"]["part"] == part
        and all(f["selector"].get(k) == v for k, v in selector.items())
    ]
    assert len(matches) <= 1
    return matches[0] if matches else None


def selected(result, left, right, record, side, exceptions=False):
    scope = result.select(left, record["id"], side=side, right=right, exceptions=exceptions)
    return list(scope.positions)


def state(hits, total):
    return "empty" if total == 0 else "all" if hits == total else "none" if hits == 0 else "some"


@settings(max_examples=150, deadline=None)
@given(
    frames(8),
    frames(8),
    st.sampled_from([["k"], ["k", "j"]]),
    st.sampled_from(["typed", "text"]),
    st.booleans(),
)
def test_every_measurement_and_selection_matches_an_enumeration_oracle(
    left, right, key, mode, dropna
):
    on = {c: c for c in key}
    result = fw.relate(left, right, on=on, compare={"v": "v"}, match=mode, dropna=dropna)
    expected = oracle.relation(left, right, on, {"v": "v"}, mode=mode, dropna=dropna)
    matched = len(expected["matched"])
    for side, name in enumerate(("left", "right")):
        record = result["sides"][name]
        assert record["rows"]["incomplete_key"] == expected["incomplete"][side]
        assert record["keys"]["distinct"] == len(expected["keys"][side])
        assert record["keys"]["matched"] == matched
    for side, direction in enumerate(("left_to_right", "right_to_left")):
        coverage = result["key_coverage"][direction]
        assert coverage["state"] == state(matched, len(expected["keys"][side]))
        assert coverage["found_rows"] == len(expected["found_rows"][side])
        found = finding(result, "coverage", direction=direction)
        name = ("left", "right")[side]
        assert selected(result, left, right, found, name) == expected["found_rows"][side]
        assert (
            selected(result, left, right, found, name, exceptions=True)
            == expected["unfound_rows"][side]
        )
    relation = result["relation"]
    assert relation["joined_rows"] == expected["joined_rows"]
    assert relation["matched_keys"] == matched
    record = finding(result, "relation")
    if not matched:
        assert relation["type"] is None and record is None
    else:
        names = {(False, False): "1:1", (False, True): "1:n", (True, False): "n:1"}
        assert relation["type"] == names.get(tuple(expected["many"]), "n:m")
        for side, name in enumerate(("left", "right")):
            assert selected(result, left, right, record, name) == expected["repeated_rows"][side]
            assert (
                selected(result, left, right, record, name, exceptions=True)
                == expected["single_rows"][side]
            )
    agreement, truth = result["agreement"][0], expected["agreement"][("v", "v")]
    for state_name in ("agree", "disagree", "ambiguous", "unavailable"):
        field = {"agree": "agreeing", "disagree": "disagreeing"}.get(state_name, state_name)
        assert agreement[f"{field}_keys"] == truth["counts"][state_name]
    record = finding(result, "agreement", compare=["v", "v"])
    assert (record is None) == (not matched)
    if record is not None:
        for side, name in enumerate(("left", "right")):
            assert selected(result, left, right, record, name) == truth["agree_rows"][side]
            assert (
                selected(result, left, right, record, name, exceptions=True)
                == truth["disagree_rows"][side]
            )


def test_known_answer_for_an_image_and_clinical_export():
    images = pd.DataFrame(
        {"acc": [10, 10, 11, 12, None], "patient": ["p1", "p1", "p2", "p3", "p4"]}
    )
    clinical = pd.DataFrame({"acc": [10.0, 11.0, 11.0, 13.0], "patient": ["p1", "p2", "p9", "p5"]})
    result = fw.relate(
        images,
        clinical,
        on={"acc": "acc"},
        compare={"patient": "patient"},
        table_ids=("image", "clinical"),
    )
    assert result.kind == "relation"
    assert result["key_coverage"]["left_to_right"] == {
        "state": "some",
        "keys": 3,
        "found": 2,
        "not_found": 1,
        "found_rows": 3,
        "not_found_rows": 1,
    }
    assert result["key_coverage"]["right_to_left"]["state"] == "some"
    assert result["sides"]["left"]["rows"]["incomplete_key"] == 1
    # acc 10: two image rows, one clinical row; acc 11: one image row, two clinical rows.
    assert result["relation"]["type"] == "n:m"
    assert result["relation"]["joined_rows"] == 4
    agreement = result["agreement"][0]
    # acc 10 agrees; acc 11 has two clinical patients, so it is ambiguous, not a disagreement.
    assert (agreement["state"], agreement["ambiguity"]) == ("all", "some")
    assert (agreement["agreeing_keys"], agreement["ambiguous_keys"]) == (1, 1)
    features = {(f["table"], f["column"]) for f in result.findings[0]["features"]}
    assert features == {("image", "acc"), ("clinical", "acc")}


def test_numbers_match_numerically_and_text_mode_matches_integer_text():
    ints = pd.DataFrame({"id": pd.array([1, 2, None], dtype="Int64")})
    floats = pd.DataFrame({"id": [1.0, 2.0, np.nan]})
    text = pd.DataFrame({"id": ["1", "02", "2.5"]})
    assert (
        fw.relate(ints, floats, on={"id": "id"})["key_coverage"]["left_to_right"]["state"] == "all"
    )
    typed = fw.relate(ints, text, on={"id": "id"})
    assert typed["key_coverage"]["left_to_right"]["state"] == "none"
    [warning] = typed["warnings"]
    assert warning["code"] == "VALUE_KIND_MISMATCH"
    assert (warning["left_kinds"], warning["right_kinds"]) == (["number"], ["string"])
    as_text = fw.relate(ints, text, on={"id": "id"}, match="text")
    # "1" matches 1; "02" is not the decimal text of 2.
    assert as_text["key_coverage"]["left_to_right"]["found"] == 1
    assert as_text["warnings"] == []
    booleans = pd.DataFrame({"id": [True, False]})
    numbers = pd.DataFrame({"id": [1, 0]})
    assert fw.relate(booleans, numbers, on={"id": "id"})["relation"]["matched_keys"] == 0


def test_temporal_keys_match_by_value_across_dtypes():
    left = pd.DataFrame({"day": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    right = pd.DataFrame({"day": pd.Series(pd.to_datetime(["2024-01-02"]), dtype=object)})
    result = fw.relate(left, right, on={"day": "day"})
    assert result["key_coverage"]["right_to_left"]["state"] == "all"


def test_composite_keys_exclude_incomplete_rows_and_sentinels_apply_per_side():
    left = pd.DataFrame({"site": ["A", "A", "B", None], "visit": [1, -1, 1, 1]})
    right = pd.DataFrame({"s": ["A", "B", "B"], "v": [1, 1, -1]})
    result = fw.relate(left, right, on={"site": "s", "visit": "v"}, missing=({"visit": [-1]}, None))
    assert result["sides"]["left"]["rows"]["incomplete_key"] == 2
    assert result["sides"]["right"]["rows"]["incomplete_key"] == 0
    assert result["sides"]["left"]["missing_convention"]["sentinels"] == {"visit": [-1]}
    assert result["key_coverage"]["left_to_right"]["state"] == "all"
    # The right side has no sentinel, so ("B", -1) is its own unmatched key.
    assert result["key_coverage"]["right_to_left"]["state"] == "some"


def test_dropna_false_compares_missing_as_a_value():
    left = pd.DataFrame({"k": [1, 2], "v": [None, "a"]})
    right = pd.DataFrame({"k": [1, 2], "v": [None, None]})
    ignored = fw.relate(left, right, on={"k": "k"}, compare={"v": "v"})["agreement"][0]
    assert (ignored["state"], ignored["unavailable_keys"]) == ("empty", 2)
    counted = fw.relate(left, right, on={"k": "k"}, compare={"v": "v"}, dropna=False)
    assert counted["agreement"][0]["state"] == "some"


def test_scopes_apply_per_side_and_selection_returns_source_positions():
    left = pd.DataFrame({"k": [1, 2, 3, 4]})
    right = pd.DataFrame({"k": [4, 3, 2, 9]})
    scope = fw.Scope.from_positions(right, [1, 3], name="late")
    result = fw.relate(left, right, on={"k": "k"}, scopes=(None, scope))
    assert result["sides"]["right"]["scope"]["name"] == "late"
    backward = finding(result, "coverage", direction="right_to_left")
    chosen = result.select(left, backward["id"], side="right", right=right, exceptions=True)
    assert chosen.positions == (3,)
    assert chosen.dataset_id == fw.Scope.from_positions(right, []).dataset_id
    assert chosen.parent == "late"
    with pytest.raises(ValueError, match="different ordered dataset"):
        fw.relate(left, right, on={"k": "k"}, scopes=(scope, None))


def test_self_reference_resolution_and_reciprocity_match_the_oracle():
    frame = pd.DataFrame(
        {
            "acc": [1, 1, 2, 3, 4, 5, 6],
            "linked": [2, 2, 1, 4, 9, 5, None],
            "patient": ["a", "a", "a", "b", "c", "d", "e"],
        }
    )
    result = fw.relate(frame, on={"linked": "acc"}, compare={"patient": "patient"})
    expected = oracle.reciprocity(frame, "linked", "acc")
    assert result["self_reference"] is True
    assert result["sides"]["left"]["source"] == result["sides"]["right"]["source"]
    # 9 does not resolve; every other complete reference does.
    assert result["key_coverage"]["left_to_right"]["state"] == "some"
    reciprocity = result["reciprocity"]
    assert reciprocity["resolved_references"] == len(expected["edges"])
    assert reciprocity["reciprocated"] == len(expected["reciprocated"])
    assert reciprocity["self_referencing_keys"] == len(expected["loops"])
    assert (reciprocity["state"], reciprocity["self_references"]) == ("some", "some")
    record = finding(result, "reciprocity")
    assert list(result.select(frame, record["id"]).positions) == expected["reciprocated_rows"]
    assert (
        list(result.select(frame, record["id"], exceptions=True).positions)
        == expected["unreciprocated_rows"]
    )
    # Referenced rows sharing the patient: 1 → 2 and 2 → 1 agree, 3 → 4 does not.
    assert result["agreement"][0]["state"] == "some"
    with pytest.raises(ValueError, match="has no right rows"):
        result.select(frame, record["id"], side="right")
    with pytest.raises(ValueError, match="omit right"):
        result.select(frame, record["id"], right=frame)


def test_self_reference_sides_can_have_different_scopes():
    frame = pd.DataFrame({"acc": [1, 2, 3], "linked": [2, 3, None]})
    targets = fw.Scope.from_positions(frame, [0, 1], name="early")
    result = fw.relate(frame, on={"linked": "acc"}, scopes=(None, targets))
    # 3 is referenced but its row is outside the target scope.
    assert result["key_coverage"]["left_to_right"]["state"] == "some"


def test_row_access_verifies_both_sources():
    left = pd.DataFrame({"k": [1, 2]})
    right = pd.DataFrame({"k": [2, 3]})
    result = fw.relate(left, right, on={"k": "k"})
    identifier = finding(result, "relation")["id"]
    # A 1:1 relation: each matched key occurs once per side, so its rows are exceptions.
    rows = result.inspect(left, identifier, exceptions=True, side="right", right=right)
    assert rows["k"].tolist() == [2]
    with pytest.raises(ValueError, match="Pass right="):
        result.inspect(left, identifier, side="right")
    with pytest.raises(ValueError, match="Pass right="):
        result.select(left, identifier)
    with pytest.raises(ValueError, match="Right source dataset differs"):
        result.select(left, identifier, right=right.iloc[::-1])
    with pytest.raises(ValueError, match="Source dataset differs"):
        result.inspect(right, identifier)
    with pytest.raises(ValueError, match="relation results only"):
        fw.missingness(left).inspect(left, 0, side="right")


def test_saved_results_recompute_and_recipes_reuse_both_sources():
    left = pd.DataFrame({"k": [1, 2, 2], "v": ["a", "b", "b"]})
    right = pd.DataFrame({"k": [2, 3], "v": ["b", "c"]})
    result = fw.relate(left, right, on={"k": "k"}, compare={"v": "v"}, limits={"example_limit": 1})
    restored = fw.Result.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
    for record in result.findings:
        for side in record["selector"]["sides"]:
            assert restored.select(left, record["id"], side=side, right=right) == result.select(
                left, record["id"], side=side, right=right
            )
    again = result.recompute(left, right=right, limits={"example_limit": 5})
    assert again["parameters"]["limits"]["example_limit"] == 5
    assert again["relation"] == result["relation"]
    with pytest.raises(ValueError, match="Pass right="):
        result.recompute(left)
    recipe = fw.Recipe("relate", result["parameters"])
    delivery = pd.DataFrame({"k": [3, 4], "v": ["c", "d"]})
    assert recipe.run(delivery, right=right)["key_coverage"]["left_to_right"]["state"] == "some"
    with pytest.raises(ValueError, match="pass a scope when running"):
        fw.Recipe("relate", {"on": {"k": "k"}, "scopes": [None, None]})


def test_empty_sides_and_invalid_arguments():
    left = pd.DataFrame({"k": pd.Series([], dtype=float)})
    right = pd.DataFrame({"k": [1]})
    result = fw.relate(left, right, on={"k": "k"})
    assert result["key_coverage"]["left_to_right"]["state"] == "empty"
    assert result["key_coverage"]["right_to_left"]["state"] == "none"
    assert result["relation"]["type"] is None
    assert [f["pattern"] for f in result.findings] == ["key_coverage", "key_coverage"]
    with pytest.raises(ValueError, match="at least one"):
        fw.relate(left, right, on={})
    with pytest.raises(KeyError):
        fw.relate(left, right, on={"k": "missing"})
    with pytest.raises(ValueError, match="must not repeat"):
        fw.relate(left, right, on={"k": "k"}, compare={"a": "k", "b": "k"})
    with pytest.raises(TypeError, match="pair"):
        fw.relate(left, right, on={"k": "k"}, table_ids="left")
    with pytest.raises(ValueError, match="match"):
        fw.relate(left, right, on={"k": "k"}, match="loose")


def test_integer_column_labels_are_matched_by_their_string_form():
    left = pd.DataFrame([[1, "a"], [2, "b"]])
    right = pd.DataFrame([[2, "b"]])
    result = fw.relate(left, right, on={0: 0}, compare={1: 1})
    assert result["parameters"]["on"] == {"0": "0"}
    assert result["agreement"][0]["state"] == "all"
