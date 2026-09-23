from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fieldwork import KeySpec, census, explore, grain, levels, visualization_data
from fieldwork._explore._kernels import exact_pair_ids


def test_typed_scalar_identity_and_strict_json() -> None:
    frame = pd.DataFrame({"mixed": pd.Series([True, 1, 1.0, "1", None], dtype=object)})
    result = levels(frame)
    records = result["per_feature"][0]["levels"]
    assert {record["value"]["type"] for record in records} == {
        "boolean",
        "integer",
        "float",
        "string",
        "missing",
    }
    json.dumps(result.to_dict(), allow_nan=False)


def test_nonfinite_and_large_integer_serialization() -> None:
    frame = pd.DataFrame({"x": pd.Series([2**63 + 5, float("inf"), float("-inf")], dtype=object)})
    values = [item["value"] for item in levels(frame)["per_feature"][0]["levels"]]
    assert {item.get("value") for item in values} == {str(2**63 + 5), "inf", "-inf"}


def test_temporal_identity_normalizes_aware_instants() -> None:
    utc = datetime(2025, 1, 1, 12, tzinfo=UTC)
    eastern = utc.astimezone(timezone(timedelta(hours=-5)))
    result = levels(pd.DataFrame({"when": pd.Series([utc, eastern], dtype=object)}))
    assert result["per_feature"][0]["levels_total"] == 1
    value = result["per_feature"][0]["levels"][0]["value"]
    assert value["type"] == "datetime_aware"
    assert value["resolution"] == "nanosecond"


def test_timedeltas_order_numerically_and_display_as_durations() -> None:
    durations = pd.to_timedelta(["100s", "-5s", "9s", "10s"])
    result = levels(pd.DataFrame({"d": durations}))
    ordered = [int(level["value"]["value"]) for level in result["per_feature"][0]["levels"]]
    assert ordered == sorted(d.value for d in durations)
    labels = [row["label"] for row in visualization_data(result)["features"][0]["rows"]]
    # Signed durations such as "-0 days 00:00:05", not raw nanoseconds.
    signed = [
        (-1 if label.startswith("-") else 1) * pd.Timedelta(label.lstrip("-")).value
        for label in labels
    ]
    assert signed == ordered


@settings(max_examples=40, deadline=None, derandomize=True)
@given(st.lists(st.tuples(st.integers(0, 5), st.integers(0, 5)), max_size=30), st.booleans())
def test_exact_pair_ids_number_each_distinct_pair_once(pairs, force_fallback) -> None:
    import numpy as np

    parents = np.array([p for p, _ in pairs], dtype=np.int64)
    children = np.array([c for _, c in pairs], dtype=np.int64)
    options = {"packing_limit": 1} if force_fallback else {}
    ids, unique = exact_pair_ids(parents, children, **options)
    assert sorted(unique) == sorted(set(pairs))
    assert [unique[i] for i in ids.tolist()] == pairs


def test_tuple_label_is_atomic_and_composite_key_explicit() -> None:
    label = ("patient", 1)
    frame = pd.DataFrame({label: [1, 1], "side": ["L", "R"]})
    assert levels(frame, [label])["per_feature"][0]["column"]["type"] == "tuple"
    result = grain(frame, [KeySpec("composite", (label, "side"))])
    assert result["keys"][0]["name"] == "composite"


def test_invalid_inputs_are_actionable() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(ValueError, match="Duplicate"):
        levels(frame)
    with pytest.raises(ValueError, match="top_n"):
        census(pd.DataFrame({"x": [1]}), ["x"], top_n=0)
    with pytest.raises(ValueError, match="schema role"):
        levels(pd.DataFrame({"x": [1]}), schema={"x": "measure"})
    with pytest.raises(TypeError, match="column label"):
        levels(pd.DataFrame([[1]], columns=[1.5]))


def test_dataframe_is_not_mutated() -> None:
    frame = pd.DataFrame({"a": [2, 1, None], "b": ["x", "y", "x"]})
    before = frame.copy(deep=True)
    explore(frame, ["a", "b"], candidate_keys=["a"])
    pd.testing.assert_frame_equal(frame, before)


def test_unsupported_values_fail() -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        levels(pd.DataFrame({"x": [{"mutable": "mapping"}]}))
