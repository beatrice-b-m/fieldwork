from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from fieldwork import KeySpec, census, explore, grain, levels, render_plaintext
from fieldwork._explore._kernels import exact_pair_ids
from fieldwork._explore.encoding import normalize_scalar


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
    text = render_plaintext(levels(pd.DataFrame({"d": durations})))
    shown = [line.strip().rsplit(": ", 1)[0] for line in text.splitlines()[3:]]
    assert shown == ["-0 days 00:00:05", "0 days 00:00:09", "0 days 00:00:10", "0 days 00:01:40"]


def test_exact_pair_fallback() -> None:
    import numpy as np

    parents = np.array([0, 4, 0, 4])
    levels_ = np.array([3, 2, 3, 2])
    ids, pairs_ = exact_pair_ids(parents, levels_, packing_limit=1)
    assert ids.tolist() == [0, 1, 0, 1]
    assert pairs_ == [(0, 3), (4, 2)]


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


def test_safe_renderer_bounds_and_controls() -> None:
    frame = pd.DataFrame({"a": ["\x1b]8;;bad\x07snowman ☃", "<NA>"]})
    text = render_plaintext(levels(frame), width=20, max_lines=4)
    assert len(text.splitlines()) <= 4
    assert all(len(line) <= 20 for line in text.splitlines())
    assert "\x1b" not in text
    assert "\\u2603" in text or "..." in text


def test_unsupported_values_fail() -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        normalize_scalar({"mutable": "mapping"})
