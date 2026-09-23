from __future__ import annotations

from collections import Counter

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from oracle import MISSING, level_counts, prefix_counts, record_token

from fieldwork import census, levels

# Every native missing spelling must collapse to one level; integers, floats,
# booleans and strings that print alike must not.
VALUES = st.one_of(
    st.sampled_from([None, float("nan"), pd.NA, pd.NaT]),
    st.booleans(),
    st.integers(-3, 12),
    st.floats(-3, 3),
    st.sampled_from(["1", "x", "", "nan", "None"]),
)


@st.composite
def frames(draw):
    width, length = draw(st.integers(1, 4)), draw(st.integers(0, 40))
    rows = draw(st.lists(st.tuples(*[VALUES] * width), min_size=length, max_size=length))
    return pd.DataFrame(rows, columns=[f"c{i}" for i in range(width)])


@settings(max_examples=60, deadline=None, derandomize=True)
@given(frames())
def test_generated_counts_match_independent_oracle(frame) -> None:
    columns = list(frame.columns)
    independent = levels(frame, max_levels=None)
    for column_index, column in enumerate(columns):
        actual = Counter(
            {
                record_token(item["value"]): item["count"]
                for item in independent["per_feature"][column_index]["levels"]
            }
        )
        assert actual == level_counts(frame, column)

    nested = census(frame, columns, max_levels=None, max_nodes=None)
    nodes = {item["node_id"]: item for item in nested["tree"]["nodes"]}
    actual_prefixes = [Counter() for _ in columns]
    for node in nodes.values():
        path = []
        cursor = node
        while cursor["parent_id"] != "root":
            path.append(record_token(cursor["value"]))
            cursor = nodes[cursor["parent_id"]]
        path.append(record_token(cursor["value"]))
        actual_prefixes[node["depth"] - 1][tuple(reversed(path))] = node["count"]
    assert actual_prefixes == prefix_counts(frame, columns)

    dropped = census(frame, columns, max_levels=None, max_nodes=None, dropna=True)
    complete = sum(all(token != MISSING for token in path) for path in _paths(frame, columns))
    assert dropped["tree"]["root"]["count"] == complete
    assert dropped["tree"]["missing_excluded_rows"] == len(frame) - complete


def _paths(frame, columns):
    return prefix_counts(frame, columns)[-1].elements()
