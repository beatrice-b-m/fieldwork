from __future__ import annotations

from collections import Counter

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fieldwork import census, levels

from .oracle import level_counts, prefix_counts

VALUES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-3, 12),
    st.floats(-3, 3, allow_nan=False, allow_infinity=False),
    st.sampled_from(["1", "x", ""]),
)


def _record_token(record):
    return (record["type"], record.get("value"))


@settings(max_examples=40, deadline=None, derandomize=True)
@given(st.lists(st.tuples(VALUES, VALUES), max_size=20))
def test_generated_counts_match_independent_oracle(rows) -> None:
    frame = pd.DataFrame(rows, columns=["a", "b"])
    independent = levels(frame, max_levels=None)
    for column_index, column in enumerate(frame.columns):
        actual = Counter(
            {
                _record_token(item["value"]): item["count"]
                for item in independent["per_feature"][column_index]["levels"]
            }
        )
        assert actual == level_counts(frame, column)

    nested = census(frame, ["a", "b"], max_levels=None, max_nodes=None)
    dictionary = {
        item["level_id"]: _record_token(item["value"]) for item in nested["level_dictionary"]
    }
    nodes = {item["node_id"]: item for item in nested["tree"]["nodes"]}
    actual_prefixes = [Counter(), Counter()]
    for node in nodes.values():
        path = []
        cursor = node
        while cursor["parent_id"] != "root":
            path.append(dictionary[cursor["level_id"]])
            cursor = nodes[cursor["parent_id"]]
        path.append(dictionary[cursor["level_id"]])
        actual_prefixes[node["depth"] - 1][tuple(reversed(path))] = node["count"]
    assert actual_prefixes == prefix_counts(frame, ["a", "b"])
