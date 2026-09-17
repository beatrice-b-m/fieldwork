from __future__ import annotations

import pandas as pd

from fieldwork import census, levels


def test_levels_are_independent() -> None:
    frame = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 2]})
    result = levels(frame, ["a", "b"], top_n=1)
    assert [item["reported_rows"] for item in result["per_feature"]] == [2, 2]
    assert [item["unreported_rows"] for item in result["per_feature"]] == [1, 1]


def test_post_census_conserves_each_expanded_parent() -> None:
    frame = pd.DataFrame({"a": ["x", "x", "x", "y", "y"], "b": [1, 1, 2, 1, 2]})
    result = census(frame, ["a", "b"], top_n=1, top_n_per_parent=True)
    nodes = {item["node_id"]: item for item in result["tree"]["nodes"]}
    nodes["root"] = result["tree"]["root"]
    for parent in nodes.values():
        if parent["expansion_state"] != "expanded":
            continue
        children = [item for item in nodes.values() if item.get("parent_id") == parent["node_id"]]
        assert (
            parent["count"]
            == sum(item["count"] for item in children) + parent["omitted_child_rows"]
        )


def test_pre_census_recomputes_common_population() -> None:
    frame = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 2]})
    result = census(frame, ["a", "b"], top_n=1, top_n_mode="pre")
    assert result["scopes"][0]["evaluated_rows"] == 1
    assert result["tree"]["root"]["count"] == 1
    assert all(node["count"] == 1 for node in result["tree"]["nodes"])


def test_degenerate_pre_is_distinct_from_empty_input() -> None:
    empty = census(pd.DataFrame({"a": []}), ["a"], top_n=1, top_n_mode="pre")
    assert empty["status"] == "empty"
    frame = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
    # Ties canonically choose x and u, which still intersect here.
    assert census(frame, ["a", "b"], top_n=1, top_n_mode="pre")["tree"]["root"]["count"] == 1


def test_row_permutation_is_canonical() -> None:
    frame = pd.DataFrame({"a": ["y", "x", "x"], "b": [2, 2, 1]})
    one = census(frame, ["a", "b"]).to_dict()
    two = census(frame.sample(frac=1, random_state=2), ["a", "b"]).to_dict()
    assert one == two
