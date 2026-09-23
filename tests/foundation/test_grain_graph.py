from __future__ import annotations

import json

import pandas as pd

from fieldwork import KeySpec, explore, grain


def _nodes(graph):
    return {name: node["id"] for node in graph["nodes"] for name in node["keys"]}


def _edges(graph):
    return {(edge["source"], edge["target"]) for edge in graph["edges"]}


def test_chain_equivalence_and_transitive_reduction():
    frame = pd.DataFrame(
        {
            "patient": [1, 1, 1, 1, 2, 2],
            "exam": [1, 1, 2, 2, 3, 3],
            "exam_alias": ["a", "a", "b", "b", "c", "c"],
            "image": range(6),
            "birth_year": [1980, 1980, 1980, 1980, 1990, 1990],
            "date": ["Mon", "Mon", "Tue", "Tue", "Wed", "Wed"],
        }
    )
    graph = grain(frame, ["patient", "exam", "exam_alias", "image"])["graph"]
    nodes = _nodes(graph)
    assert nodes["exam"] == nodes["exam_alias"]
    assert _edges(graph) == {(nodes["patient"], nodes["exam"]), (nodes["exam"], nodes["image"])}
    assignments = {a["target"]: a["nodes"] for a in graph["assignments"]}
    assert assignments["birth_year"] == [nodes["patient"]]
    assert assignments["date"] == [nodes["exam"]]
    assert any(
        r == {"determinant": "image", "dependent": "patient", "holds": True}
        for r in graph["key_relationships"]
    )
    json.dumps(graph, allow_nan=False)


def test_composite_example_support_and_unplaced():
    frame = pd.DataFrame(
        {
            "exam": [1, 1, 2, 2, 3, 3],
            "side": ["L", "R", "L", "R", "L", "L"],
            "finding": ["a", "b", "c", "c", "d", "d"],
            "noise": range(6),
        }
    )
    graph = grain(frame, ["exam", KeySpec("exam_side", ("exam", "side"))])["graph"]
    nodes = _nodes(graph)
    assert _edges(graph) == {(nodes["exam"], nodes["exam_side"])}
    evidence = [r for r in graph["tests"] if r["target"] == "finding"]
    assert [
        (
            r["holds"],
            r["evaluated_groups"],
            r["violating_groups"],
            r["singleton_groups"],
            r["repeated_groups"],
        )
        for r in evidence
    ] == [(False, 3, 1, 0, 3), (True, 5, 0, 4, 1)]
    assert graph["unplaced"] == [{"target": "noise", "reason": "no_supported_key"}]


def test_diamond_and_shared_assignment():
    frame = pd.DataFrame(
        {
            "patient": [1, 1, 2, 2],
            "scanner": ["x", "y", "x", "y"],
            "image": range(4),
            "constant": ["yes"] * 4,
        }
    )
    graph = grain(frame, ["patient", "scanner", "image"])["graph"]
    nodes = _nodes(graph)
    assert _edges(graph) == {(nodes["patient"], nodes["image"]), (nodes["scanner"], nodes["image"])}
    assert graph["assignments"][-1]["nodes"] == [nodes["patient"], nodes["scanner"]]


def test_missing_targets_do_not_merge_global_nodes():
    frame = pd.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4], "target": [1, None, 2, None]})
    graph = grain(frame, ["a", "b"], dropna=True)["graph"]
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
    assert graph["unplaced"][0]["reason"] == "different_target_population"
    evidence = [r for r in graph["tests"] if r["target"] == "target"]
    assert all(r["holds"] and not r["compatible"] for r in evidence)
    assert all(r["evaluated_rows"] == 2 for r in evidence)
    assert graph["evaluated_rows"] == 4


def test_common_key_missingness_and_empty_are_explicit():
    frame = pd.DataFrame({"a": [1, 2, None], "b": [1, None, 2], "target": ["x"] * 3})
    graph = grain(frame, ["a", "b"], dropna=True)["graph"]
    assert graph["evaluated_rows"] == 1
    assert len(graph["nodes"]) == 1
    empty = grain(frame.iloc[:0], ["a", "b"])["graph"]
    assert len(empty["nodes"]) == 2
    assert empty["edges"] == []
    assert all(r["holds"] is None for r in empty["key_relationships"])
    assert empty["unplaced"]


def test_graph_scope_is_the_census_cohort():
    frame = pd.DataFrame({"a": ["x", "x", "y"], "key": [1, 2, 3], "value": [1, 2, 3]})
    section = explore(
        frame, ["a"], candidate_keys=["key"], top_n=1, top_n_mode="pre", top_n_applies_to="both"
    )["sections"]["grain"]
    # The grain graph evaluates only the census pre-selection's rows.
    assert section["scope"]["name"] == "census top_n cohort"
    assert section["scope"]["input_rows"] == 3
    assert section["scope"]["restriction_excluded_rows"] == 1
    assert section["graph"]["evaluated_rows"] == 2


def test_key_refinement_matches_row_partition_oracle():
    import numpy as np

    rng = np.random.default_rng(739)
    for _ in range(12):
        frame = pd.DataFrame({c: rng.integers(0, 3, size=15) for c in "abc"})
        keys = [KeySpec("a", ("a",)), KeySpec("b", ("b",)), KeySpec("ab", ("a", "b"))]
        graph = grain(frame, keys)["graph"]
        specs = {key.name: key.columns for key in keys}
        for relation in graph["key_relationships"]:
            groups = {}
            for row in frame.to_dict("records"):
                left = tuple(row[c] for c in specs[relation["determinant"]])
                right = tuple(row[c] for c in specs[relation["dependent"]])
                groups.setdefault(left, set()).add(right)
            assert relation["holds"] == all(len(values) == 1 for values in groups.values())
