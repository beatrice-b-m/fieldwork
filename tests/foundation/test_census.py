from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from oracle import MISSING, column_tokens, record_token

from fieldwork import census, levels


def test_levels_are_independent() -> None:
    frame = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 2]})
    result = levels(frame, ["a", "b"], top_n=1)
    assert [item["reported_rows"] for item in result["per_feature"]] == [2, 2]
    assert [item["unreported_rows"] for item in result["per_feature"]] == [1, 1]


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


DIMENSIONS = ["a", "b", "c"]
CELLS = st.sampled_from([None, 1, 2, 3, "x", "y"])


@st.composite
def census_cases(draw):
    length = draw(st.integers(1, 30))
    rows = draw(st.lists(st.tuples(CELLS, CELLS, CELLS), min_size=length, max_size=length))
    options = {
        "top_n": draw(st.none() | st.integers(1, 3)),
        "top_n_mode": draw(st.sampled_from(["pre", "post"])),
        "top_n_per_parent": draw(st.booleans()),
        "max_levels": draw(st.none() | st.integers(0, 3)),
        "min_count": draw(st.integers(0, 3)),
        "max_depth": draw(st.none() | st.integers(1, 3)),
        "max_nodes": draw(st.none() | st.integers(0, 12)),
        "dropna": draw(st.booleans()),
    }
    return pd.DataFrame(rows, columns=DIMENSIONS).astype(object), options


def _threshold(counts: Counter, top_n: int) -> int:
    """Count of the top_n-th most frequent level: anything above it is surely kept."""
    ranked = sorted(counts.values(), reverse=True)
    return ranked[min(top_n, len(ranked)) - 1]


@settings(max_examples=120, deadline=None, derandomize=True)
@given(census_cases())
def test_census_options_bound_output_without_changing_counts(case) -> None:
    frame, options = case
    active = DIMENSIONS[: options["max_depth"] or len(DIMENSIONS)]
    tokens = [tuple(row) for row in zip(*(column_tokens(frame, c) for c in active))]
    eligible = [i for i, path in enumerate(tokens) if not options["dropna"] or MISSING not in path]
    top_n, pre = options["top_n"], options["top_n_mode"] == "pre"
    try:
        result = census(frame, DIMENSIONS, **options)
    except ValueError as error:
        # Pre-selection may keep no row; then no row can have only surely-kept levels.
        assert pre and top_n is not None and "DEGENERATE_TOP_N" in str(error)
        assert not options["top_n_per_parent"]
        counts = [Counter(tokens[i][d] for i in eligible) for d in range(len(active))]
        sure = [{v for v, n in c.items() if n > _threshold(c, top_n)} for c in counts]
        assert not any(all(tokens[i][d] in sure[d] for d in range(len(active))) for i in eligible)
        return

    root = result["tree"]["root"]
    nodes = {n["node_id"]: n for n in result["tree"]["nodes"]}
    paths = {"root": ()}
    for node in result["tree"]["nodes"]:  # parents precede children
        paths[node["node_id"]] = (*paths[node["parent_id"]], record_token(node["value"]))

    # The evaluated population: complete cases (dropna), then any pre-selection.
    if pre and top_n is not None:
        retained = result["tree"]["retained_sets"]
        if options["top_n_per_parent"]:
            chosen = {
                (tuple(record_token(v) for v in r["path"]), record_token(value))
                for r in retained
                for value in r["values"]
            }
            evaluated = [
                i
                for i in eligible
                if all((tokens[i][:d], tokens[i][d]) in chosen for d in range(len(active)))
            ]
            for record in retained:
                prefix = tuple(record_token(v) for v in record["path"])
                depth = record["depth"] - 1
                siblings = Counter(
                    tokens[i][depth] for i in eligible if tokens[i][:depth] == prefix
                )
                kept = {record_token(v) for v in record["values"]}
                assert len(kept) == min(top_n, len(siblings))
                assert min(siblings[v] for v in kept) >= max(
                    (n for v, n in siblings.items() if v not in kept), default=0
                )
        else:
            kept = [{record_token(v) for v in r["values"]} for r in retained]
            for depth, levels_kept in enumerate(kept):
                counts = Counter(tokens[i][depth] for i in eligible)
                assert len(levels_kept) == min(top_n, len(counts))
                assert min(counts[v] for v in levels_kept) >= max(
                    (n for v, n in counts.items() if v not in levels_kept), default=0
                )
            evaluated = [
                i for i in eligible if all(tokens[i][d] in kept[d] for d in range(len(active)))
            ]
    else:
        evaluated = eligible
    assert root["count"] == len(evaluated) == result["scopes"][0]["evaluated_rows"]
    assert result["scopes"][0]["missing_excluded_rows"] == len(frame) - len(eligible)
    assert result["scopes"][0]["restriction_excluded_rows"] == len(eligible) - len(evaluated)

    prefixes = Counter(tokens[i][:d] for i in evaluated for d in range(1, len(active) + 1))
    children = {"root": []} | {node_id: [] for node_id in nodes}
    for node in nodes.values():
        children[node["parent_id"]].append(node)
    assert len(result["features"]) == len(active)
    if options["max_nodes"] is not None:
        assert len(nodes) <= options["max_nodes"]
    for node_id, parent in [("root", root), *nodes.items()]:
        path = paths[node_id]
        if node_id != "root":
            # Exact counts on the evaluated population, whatever the display limits.
            assert parent["count"] == prefixes[path] >= options["min_count"]
            assert parent["share_of_total"] == pytest.approx(parent["count"] / root["count"])
        if parent["expansion_state"] != "expanded":
            continue
        emitted = children[node_id]
        siblings = Counter({p[-1]: n for p, n in prefixes.items() if p[:-1] == path})
        assert parent["count"] == sum(c["count"] for c in emitted) + parent["omitted_child_rows"]
        assert parent["omitted_child_levels"] == len(siblings) - len(emitted)
        if options["max_levels"] is not None:
            assert len(emitted) <= options["max_levels"]
        shown = {paths[c["node_id"]][-1] for c in emitted}
        global_post = top_n is not None and not pre and not options["top_n_per_parent"]
        if top_n is not None and not pre and options["top_n_per_parent"]:
            assert len(emitted) <= top_n
        if not global_post and options["max_nodes"] is None:
            # Without a node budget the number shown is fully determined.
            ranked = sorted(siblings.values(), reverse=True)
            if top_n is not None and not pre:
                ranked = ranked[:top_n]
            expected = len([n for n in ranked if n >= options["min_count"]])
            if options["max_levels"] is not None:
                expected = min(expected, options["max_levels"])
            assert len(emitted) == expected
        if not global_post:
            # Children are the most frequent siblings; limits only cut the tail.
            assert min((c["count"] for c in emitted), default=len(evaluated)) >= max(
                (n for v, n in siblings.items() if v not in shown), default=0
            )
    if top_n is not None and not pre and not options["top_n_per_parent"]:
        for depth in range(len(active)):
            counts = Counter(tokens[i][depth] for i in evaluated)
            shown = {path[depth] for path in paths.values() if len(path) > depth}
            assert len(shown) <= top_n
            assert all(counts[v] >= _threshold(counts, top_n) for v in shown)
