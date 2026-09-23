"""Row order never changes the evidence: only saved positions are relabeled."""

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import fieldwork as fw

IGNORED = {"source", "dataset_id", "selector", "id", "finding_id", "finding_ids"}


def canonical(result, order):
    """Map positions in a frame reordered by ``order`` back to original positions.

    Finding IDs follow emission order and lists of records are compared as
    multisets, so only evidence content is compared. A truncated example sample
    holds the first matches in source order, so only its totals are compared.
    """

    def walk(value, key=None):
        if isinstance(value, dict):
            truncated = value.get("omitted", 0) > 0 and "positions" in value
            return {
                k: walk(v, k)
                for k, v in value.items()
                if k not in IGNORED and not (truncated and k == "positions")
            }
        if isinstance(value, list):
            if key == "positions":
                return sorted(int(order[p]) for p in value)
            items = [walk(v) for v in value]
            if items and all(isinstance(v, dict) for v in items):
                return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, default=str))
            return items
        return value

    return walk(result.to_dict())


def assert_invariant(analysis, frame, order):
    identity = np.arange(len(frame))
    original = canonical(analysis(frame), identity)
    shuffled = canonical(analysis(frame.iloc[order].reset_index(drop=True)), order)
    assert shuffled == original


def rich_frame():
    rng = np.random.default_rng(7)
    n = 60
    entity = rng.integers(0, 12, n)
    site = np.array(["north", "south", "east"])[entity % 3]
    frame = pd.DataFrame(
        {
            "entity": entity,
            "site": site,
            "site_code": pd.Series(site).map({"north": 1, "south": 2, "east": 3}),
            "arm": rng.choice(["t", "c"], n),
            "visit": rng.integers(1, 4, n),
            # Ties within groups make the modal choice depend on its tie-break rule.
            "tied": rng.choice(["x", "y"], n),
            "sparse": np.where(rng.random(n) < 0.6, None, rng.choice(["p", "q"], n)),
            "twin": None,
            "measure": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n).round(1)),
        }
    )
    frame["twin"] = frame["sparse"].where(frame["sparse"].isna(), "set")
    frame.loc[5, "site_code"] = 9  # one exception to site -> site_code
    return frame


DEPENDENCY_FEATURES = ["entity", "site", "site_code", "tied", "sparse", "measure"]
ANALYSES = {
    "missingness": lambda df: fw.missingness(df, limits={"example_limit": len(df)}),
    "missingness_entities": lambda df: fw.missingness(
        df,
        entity="entity",
        unit="entities",
        by=["site"],
        limits={"example_limit": len(df)},
    ),
    "dependencies": lambda df: fw.discover_dependencies(
        df,
        features=DEPENDENCY_FEATURES,
        by=["arm"],
        min_accuracy=0,
        limits={"example_limit": len(df)},
    ),
    "dependencies_na": lambda df: fw.discover_dependencies(
        df,
        features=DEPENDENCY_FEATURES,
        dropna=False,
        min_accuracy=0,
        limits={"example_limit": len(df)},
    ),
    "paths": lambda df: fw.suggest_paths(df),
    "availability_paths": lambda df: fw.suggest_paths(df, objective="availability"),
    "target_paths": lambda df: fw.suggest_paths(df, objective="target", target="arm"),
}


@pytest.mark.parametrize("name", ANALYSES)
def test_findings_do_not_depend_on_row_order(name):
    frame = rich_frame()
    order = np.random.default_rng(0).permutation(len(frame))
    assert_invariant(ANALYSES[name], frame, order)


# Tiny domains make tied modes and tied path scores common.
CELLS = st.sampled_from([None, 0, 1, "x", "y"])


@settings(max_examples=30, deadline=None, derandomize=True)
@given(st.lists(st.tuples(CELLS, CELLS, CELLS), min_size=1, max_size=16), st.randoms())
def test_small_tied_frames_are_order_invariant(rows, random):
    frame = pd.DataFrame(rows, columns=["a", "b", "c"]).astype(object)
    order = np.array(random.sample(range(len(frame)), len(frame)))
    for analysis in (
        lambda df: fw.missingness(df, limits={"example_limit": len(df)}),
        lambda df: fw.discover_dependencies(df, min_accuracy=0, limits={"example_limit": len(df)}),
        lambda df: fw.discover_dependencies(df, dropna=False, limits={"example_limit": len(df)}),
        lambda df: fw.suggest_paths(df, objective="target", target="c"),
    ):
        assert_invariant(analysis, frame, order)
