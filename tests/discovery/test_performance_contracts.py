"""Semantic invariants for the optimized fingerprint and counting paths."""

from datetime import date

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fieldwork.evidence import fingerprint

CELLS = [None, True, 1, 1.0, "1", np.nan, -0.0, np.inf, date(2020, 1, 1), ("a", 1), ["x"], {"k": 1}]


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    st.lists(st.sampled_from(range(len(CELLS))), min_size=1, max_size=30),
    st.data(),
)
def test_fingerprint_detects_any_single_cell_change(indices, data):
    values = [CELLS[i] for i in indices]
    frame = pd.DataFrame({"v": pd.Series(values, dtype=object)})
    identity = fingerprint(frame)
    assert fingerprint(frame.copy()) == identity
    position = data.draw(st.integers(0, len(values) - 1))
    replacement = data.draw(
        st.sampled_from([c for c in range(len(CELLS)) if c != indices[position]])
    )
    changed = frame.copy()
    changed.iat[position, 0] = CELLS[replacement]
    same_repr = repr(CELLS[replacement]) == repr(values[position]) and type(
        CELLS[replacement]
    ) is type(values[position])
    assert (fingerprint(changed) == identity) == same_repr


def test_fingerprint_covers_order_labels_index_and_dtype():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]}, index=[0, 0, 1])
    identity = fingerprint(frame)
    variants = [
        frame.iloc[[1, 0, 2]],
        frame.rename(columns={"a": "c"}),
        frame.set_axis([0, 1, 1]),
        frame.astype({"a": "float64"}),
        frame[["b", "a"]],
    ]
    assert all(fingerprint(v) != identity for v in variants)
    multi = frame.set_axis(pd.MultiIndex.from_tuples([("x", 1.5), ("x", 2.5), ("y", 1.5)]))
    assert fingerprint(multi) == fingerprint(multi.copy()) != identity


def test_packed_signature_counts_and_ties_beyond_one_byte():
    from collections import Counter

    import fieldwork as fw

    rng = np.random.default_rng(23)
    present = rng.integers(0, 2, (90, 19)).astype(bool)
    present[40:50] = present[:10]
    frame = pd.DataFrame(np.where(present, 1.0, np.nan), columns=[f"c{i}" for i in range(19)])
    analysis = fw.missingness(frame, max_signatures=200)
    expected = sorted(Counter(map(tuple, present)).items(), key=lambda x: (-x[1], x[0]))
    for record, (mask, count) in zip(analysis["signatures"], expected):
        assert record["present"] == [c for c, populated in zip(frame.columns, mask) if populated]
        assert record["count"] == count
