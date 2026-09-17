"""Semantic invariants for the optimized preparation/counting paths."""

import hashlib
import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fieldwork._explore.encoding import encode_series, normalize_scalar
from fieldwork.evidence import fingerprint


def original_fingerprint(df):
    digest = hashlib.sha256()
    for values in (df.columns, df.index, *[df[c].array for c in df]):
        digest.update(b"[")
        for value in values:
            digest.update(
                json.dumps(
                    normalize_scalar(value, label=isinstance(value, tuple)).to_dict(),
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            )
            digest.update(b"\n")
        digest.update(b"]")
    return digest.hexdigest()


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    st.lists(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(-10, 10),
            st.floats(allow_nan=True, allow_infinity=True),
            st.text(max_size=12),
        ),
        max_size=50,
    )
)
def test_fingerprint_preserves_typed_ordered_byte_stream(values):
    frame = pd.DataFrame({"v": pd.Series(values, dtype=object)})
    frame.index = [0] * len(values)
    assert fingerprint(frame) == original_fingerprint(frame)
    tokens, codes = encode_series(pd.Series(values, dtype=object))
    assert [tokens[c] for c in codes] == [normalize_scalar(v) for v in values]


@pytest.mark.parametrize(
    "values",
    [
        [None, pd.NA, pd.NaT, np.nan],
        [0.0, -0.0, np.inf, -np.inf],
        [date(2020, 1, 1), None],
        [datetime(2020, 1, 1, tzinfo=UTC).replace(tzinfo=None), None],
        [datetime(2020, 1, 1, tzinfo=UTC), None],
        [timedelta(days=1), pd.Timedelta("1ns"), None],
        [True, 1, 1.0, "1"],
        [("a", 1), ("b", 2)],
    ],
)
def test_fingerprint_labels_and_scalar_families(values):
    frame = pd.DataFrame({("typed", 2): pd.Series(values, dtype=object)})
    frame.index = pd.MultiIndex.from_tuples([("row", i % 2) for i in range(len(frame))])
    assert fingerprint(frame) == original_fingerprint(frame)


def test_fingerprint_chunk_boundary_and_mutation():
    frame = pd.DataFrame({"v": np.tile([1.0, np.nan, 2.0], 3000)})
    before = fingerprint(frame)
    assert before == original_fingerprint(frame)
    frame.iloc[8192, 0] = 4
    assert fingerprint(frame) == original_fingerprint(frame) != before


def test_presence_only_preparation_avoids_unused_value_codes(monkeypatch):
    from fieldwork import _runtime, evidence

    frame = pd.DataFrame({"selected": [1, 2], "present_only": [3.0, np.nan], "unused": [4, 5]})
    calls = []
    actual = evidence.encode_series

    def encode(series):
        calls.append(series.name)
        return actual(series)

    monkeypatch.setattr(evidence, "_fingerprint", lambda *args: "identity")
    monkeypatch.setattr(evidence, "encode_series", encode)

    @_runtime.operation("test")
    def run():
        _, _, codes, available, _ = evidence.prepare(
            frame, features=["selected"], presence_features=["present_only"]
        )
        assert list(codes) == ["selected"]
        assert available["present_only"].tolist() == [True, False]
        evidence.prepare(frame, features=["selected"], presence_features=["present_only"])
        assert calls == ["selected"]
        evidence.prepare(frame, features=["present_only"])
        assert calls == ["selected", "present_only"]

    run()
    assert _runtime.current_session() is None


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
