"""discover_dependencies against an independent pandas groupby recomputation."""

from collections import Counter

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from oracle import dependency as expected_dependency
from oracle import determinants, grouping, record_token, token

import fieldwork as fw

FEATURES = ["a", "b", "c"]
# Small domains make repeated groups, exact rules and near-rules common. Integers
# and floats stay distinct, and every native missing spelling is one token.
CELLS = st.sampled_from([None, float("nan"), 0, 1, 2, 1.5, "x", "y", True])
CONTEXT = st.sampled_from([None, "p", "q"])
RECORD_FIELDS = [
    "exact",
    "modal_accuracy",
    "repair_rows",
    "evaluated_rows",
    "determinant_evaluated_rows",
    "target_observed_rows",
    "target_coverage",
    "target_missing_excluded_rows",
    "repeated_rows",
    "repeat_coverage",
    "repeat_modal_accuracy",
    "missing_excluded_rows",
    "evaluated_groups",
    "violating_groups",
    "affected_rows",
    "group_violation_rate",
    "repeated_groups",
]
CANDIDATE_FIELDS = [
    "evaluated_rows",
    "missing_excluded_rows",
    "groups",
    "unique",
    "uniqueness",
    "repeated_groups",
    "repeated_rows",
]


@st.composite
def frames(draw, min_rows=0, max_rows=24):
    length = draw(st.integers(min_rows, max_rows))
    row = st.tuples(CELLS, CELLS, CELLS, CONTEXT)
    rows = draw(st.lists(row, min_size=length, max_size=length))
    columns = [*FEATURES, "g"]
    return pd.DataFrame(
        {name: pd.Series([r[i] for r in rows], dtype=object) for i, name in enumerate(columns)}
    )


def approx(value):
    return value if value is None or isinstance(value, bool) else pytest.approx(value)


def context_populations(frame, by, rows):
    """Global population plus one joint context per observed value, missing included."""
    populations = {None: rows}
    keys = {i: tuple(token(frame.iat[i, frame.columns.get_loc(c)]) for c in by or []) for i in rows}
    for key in dict.fromkeys(keys.values()) if by else []:
        populations[tuple(zip(by, key))] = [i for i in rows if keys[i] == key]
    return populations


def context_key(record):
    context = record["context"]
    if context is None:
        return None
    return tuple((column, record_token(value)) for column, value in context.items())


def check_against_oracle(frame, *, dropna, by, positions=None, max_key_size=2):
    example_limit = len(frame) + 1
    scope = None if positions is None else fw.Scope.from_positions(frame, positions)
    result = fw.discover_dependencies(
        frame,
        features=FEATURES,
        by=by,
        dropna=dropna,
        scope=scope,
        max_key_size=max_key_size,
        min_accuracy=0,
        example_limit=example_limit,
        include_grain=False,
    )
    rows = list(range(len(frame))) if positions is None else sorted(positions)
    populations = context_populations(frame, by, rows)

    records = {
        (tuple(d["determinant"]), d["target"], context_key(d)): d for d in result["dependencies"]
    }
    expected_keys = {
        (key, target, context)
        for key in determinants(FEATURES, max_key_size)
        for target in FEATURES
        if target not in key
        for context in populations
    }
    assert records.keys() == expected_keys
    assert result["coverage"]["dependency_tests"] == len(expected_keys)

    oracle = {}
    for (key, target, context), record in records.items():
        expected = expected_dependency(frame, key, target, populations[context], dropna=dropna)
        oracle[key, target, context] = expected
        for field in RECORD_FIELDS:
            assert record[field] == approx(expected[field]), (key, target, context, field)
        violating = {
            group: (positions, counts)
            for group, (positions, counts) in expected["groups"].items()
            if len(counts) > 1
        }
        assert record["omitted_exception_groups"] == 0
        assert len(record["exception_groups"]) == len(violating)
        for exception in record["exception_groups"]:
            group = tuple(record_token(exception["key_values"][column]) for column in key)
            positions, counts = violating[group]
            assert exception["rows"] == len(positions)
            assert exception["distinct_targets"] == len(counts)
            assert exception["positions"] == positions

    # Every nonempty test is a finding when min_accuracy=0. Its exceptions are
    # exactly the repair rows: dropping them leaves each group with its modal value.
    findings = [f for f in result["findings"] if "determinant" in f["measurements"]]
    assert len(findings) == sum(1 for d in records.values() if d["evaluated_rows"])
    for finding in findings:
        measured = finding["measurements"]
        expected = oracle[tuple(measured["determinant"]), measured["target"], context_key(measured)]
        assert finding["pattern"] == (
            "exact_dependency" if expected["exact"] else "approximate_dependency"
        )
        assert finding["exceptions"]["total"] == expected["repair_rows"]
        repairs = set(finding["exceptions"]["positions"])
        conforming = set(finding["examples"]["positions"])
        assert finding["examples"]["total"] == len(conforming)
        evaluated = {i for positions, _ in expected["groups"].values() for i in positions}
        assert conforming | repairs == evaluated
        assert not conforming & repairs
        for positions, counts in expected["groups"].values():
            kept = Counter(
                token(frame.at[i, measured["target"]]) for i in positions if i not in repairs
            )
            assert len(kept) == 1
            assert sum(kept.values()) == max(counts.values())

    global_populations = populations[None]
    for candidate, key in zip(result["candidates"], determinants(FEATURES, max_key_size)):
        assert candidate["columns"] == list(key)
        expected = grouping(frame, key, global_populations, dropna=dropna)
        for field in CANDIDATE_FIELDS:
            assert candidate[field] == approx(expected[field]), (key, field)
        targets = [t for t in FEATURES if t not in key]
        exact = [t for t in targets if oracle[key, t, None]["exact"]]
        assert candidate["determines"] == exact
        assert candidate["determines_with_repeated_support"] == [
            t for t in exact if oracle[key, t, None]["repeated_groups"]
        ]
        assert candidate["global_targets_tested"] == len(targets)


@settings(max_examples=60, deadline=None, derandomize=True)
@given(frames(), st.booleans(), st.booleans())
def test_dependency_measurements_match_groupby_oracle(frame, dropna, with_context):
    check_against_oracle(frame, dropna=dropna, by=["g"] if with_context else None)


@settings(max_examples=25, deadline=None, derandomize=True)
@given(frames(min_rows=1), st.booleans(), st.data())
def test_scoped_dependencies_match_oracle_on_the_selected_rows(frame, dropna, data):
    positions = data.draw(st.sets(st.integers(0, len(frame) - 1)))
    check_against_oracle(frame, dropna=dropna, by=["g"], positions=sorted(positions))


def test_oracle_sees_composite_keys_and_near_rules():
    # A fixed frame keeps the property tests honest: it has an exact composite
    # key, a near-rule with one exception, and missing values in every role.
    frame = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2, None, 3],
            "b": ["x", "y", "x", "x", "y", "y", "x", None],
            "c": [10, 20, 10, 30, 40, 40, 50, 60],
            "g": ["p", "p", "q", "q", "p", None, "q", "p"],
        }
    ).astype(object)
    for dropna in (True, False):
        check_against_oracle(frame, dropna=dropna, by=["g"])
    result = fw.discover_dependencies(frame, features=FEATURES, include_grain=False)
    records = {(tuple(d["determinant"]), d["target"]): d for d in result["dependencies"]}
    assert records[("a", "b"), "c"]["exact"]
    assert records[("a",), "c"]["repair_rows"] == 2
