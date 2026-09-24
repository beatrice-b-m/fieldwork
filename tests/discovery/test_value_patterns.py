"""value_patterns measurements against known answers and a string-shape oracle."""

import json
import string
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import fieldwork as fw


def findings(result, pattern):
    return {
        tuple(c["column"] for c in f["features"]): f
        for f in result["findings"]
        if f["pattern"] == pattern
    }


def shape(text):
    """Digit runs become 9 and ASCII letter runs become A; other characters stay."""
    output = []
    for character in text:
        kind = (
            "9"
            if character.isdecimal()
            else "A"
            if character in string.ascii_letters
            else character
        )
        if kind in "9A" and output and output[-1] == kind:
            continue
        output.append(kind)
    return "".join(output)


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    st.lists(st.one_of(st.none(), st.text("aZ09-_ é٣", max_size=6)), min_size=1, max_size=30),
    st.integers(1, 4),
    st.integers(0, 3),
)
def test_string_formats_lengths_and_prefixes_match_oracle(values, max_patterns, min_count):
    frame = pd.DataFrame({"code": pd.Series(values, dtype=object)})
    populated = [v for v in values if v is not None]
    result = fw.value_patterns(frame, limits={"max_patterns": max_patterns}, min_count=min_count)
    record = findings(result, "string_patterns").get(("code",))
    if not populated:
        assert record is None
        return
    m = record["measurements"]
    assert (m["populated"], m["missing"]) == (len(populated), len(values) - len(populated))
    expected = {
        "formats": Counter(shape(v) for v in populated),
        "lengths": Counter(len(v) for v in populated),
        "prefixes": Counter(v[:3] for v in populated),
    }
    for field, counts in expected.items():
        shown = dict(m[field])
        eligible = [n for n in counts.values() if n >= min_count]
        assert len(shown) == min(max_patterns, len(eligible)), field
        assert all(counts[value] == n >= min_count for value, n in shown.items()), field
        # Shown patterns are the most frequent ones.
        ranked = [n for _, n in m[field]]
        assert ranked == sorted(ranked, reverse=True)
        assert min(ranked, default=min_count) >= max(
            (n for value, n in counts.items() if value not in shown and n >= min_count),
            default=0,
        )
    assert m["format_count"] == len(expected["formats"])
    assert m["omitted_format_rows"] == len(populated) - sum(n for _, n in m["formats"])
    # Topology keeps the reported formats, canonically ordered and without counts.
    structure = fw.visualization_data(result, detail="topology")["findings"][0]["structure"]
    assert structure == {
        "formats": sorted(dict(m["formats"])),
        "formats_omitted": len(m["formats"]) < len(expected["formats"]),
    }


def test_string_patterns_known_answer():
    values = ["AB-12", "AB-13", "CD-7", "x9", "AB-15", None, "ÅB1", "  "]
    record = findings(fw.value_patterns(pd.DataFrame({"id": values})), "string_patterns")[("id",)]
    m = record["measurements"]
    assert dict(m["formats"]) == {"A-9": 4, "A9": 1, "ÅA9": 1, "  ": 1}
    assert dict(m["lengths"]) == {5: 3, 4: 1, 2: 2, 3: 1}
    assert dict(m["prefixes"]) == {"AB-": 3, "CD-": 1, "x9": 1, "ÅB1": 1, "  ": 1}
    assert record["examples"]["total"] == 7
    # Mixed objects and non-string dtypes have no string-format summary.
    mixed = pd.DataFrame({"m": pd.Series(["a", 1], dtype=object), "b": [True, False]})
    assert not findings(fw.value_patterns(mixed), "string_patterns")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            [0, 2.5, 5, 10, None, np.inf],
            {"minimum": 0.0, "maximum": 10.0, "nonfinite": 1, "observed_step": 2.5, "grid": True},
        ),
        (
            [0.0, 1.0, 2.5],
            {"minimum": 0.0, "maximum": 2.5, "nonfinite": 0, "observed_step": 1.0, "grid": False},
        ),
        (
            [3, 7, 11, 11],
            {"minimum": 3.0, "maximum": 11.0, "nonfinite": 0, "observed_step": 4.0, "grid": True},
        ),
        (
            [4, 4, None],
            {"minimum": 4.0, "maximum": 4.0, "nonfinite": 0, "observed_step": None, "grid": None},
        ),
        (
            [np.inf, -np.inf],
            {"minimum": None, "maximum": None, "nonfinite": 2, "observed_step": None, "grid": None},
        ),
    ],
)
def test_numeric_range_known_answers(values, expected):
    frame = pd.DataFrame({"x": pd.Series(values, dtype="float64")})
    m = findings(fw.value_patterns(frame), "numeric_range")[("x",)]["measurements"]
    assert m["populated"] == sum(v is not None for v in values)
    assert {
        "minimum": m["minimum"],
        "maximum": m["maximum"],
        "nonfinite": m["nonfinite"],
        "observed_step": m["observed_step"],
        "grid": m["on_observed_step_grid"],
    } == expected
    assert not findings(fw.value_patterns(pd.DataFrame({"b": [True, False]})), "numeric_range")


def test_numeric_offset_and_ratio_use_only_valid_rows():
    x = [0.0, 1.0, 2.0, 3.0, None, 5.0]
    frame = pd.DataFrame(
        {
            "x": x,
            "shifted": [v + 3 if v is not None else 9.0 for v in x],
            "doubled": [v * 2 if v is not None else 1.0 for v in x],
            "noise": [0.0, 7.0, 1.0, 4.0, 2.0, 9.0],
        }
    )
    result = fw.value_patterns(frame)
    offset = findings(result, "numeric_offset")
    ratio = findings(result, "numeric_ratio")
    m = offset[("x", "shifted")]["measurements"]
    assert (m["value"], m["evaluated_rows"], m["excluded_rows"]) == (3.0, 5, 1)
    # The zero denominator in x is excluded as well as the missing row.
    assert ratio[("x", "doubled")]["measurements"]["value"] == 2.0
    assert ratio[("x", "doubled")]["measurements"]["evaluated_rows"] == 4
    assert ratio[("x", "doubled")]["examples"]["positions"] == [1, 2, 3, 5]
    assert not any("noise" in pair for pair in (*offset, *ratio))
    assert result["coverage"] == {"pair_candidates": 6, "pairs_evaluated": 6}
    # Pairs are tested in column-combination order: only (x, shifted) fits one test.
    limited = fw.value_patterns(frame, limits={"max_pairs": 1})
    assert limited["coverage"]["pairs_evaluated"] == 1
    assert set(findings(limited, "numeric_offset")) == {("x", "shifted")}
    assert not findings(limited, "numeric_ratio")


def test_indexed_families_group_names_and_check_availability():
    frame = pd.DataFrame(
        {
            "score_1": [1, 2, None],
            "score_2": [3, 4, None],
            "score-10": [5, 6, None],
            "visit1": [1, None, 2],
            "visit2": [1, 2, 3],
            "solo_7": [1, 1, 1],
            "name": ["a", "b", "c"],
        }
    )
    result = fw.value_patterns(frame)
    assert {tuple(f["features"]): f["evidence"] for f in result["families"]} == {
        ("score_1", "score_2", "score-10"): ["indexed_name", "identical_availability"],
        ("visit1", "visit2"): ["indexed_name"],
    }
    assert set(findings(result, "indexed_family")) == {
        ("score_1", "score_2", "score-10"),
        ("visit1", "visit2"),
    }


def test_context_constancy_counts_populated_groups():
    frame = pd.DataFrame(
        {
            "site": ["A", "A", "B", "B", "C", None],
            "arm": ["t", "t", "t", "c", "c", "t"],
            "value": [1, 1, 2, 3, None, 4],
        }
    )
    single = findings(fw.value_patterns(frame, by=["site"]), "context_constancy")
    value = single[("site", "value")]
    # Rows with a missing context are excluded; C has no populated value.
    assert value["measurements"] == {
        "evaluated_groups": 2,
        "constant_groups": 1,
        "constant_group_fraction": 0.5,
    }
    assert value["examples"]["positions"] == [0, 1]
    assert value["exceptions"]["positions"] == [2, 3]
    arm = single[("site", "arm")]["measurements"]
    assert (arm["evaluated_groups"], arm["constant_groups"]) == (3, 2)
    joint = findings(fw.value_patterns(frame, by=["site", "arm"]), "context_constancy")
    m = joint[("site", "arm", "value")]["measurements"]
    # Joint groups (A,t), (B,t), (B,c) each hold one value; (C,c) holds none.
    assert (m["evaluated_groups"], m["constant_groups"]) == (3, 3)


def test_live_string_patterns_match_their_saved_export():
    # Regression: counts were tuples in memory and lists once saved, so a live
    # result rendered differently from its own export.
    result = fw.value_patterns(pd.DataFrame({"id": ["AB-1", "AB-2", "x"]}))
    saved = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert result.to_dict() == saved
    for render in (fw.render_plaintext, fw.render_svg, fw.render_html):
        assert render(result) == render(saved)


def test_numeric_summaries_follow_values_not_dtype():
    numbers = [1, 2.5, None, 4]
    native = pd.DataFrame({"x": numbers, "y": [v + 1 if v is not None else None for v in numbers]})
    stored = native.astype(object)
    for frame in (native, stored):
        result = fw.value_patterns(frame)
        assert findings(result, "numeric_range")[("x",)]["measurements"]["minimum"] == 1.0
        assert findings(result, "numeric_offset")[("x", "y")]["measurements"]["value"] == 1.0
    # One string among the values makes the column non-numeric.
    mixed = pd.DataFrame({"x": [1, "2", 3]})
    assert not findings(fw.value_patterns(mixed), "numeric_range")


def test_context_constancy_skips_the_context_columns():
    frame = pd.DataFrame({"site": ["A", "A", "B"], "arm": ["t", "c", "t"], "v": [1, 1, 2]})
    result = fw.value_patterns(frame, by=["site", "arm"])
    assert set(findings(result, "context_constancy")) == {("site", "arm", "v")}
