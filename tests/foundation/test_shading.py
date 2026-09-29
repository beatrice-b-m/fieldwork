"""Shading ranges of association and joint-count figures: rules, fixed cut-offs, errors."""

from __future__ import annotations

import pandas as pd
import pytest

from fieldwork import joint_counts, pairs, render_html, render_svg, visualization_data


def counts_result():
    # Cell counts 1, 2, 4, 8, 16, 32, 64, 128 over a 2 x 4 grid.
    rows = []
    for index, count in enumerate([1, 2, 4, 8, 16, 32, 64, 128]):
        rows += [{"a": index // 4, "b": index % 4}] * count
    return joint_counts(pd.DataFrame(rows), ["a", "b"])


def pairs_result():
    frame = pd.DataFrame(
        {
            "x": list("aabbccdd") * 3,
            "y": list("aabbccdd") * 2 + list("abababab"),
            "z": list("abcdabcd") * 3,
        }
    )
    return pairs(frame, ["x", "y", "z"])


def ranges(values, cuts):
    return [sum(v >= c for c in cuts) for v in values]


def test_default_count_cutoffs_are_increasing_whole_counts_on_a_log_scale():
    shading = visualization_data(counts_result())["shading"]
    cuts = shading["counts"]
    assert shading["counts_rule"] == "log"
    assert all(float(c).is_integer() for c in cuts)
    assert cuts[0] < cuts[1] < cuts[2] <= 128
    # Evenly spaced in log from 1 to 128 (about 3.4, 11 and 38), then rounded.
    assert cuts == [3.0, 10.0, 40.0]
    assert ranges([1, 2, 4, 8, 16, 32, 64, 128], cuts) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_quantile_puts_a_quarter_of_the_cells_in_each_range():
    projection = visualization_data(counts_result(), shading={"counts": "quantile"})
    cuts = projection["shading"]["counts"]
    counts = [cell["count"] for cell in projection["cells"]]
    assert sorted(ranges(counts, cuts)) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_default_association_cutoffs_split_the_observed_range_equally():
    projection = visualization_data(pairs_result())
    values = [c["association"] for ctx in projection["contexts"] for c in ctx["cells"]]
    low, high = min(values), max(values)
    expected = [round(low + (high - low) * k / 4, 2) for k in (1, 2, 3)]
    assert projection["shading"] == {"association": expected, "association_rule": "equal"}


def test_fixed_cutoffs_give_every_report_the_same_scale():
    fixed = {"association": (0.1, 0.3, 0.5), "counts": (2, 5, 10)}
    for result in (pairs_result(), counts_result()):
        shading = visualization_data(result, shading=fixed)["shading"]
        key = "association" if "association" in shading else "counts"
        assert shading[key] == [float(v) for v in fixed[key]]
        assert shading[f"{key}_rule"] == "fixed"
    assert "≥ 0.50" in render_svg(pairs_result(), view="association", shading=fixed)
    assert render_html(counts_result(), shading=fixed).startswith("<!doctype html>")


def test_topology_projections_have_no_cutoffs():
    for result in (pairs_result(), counts_result()):
        assert "shading" not in visualization_data(result, detail="topology")


@pytest.mark.parametrize(
    "shading",
    [
        {"counts": "linear"},
        {"counts": (5, 2, 10)},
        {"counts": (1, 2)},
        {"counts": (1, 2, float("nan"))},
        {"association": (True, 0.3, 0.5)},
        {"colour": "log"},
    ],
)
def test_invalid_shading_rules_are_rejected(shading):
    with pytest.raises(ValueError, match="shading"):
        visualization_data(counts_result(), shading=shading)


def test_shading_must_be_a_mapping():
    with pytest.raises(TypeError, match="shading"):
        render_svg(counts_result(), shading="log")  # type: ignore[arg-type]
