from __future__ import annotations

import json

import pandas as pd
import pytest

from fieldwork import Recipe, joint_counts, render_plaintext, render_svg, visualization_data


def test_joint_counts_context_scope_and_typed_values():
    frame = pd.DataFrame(
        {
            "a": pd.Series([True, 1, "1", None, 1], dtype=object),
            "b": ["x", "x", "x", "y", "x"],
            "site": ["N", "N", "S", "N", None],
        }
    )
    full = joint_counts(frame, ["a", "b"])
    assert sum(c["count"] for c in full["cells"]) == 5
    assert len(full["a"]) == 4
    local = joint_counts(frame, ["a", "b"], context={"site": "N"}, dropna=True)
    assert sum(c["count"] for c in local["cells"]) == 2
    assert local["missing_excluded_rows"] == 2
    assert local["restriction_excluded_rows"] == 1
    json.dumps(local.to_dict(), allow_nan=False)
    assert "observed" in render_svg(local, detail="topology")


def test_joint_cell_budget_includes_unobserved_cells_and_invalid_inputs():
    frame = pd.DataFrame({"a": range(10), "b": range(10)})
    with pytest.raises(ValueError, match="exceeds max_cells"):
        joint_counts(frame, ["a", "b"], max_cells=99)
    assert len(joint_counts(frame, ["a", "b"], max_cells=100)["cells"]) == 10
    with pytest.raises(ValueError, match="exactly two"):
        joint_counts(frame, ["a"])
    with pytest.raises(ValueError, match="disjoint"):
        joint_counts(frame, ["a", "b"], context={"a": 1})
    with pytest.raises(ValueError, match="max_cells"):
        joint_counts(frame, ["a", "b"], max_cells=None)


def test_min_count_omits_small_cells_and_reports_their_mass():
    frame = pd.DataFrame({"a": list("xxxxyyzw"), "b": list("ppqqppqq")})
    result = joint_counts(frame, ["a", "b"], min_count=2)
    # (z, q) and (w, q) have one row each; z and w then support no reported cell.
    assert (result["a"], result["b"]) == (["x", "y"], ["p", "q"])
    assert [c["count"] for c in result["cells"]] == [2, 2, 2]
    assert (result["omitted_cells"], result["omitted_rows"]) == (2, 2)
    assert result["evaluated_rows"] == 8
    assert Recipe("joint_counts", result["parameters"]).run(frame).to_dict() == result.to_dict()
    topology = visualization_data(result, detail="topology")
    assert topology["omitted"] is True
    assert not {"z", "w"} & set(topology["a"])
    for detail in ("full", "topology"):
        assert "z" not in render_plaintext(result, detail=detail).split("a / b")[1]
    every = joint_counts(frame, ["a", "b"])
    assert (len(every["cells"]), every["omitted_cells"], every["omitted_rows"]) == (5, 0, 0)
    assert visualization_data(every, detail="topology")["omitted"] is False
    # The budget applies to the reported grid.
    assert joint_counts(frame, ["a", "b"], min_count=2, max_cells=4)["cells"]
    with pytest.raises(ValueError, match="max_cells"):
        joint_counts(frame, ["a", "b"], max_cells=4)
    for invalid in (-1, None, True):
        with pytest.raises(ValueError, match="min_count"):
            joint_counts(frame, ["a", "b"], min_count=invalid)
