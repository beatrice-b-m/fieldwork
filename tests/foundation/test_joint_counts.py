from __future__ import annotations

import json

import pandas as pd
import pytest

from fieldwork import joint_counts, render_svg


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
    assert local["scopes"][0]["missing_excluded_rows"] == 2
    assert local["scopes"][0]["restriction_excluded_rows"] == 1
    assert local["scopes"][0]["conditional"]
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
