"""Target populations, repeated support, and source-free schema-1.0 compatibility."""

import json
from pathlib import Path

import pandas as pd

import fieldwork as fw


LEGACY = Path(__file__).parent / "fixtures" / "dependency-schema-1.0.json"


def sparse_frame():
    return pd.DataFrame({"X": [1, 1, 2, 2], "Y": ["a", None, "b", None]}, index=[0] * 4)


def dependency(result, key=("X",), target="Y", context=None):
    return next(
        d
        for d in result["dependencies"]
        if d["determinant"] == list(key) and d["target"] == target and d["context"] == context
    )


def test_sparse_target_characterization():
    result = fw.discover_dependencies(sparse_frame(), max_key_size=1)
    d = dependency(result)
    assert d["exact"] is True
    assert d["evaluated_rows"] == 2
    assert d["repeated_groups"] == 0
    assert result["candidates"][0]["repeated_rows"] == 4
    assert result["candidates"][0]["determines"] == ["Y"]
    finding = next(f for f in result["findings"] if f["measurements"]["determinant"] == ["X"])
    assert result.select(sparse_frame(), finding["id"]).positions == (0, 2)


def test_legacy_export_loads_without_source():
    data = json.loads(LEGACY.read_text())
    result = fw.InvestigationResult.from_dict(data)
    assert result.schema_version == "1.0"
    assert result.to_dict() == data
    for render in (fw.render_plaintext, fw.render_svg, fw.render_html):
        assert "X" in render(result)
    assert result.select(sparse_frame(), result["findings"][0]["id"]).positions == (0, 2)
