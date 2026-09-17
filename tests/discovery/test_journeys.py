"""Regression journeys across discovery, inspection, scope and saved presentation."""

import json

import pandas as pd
import pytest

import fieldwork as fw


def test_recommendation_census_preserves_context_and_original_population():
    df = pd.DataFrame({"site": ["A", "A", "B", "B"], "value": [1, -999, 2, 3]}, index=[0] * 4)
    scope = fw.Scope.from_positions(df, [0, 1, 2], name="selected")
    paths = fw.suggest_paths(df, scope=scope, missing={"value": [-999]}, start_with=["value"])
    paths = fw.InvestigationResult.from_dict(json.loads(json.dumps(paths.to_dict())))
    tree = paths.best.census(df, dropna=True)
    assert tree["source"]["rows"] == 4
    assert tree["scopes"][0]["input_rows"] == 4
    assert tree["scopes"][0]["evaluated_rows"] == 2
    assert tree["scopes"][0]["missing_excluded_rows"] == 1
    assert tree["scopes"][0]["restriction_excluded_rows"] == 1
    preview = paths["paths"][0]["preview"]
    assert preview["scopes"][0]["input_rows"] == 4
    assert preview["scopes"][0]["evaluated_rows"] == 3
    assert {"type": "missing"} in [v["value"] for v in preview["level_dictionary"]]
    assert paths.best.census(df)["tree"] == preview["tree"]
    with pytest.raises(ValueError, match="differs"):
        paths.best.census(df.iloc[::-1])
    with pytest.raises(ValueError, match="context"):
        paths.best.census(df, missing={})
    explicit = fw.explore(
        df, ["value"], discovery={"scope": scope, "missing": {"value": [-999]}}, dropna=True
    )
    for name in ("levels", "census", "pairs"):
        for population in explicit["sections"][name]["scopes"]:
            assert population["input_rows"] == 4
            assert population["restriction_excluded_rows"] == 1
    with pytest.raises(ValueError, match="search options"):
        fw.explore(df, ["site"], discovery={"objective": "compact"})
