"""Discover → inspect → refine scope → compare → save/reapply on one source."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import fieldwork as fw


def sample():
    return pd.DataFrame(
        {
            "site": ["North"] * 4 + ["South"] * 4,
            "exam_id": ["N01", "N01", "N02", "N02", "S01", "S01", "S02", "S02"],
            "modality": ["MR", "MR", "CT", "CT", "MR", "MR", "CT", "CT"],
            "image_1": [10, 11, None, None, 20, 21, None, None],
            "image_2": [12, 13, None, None, 22, None, None, None],
            "dose": [None, None, 2, 3, None, None, 4, 5],
        },
        index=[0] * 8,
    )


def investigate():
    df = sample()
    availability = fw.missingness(df, by=["site"], entity="exam_id", min_implication=0.75)
    edge = next(
        f
        for f in availability["findings"]
        if f["pattern"] == "presence_implication"
        and [c["column"] for c in f["features"]] == ["image_1", "image_2"]
    )
    exceptions = availability.inspect(df, edge["id"], exceptions=True)
    signatures = fw.missingness(df, features=["image_1", "dose"], example_limit=1)
    signature = next(s for s in signatures["signatures"] if s["present"] == ["image_1"])
    image_scope = signatures.select(df, signature["finding_id"], name="image-bearing rows")
    assert image_scope.positions == (0, 1, 4, 5)
    local = fw.missingness(df, scope=image_scope, entity="exam_id")
    comparison = fw.compare(availability, local)
    assert comparison["changes"]
    paths = fw.suggest_paths(
        df,
        features=["site", "exam_id", "image_2"],
        start_with=["site"],
        max_dimensions=3,
        scope=image_scope,
        missing={"image_2": [-999]},
    )
    tree = paths.best.census(df)
    assert tree["scopes"][0]["input_rows"] == len(df)
    assert tree["scopes"][0]["evaluated_rows"] == len(image_scope.positions)
    recipe = fw.Recipe("missingness", {"entity": "exam_id", "unit": "entities"})
    with TemporaryDirectory() as directory:
        path = Path(directory) / "availability-recipe.json"
        recipe.save(path)
        assert fw.Recipe.load(path).run(df)["analysis_unit"]["denominator"] == 4
    saved = json.loads(json.dumps(signatures.to_dict(), allow_nan=False))
    restored = fw.InvestigationResult.from_dict(saved)
    assert restored.select(df, signature["finding_id"]).positions == image_scope.positions
    assert "Inspect findings" in fw.render_html(restored)
    overview = fw.explore(df)
    assert not overview.relationships("image_1", kinds=["indexed_name"]).empty
    return df, availability, exceptions, paths, tree


if __name__ == "__main__":
    _, availability, exceptions, paths, tree = investigate()
    print(availability)
    print(exceptions)
    print(paths.best.dimensions)
    print(tree)
