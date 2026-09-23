"""Which exams have complete paired-image references, and did a re-export fix the gap?

The notebook explains each decision and output. This compact companion runs the
same synthetic investigation and supplies the documentation's rendered examples.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import fieldwork as fw

MISSING = {"image_2": [-999]}
FEATURES = ["image_1", "image_2", "dose"]


def sample():
    """One row per export slot; only the synthetic MR workflow expects image pairs."""
    return pd.DataFrame(
        {
            "site": ["North"] * 7 + ["South"] * 5,
            "exam_id": ["N01"] * 3
            + ["N02"] * 2
            + ["N03"] * 2
            + ["S01"] * 2
            + ["S02"]
            + ["S03"] * 2,
            "modality": ["MR"] * 5 + ["CT"] * 2 + ["MR"] * 3 + ["CT"] * 2,
            "slot": [1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 1, 2],
            "image_1": [10, 11, 12, 13, 14, None, None, 20, 21, 22, None, None],
            "image_2": [110, 111, 112, 113, 114, None, None, 120, -999, 122, None, None],
            "dose": [None] * 5 + [2, 3] + [None] * 3 + [4, 5],
        },
        index=[0] * 12,
    )


def mr_population(df):
    """Rebuild eligibility from modality, independently of image completeness."""
    cohorts = fw.missingness(
        df, features=FEATURES, by=["modality"], missing=MISSING, example_limit=1
    )
    context = next(c for c in cohorts["contexts"] if c["values"]["modality"] == "MR")
    return cohorts.select(df, context["finding_id"], name="MR export slots")


def corrected_delivery(df):
    """Simulate a provider re-export of the same slots with the missing reference recovered."""
    corrected = df.copy()
    corrected.loc[(corrected["exam_id"] == "S01") & (corrected["slot"] == 2), "image_2"] = 121
    return corrected


def investigate():
    df = sample()
    availability = fw.missingness(
        df,
        features=FEATURES,
        by=["modality"],
        missing=MISSING,
        min_implication=0.85,
        example_limit=1,
    )
    edge = next(
        f
        for f in availability["findings"]
        if f["pattern"] == "presence_implication"
        and [c["column"] for c in f["features"]] == ["image_1", "image_2"]
    )
    exceptions = availability.inspect(df, edge["id"], exceptions=True)
    mr_scope = mr_population(df)
    local = fw.missingness(
        df,
        features=["image_1", "image_2"],
        scope=mr_scope,
        entity="exam_id",
        missing=MISSING,
    )
    # This is a population contrast, not a change between deliveries.
    comparison = fw.compare(availability, local)
    assert comparison["changes"]
    paths = fw.suggest_paths(
        df,
        features=["site", "exam_id"],
        start_with=["site"],
        max_dimensions=2,
        scope=mr_scope,
        missing=MISSING,
    )
    tree = paths.best.census(df)
    assert tree["scopes"][0]["input_rows"] == 12
    assert tree["scopes"][0]["evaluated_rows"] == 8
    recipe = fw.Recipe(
        "missingness",
        {
            "features": ["image_1", "image_2"],
            "entity": "exam_id",
            "unit": "entities",
            "entity_presence": "all",
            "missing": MISSING,
        },
        notes="Complete references on every MR export slot; select MR anew for each delivery.",
    )
    next_delivery = corrected_delivery(df)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "availability-recipe.json"
        recipe.save(path)
        loaded = fw.Recipe.load(path)
        before = loaded.run(df, scope=mr_scope)
        after = loaded.run(next_delivery, scope=mr_population(next_delivery))
    delivery_change = fw.compare(before, after)
    image_change = next(c for c in delivery_change["changes"] if c["feature"] == "image_2")
    assert image_change["before"]["populated"] == 3
    assert image_change["after"]["populated"] == 4
    # Results retain evidence for this source; recipes retain reusable settings.
    saved = json.loads(json.dumps(availability.to_dict(), allow_nan=False))
    restored = fw.InvestigationResult.from_dict(saved)
    assert restored.select(df, edge["id"], exceptions=True).positions == (8,)
    assert "Inspect findings" in fw.render_html(restored)
    return df, availability, exceptions, paths, tree


if __name__ == "__main__":
    _, availability, exceptions, paths, tree = investigate()
    print("Paired-image export investigation (synthetic data)")
    print(availability)
    print("Exception to investigate: South / S01 / slot 2")
    print(exceptions)
    print(paths.best.dimensions)
    print(tree)
    print("Complete MR exams: 3/4 initially; 4/4 after the simulated corrected delivery.")
