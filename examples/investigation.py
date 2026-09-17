"""Discover a field family, inspect its exception, and choose a census path."""

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
    paths = fw.suggest_paths(df, features=["site", "exam_id", "modality"], max_dimensions=3)
    tree = fw.census(df, paths.best.dimensions)
    return df, availability, exceptions, paths, tree


if __name__ == "__main__":
    _, availability, exceptions, paths, tree = investigate()
    print(availability)
    print(exceptions)
    print(paths.best.dimensions)
    print(tree)
