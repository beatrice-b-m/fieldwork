"""Realistic-scale smoke test: a wide 50k-row export with high-cardinality keys.

Deselect locally with ``pytest -m "not slow"``; CI always runs it.
"""

import json

import numpy as np
import pandas as pd
import pytest

import fieldwork as fw

ROWS = 50_000


def export():
    rng = np.random.default_rng(11)
    patient = rng.integers(0, 8_000, ROWS)
    site = patient % 12
    frame = pd.DataFrame(
        {
            "row_id": np.arange(ROWS),
            "patient_id": [f"P{p:05d}" for p in patient],
            "exam_id": patient * 4 + rng.integers(0, 4, ROWS),
            "site": [f"S{s:02d}" for s in site],
            "region": np.array(["north", "south", "east", "west"])[site % 4],
            "birth_year": 1940 + patient % 60,
            "modality": rng.choice(["CT", "MR", "US", "XR"], ROWS),
            "value": rng.normal(size=ROWS).round(3),
        }
    )
    exceptions = np.sort(rng.choice(ROWS, 25, replace=False))
    frame.loc[exceptions, "region"] = "unknown"
    sparse = rng.random(ROWS) < 0.3
    for i in range(12):
        frame[f"lab_{i}"] = np.where(sparse, rng.integers(0, 50, ROWS), np.nan)
    for i in range(12):
        frame[f"note_{i}"] = rng.choice(["a", "b", "c", None], ROWS)
    return frame, exceptions, sparse


def saved_positions(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"positions", "selection_positions"} and isinstance(child, list):
                yield child
            yield from saved_positions(child)
    elif isinstance(value, list):
        for child in value:
            yield from saved_positions(child)


@pytest.mark.slow
def test_wide_export_overview_surfaces_planted_structure():
    frame, exceptions, sparse = export()
    original = frame.copy(deep=True)
    overview = fw.explore(frame)
    pd.testing.assert_frame_equal(frame, original)

    # The planted near-rule is the top lead, and its exceptions are the planted rows.
    top = overview["findings"][0]
    assert top["pattern"] == "approximate_dependency"
    assert top["measurements"]["target"] == "region"
    rule = next(
        f
        for f in overview["findings"]
        if f["pattern"] == "approximate_dependency" and f["measurements"]["determinant"] == ["site"]
    )
    assert rule["measurements"]["repair_rows"] == len(exceptions)
    selected = overview.select(frame, rule["id"], exceptions=True)
    assert selected.positions == tuple(exceptions)

    families = overview["sections"]["missingness"]["families"]
    assert [f["features"] for f in families] == [[f"lab_{i}" for i in range(12)]]
    assert families[0]["populated"] == sparse.sum()

    grains = {tuple(g["columns"]): g for g in fw.visualization_data(overview)["overview"]["grains"]}
    assert grains[("row_id",)]["role"] == "unique identifier"
    assert {"patient_id", "birth_year"} <= set(grains[("exam_id",)]["determines"])
    assert "birth_year" in grains[("patient_id",)]["determines"]
    coverage = overview["sections"]["dependencies"]["coverage"]
    assert coverage["candidates_evaluated"] < coverage["candidate_space"]

    # Saved evidence stays bounded and strict-JSON at scale.
    data = json.loads(json.dumps(overview.to_dict(), allow_nan=False))
    assert max(len(p) for p in saved_positions(data)) <= 5
    assert fw.render_plaintext(data)
    assert fw.render_html(data)
