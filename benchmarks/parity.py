"""Compare default discovery JSON across revisions on deterministic semantic cases.

Run this same script with each source checkout on PYTHONPATH and the same dependency
versions. No progress or new budget options are used, so the baseline can run it.
Outputs are intentionally separate from timing measurements.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fieldwork as fw

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--compare", type=Path)
args = parser.parse_args()
results = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            f"c{i}": np.where(rng.random(31) < 0.25, np.nan, rng.integers(0, 4, 31))
            for i in range(5)
        },
        index=[1] * 31,
    )
    df["label"] = pd.Series(["a1", "b02", None, "c3"] * 8).iloc[:31].to_numpy()
    scope = fw.Scope.from_positions(df, range(0, 31, 2), name="evens")
    for context in ({}, {"scope": scope, "missing": {"c0": [1]}, "table_id": "sample"}):
        for name, func, kw in [
            ("missingness", fw.missingness, {}),
            (
                "entities",
                fw.missingness,
                {"entity": "c0", "unit": "entities", "entity_presence": "all", "by": ["c1"]},
            ),
            ("entity_summaries", fw.missingness, {"entity": "c0", "by": ["c1"]}),
            (
                "dependencies",
                fw.discover_dependencies,
                {"max_candidates": 8, "by": ["c2"], "min_accuracy": 0.1},
            ),
            (
                "dependencies_na",
                fw.discover_dependencies,
                {"max_candidates": 8, "dropna": False, "min_accuracy": 0.1},
            ),
            ("paths", fw.suggest_paths, {"max_candidates": 50}),
            (
                "target",
                fw.suggest_paths,
                {"objective": "target", "target": "c0", "max_candidates": 50},
            ),
            ("availability", fw.suggest_paths, {"objective": "availability", "max_candidates": 50}),
            ("patterns", fw.value_patterns, {"by": ["c0"]}),
            ("overview", fw.explore, {"discovery": {"max_candidates": 30, "by": ["c0"]}}),
        ]:
            results.append(
                {
                    "seed": seed,
                    "kind": name,
                    "scoped": bool(context),
                    "result": func(df, **context, **kw).to_dict(),
                }
            )
encoded = json.dumps(results, allow_nan=False, sort_keys=True)
args.output.write_text(encoded + "\n")
if args.compare and json.loads(args.compare.read_text()) != json.loads(encoded):
    raise SystemExit("Analytical result parity failed")
print(f"{len(results)} cases: {fw.__file__}")
