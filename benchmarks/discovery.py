"""Measure end-to-end discovery and strict JSON result size on seeded wide data."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fieldwork as fw

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=2000)
parser.add_argument("--columns", type=int, default=24)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rng = np.random.default_rng(721)
frame = pd.DataFrame(
    {
        f"field_{i}": np.where(rng.random(args.rows) < 0.7, np.nan, rng.integers(0, 8, args.rows))
        for i in range(args.columns)
    }
)
frame["entity"] = np.arange(args.rows) // 4
records = []
for name, run in [
    ("missingness", lambda: fw.missingness(frame, entity="entity")),
    ("dependencies", lambda: fw.discover_dependencies(frame, limits={"max_candidates": 12})),
    ("paths", lambda: fw.suggest_paths(frame, limits={"max_candidates": 60})),
]:
    start = time.perf_counter()
    result = run()
    elapsed = time.perf_counter() - start
    encoded = json.dumps(result.to_dict(), allow_nan=False).encode()
    records.append(
        {
            "operation": name,
            "seconds": elapsed,
            "result_bytes": len(encoded),
            "coverage": result["coverage"],
        }
    )
args.output.write_text(
    json.dumps({"rows": args.rows, "columns": len(frame.columns), "results": records}, indent=2)
    + "\n"
)
print(args.output.read_text())
