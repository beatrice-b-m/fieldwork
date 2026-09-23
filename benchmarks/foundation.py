"""Reproducible synthetic end-to-end benchmark for the hierarchy explorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from fieldwork import grain, levels, profile, render_plaintext


def fixture(name: str, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cardinalities = {
        "low3": (2, 4, 8),
        "low6": (4, 4, 4, 4, 4, 4),
        "high6": (100, 100, 100, 100, 100, 100),
        "mixed6": (17_000, 50_000, 2, 4, 8, 10_000),
    }[name]
    return pd.DataFrame(
        {
            f"c{index}": np.char.add("v", rng.integers(0, cardinality, rows).astype(str))
            for index, cardinality in enumerate(cardinalities)
        }
    )


def workload(frame: pd.DataFrame, name: str):
    columns = frame.columns.tolist()
    if name == "s1":
        return levels(frame, columns, top_n=5)
    if name == "s3":
        return grain(frame, columns[:2])
    result = profile(
        frame,
        columns[:6],
        candidate_keys=columns[:2],
        top_n=5,
        max_nodes=10_000,
        include_pairs=True,
    )
    if name == "render":
        return render_plaintext(result, max_lines=200)
    return json.dumps(result.to_dict(), allow_nan=False, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="acceptance", choices=["smoke", "acceptance"])
    parser.add_argument("--fixture", default="low6", choices=["low3", "low6", "high6", "mixed6"])
    parser.add_argument("--workload", default="explore", choices=["s1", "s3", "explore", "render"])
    parser.add_argument("--rows", type=int)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = args.rows or (10_000 if args.suite == "smoke" else 150_000)
    frame = fixture(args.fixture, rows, args.seed)
    workload(frame, args.workload)  # warmup
    times = []
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(args.repeats):
        start = time.perf_counter()
        workload(frame, args.workload)
        times.append(time.perf_counter() - start)
    # Allocation tracing is a separate run because it changes timings.
    tracemalloc.start()
    traced_output = workload(frame, args.workload)
    _, traced_peak = tracemalloc.get_traced_memory()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.stop()
    canonical = (
        traced_output
        if isinstance(traced_output, str)
        else json.dumps(traced_output.to_dict(), allow_nan=False)
    )
    record = {
        "fixture": args.fixture,
        "workload": args.workload,
        "rows": rows,
        "columns": len(frame.columns),
        "observed_cardinalities": {
            column: int(frame[column].nunique(dropna=False)) for column in frame.columns
        },
        "seed": args.seed,
        "repeats": args.repeats,
        "seconds": {
            "median": statistics.median(times),
            "minimum": min(times),
            "maximum": max(times),
            "spread": max(times) - min(times),
            "raw": times,
        },
        "memory": {
            "baseline_peak_rss_platform_units": baseline_rss,
            "absolute_peak_rss_platform_units": peak_rss,
            "incremental_peak_rss_platform_units": max(0, peak_rss - baseline_rss),
            "tracemalloc_peak_bytes": traced_peak,
        },
        "output_bytes": len(canonical.encode()),
        "output_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"{args.fixture}-{args.workload}-{args.seed}.json"
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
