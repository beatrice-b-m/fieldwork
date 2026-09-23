"""Isolated, timeout-bounded scaling measurements; no timing assertions.

Run with the project's Python environment. Each operation runs in a fresh process.
Fixture construction, analysis, and serialization are timed separately. Optional
cProfile runs are diagnostic and must not be compared with unprofiled timings.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fieldwork as fw
from fieldwork.evidence import fingerprint, prepare

OPERATIONS = (
    "fingerprint",
    "prepare",
    "missingness",
    "dependencies",
    "paths",
    "patterns",
    "explore",
    "levels",
    "census",
    "grain",
    "pairs",
    "joint_counts",
    "infer_schema",
)


def fixture(rows, columns, kind, seed):
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(columns):
        values = rng.integers(0, 8, rows).astype(float)
        if kind == "sparse":
            values[rng.random(rows) < 0.7] = np.nan
        elif kind == "mixed":
            if i % 3 == 0:
                values = rng.normal(size=rows)
            elif i % 3 == 1:
                values = np.array([f"code-{int(v):03}" for v in values], dtype=object)
        elif kind == "structured":
            # Unique IDs, repeated entities/contexts and nested availability.
            if i == 0:
                values = np.arange(rows)
            elif i == 1:
                values = np.arange(rows) // 4
            elif i == 2:
                values = np.arange(rows) % 8
            else:
                values = (np.arange(rows) // 4 % (8 + i % 4)).astype(float)
                values[np.arange(rows) % 10 < i % 7] = np.nan
        data[f"field_{i}"] = values
    return pd.DataFrame(data)


def workload(frame, args, progress=None):
    selected = list(frame.columns[: args.features]) if args.features else None
    common = {"features": selected}
    # Match the overview's dependency budget, rather than the standalone 100.
    candidates = {"max_candidates": args.candidates or 20}
    dependency_options = {"max_key_size": 1, "limits": candidates, **common}
    path_options = {}
    if args.candidates is not None:
        path_options["limits"] = {"max_candidates": args.candidates}
    dimensions = list(frame.columns[:4])
    operations = {
        "fingerprint": lambda: fingerprint(frame),
        "prepare": lambda: prepare(frame),
        "missingness": lambda: fw.missingness(frame, progress=progress, **common),
        "dependencies": lambda: fw.discover_dependencies(
            frame, progress=progress, **dependency_options
        ),
        "paths": lambda: fw.suggest_paths(frame, progress=progress, **common, **path_options),
        "patterns": lambda: fw.value_patterns(
            frame, limits={"max_pairs": 20}, progress=progress, **common
        ),
        "explore": lambda: fw.explore(
            frame,
            features=selected,
            options={"paths": path_options} if path_options else None,
            progress=progress,
        ),
        "levels": lambda: fw.levels(
            frame, selected or list(frame.columns), top_n=5, progress=progress
        ),
        "census": lambda: fw.census(
            frame, dimensions, max_nodes=40, max_levels=8, progress=progress
        ),
        "grain": lambda: fw.grain(
            frame[selected] if selected else frame,
            (selected or list(frame.columns))[:2],
            progress=progress,
        ),
        "pairs": lambda: fw.pairs(frame, dimensions, progress=progress),
        "joint_counts": lambda: fw.joint_counts(frame, dimensions[:2], progress=progress),
        "infer_schema": lambda: fw.infer_schema(frame, progress=progress),
    }
    return operations[args.worker]()


def worker(args):
    started = time.perf_counter()
    frame = fixture(args.rows, args.columns, args.fixture, args.seed)
    fixture_seconds = time.perf_counter() - started
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    events = []
    started = time.perf_counter()
    result = workload(frame, args, events.append if args.progress else None)
    seconds = time.perf_counter() - started
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(args.profile))
    record = {
        "status": "completed",
        "analysis_seconds": seconds,
        "fixture_seconds": fixture_seconds,
        "frame_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
    }
    if args.progress:
        record["progress_events"] = len(events)
        record["phases"] = [
            {
                "phase": e.phase,
                "phase_id": e.phase_id,
                "parent_id": e.parent_id,
                "seconds": e.phase_elapsed_seconds,
                "completed": e.completed,
                "total": e.total,
            }
            for e in events
            if e.status == "completed"
        ]
    if hasattr(result, "to_dict"):
        started = time.perf_counter()
        encoded = json.dumps(result.to_dict(), allow_nan=False).encode()
        record.update(
            serialization_seconds=time.perf_counter() - started,
            result_bytes=len(encoded),
            coverage=result.payload.get("coverage"),
        )
        if "grain_views" in result.payload:
            record["grain_views"] = len(result["grain_views"])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    print(json.dumps(record, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--columns", type=int, default=150)
    parser.add_argument(
        "--fixture", choices=["sparse", "dense", "mixed", "structured"], default="sparse"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Collect callback counts and inclusive phase durations",
    )
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=["explore"])
    parser.add_argument(
        "--features", type=int, help="First N features for discovery/levels/grain; input unchanged"
    )
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", type=Path, help="One operation/repeat only; timings include overhead"
    )
    parser.add_argument("--worker", choices=OPERATIONS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rows < 1 or args.columns < 4 or args.repeats < 1 or args.timeout <= 0:
        parser.error("rows/repeats/timeout must be positive; columns must be at least four")
    if args.features is not None and not 1 <= args.features <= args.columns:
        parser.error("features must be between one and columns")
    if args.profile and (len(args.operations) != 1 or args.repeats != 1):
        parser.error("profiling requires exactly one operation and one repeat")
    if args.worker:
        worker(args)
        return
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "fieldwork": fw.__version__,
        },
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "worker"
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
    for operation in args.operations:
        for repeat in range(args.repeats):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker",
                operation,
            ]
            print(
                f"Starting {operation}, repeat {repeat + 1}, {args.rows} x {args.columns}",
                flush=True,
            )
            started = time.perf_counter()
            try:
                child = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=args.timeout,
                    env=os.environ.copy(),
                )
                record = (
                    json.loads(child.stdout)
                    if child.returncode == 0
                    else {"status": "error", "stderr": child.stderr}
                )
            except subprocess.TimeoutExpired:
                record = {"status": "timeout", "timeout_seconds": args.timeout}
            record.update(
                operation=operation,
                repeat=repeat + 1,
                process_seconds=time.perf_counter() - started,
            )
            report["results"].append(record)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
