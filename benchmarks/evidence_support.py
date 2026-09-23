"""Deterministic dependency evidence qualification; timings live in discovery.py.

Run the same harness against both source revisions with identical dependencies.
Missing additive measurements are recorded as null, never measured zero.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import fieldwork as fw

VERSION = 2
DEPENDENCY_FIELDS = (
    "exact",
    "evaluated_rows",
    "missing_excluded_rows",
    "evaluated_groups",
    "repeated_groups",
    "repair_rows",
    "modal_accuracy",
    "determinant_evaluated_rows",
    "target_observed_rows",
    "target_coverage",
    "target_missing_excluded_rows",
    "repeated_rows",
    "repeat_coverage",
    "repeat_modal_accuracy",
)


def cases():
    sparse = pd.DataFrame({"X": [1, 1, 2, 2], "Y": ["a", None, "b", None]})
    yield "sparse_target", sparse, {}
    yield "singleton_inflation", pd.DataFrame({"X": [*range(99), 98], "Y": ["a"] * 99 + ["b"]}), {}
    yield "empty", sparse.iloc[:0], {}
    yield "missing_target", sparse.assign(Y=None), {}
    yield "missing_category", sparse, {"dropna": False}
    yield (
        "sentinel_category",
        sparse.fillna("absent"),
        {"dropna": False, "missing": {"Y": ["absent"]}},
    )
    contextual = pd.DataFrame(
        {
            "X": [1, 1, 2, 2, None],
            "Y": ["a", "a", "b", None, "c"],
            "C": [None, None, "rare", "other", "other"],
        },
        index=[0] * 5,
    )
    yield (
        "conditional_composite",
        contextual,
        {"limits": {"max_candidates": 6}, "max_key_size": 2, "by": ["C"]},
    )
    yield "scoped", contextual, {"scope": fw.Scope.from_positions(contextual, [0, 1, 3, 4])}
    yield "zero_tests", sparse, {"limits": {"max_dependency_tests": 0}}
    yield "partial_tests", contextual, {"limits": {"max_candidates": 3, "max_dependency_tests": 2}}
    yield (
        "permuted_partial_tests",
        contextual[["Y", "C", "X"]],
        {"limits": {"max_candidates": 3, "max_dependency_tests": 2}},
    )
    yield (
        "overlapping_grains",
        pd.DataFrame({"X": [1, 1, 2, 2], "Y": [1, 1, None, None], "Z": [None, None, 2, 2]}),
        {"limits": {"max_candidates": 3}},
    )

    yield "missing_determinant", sparse.assign(X=None), {}
    yield "sentinel_excluded", sparse.fillna("absent"), {"missing": {"Y": ["absent"]}}
    yield "independent_graph_budget", sparse, {"limits": {"max_grain_views": 0}}
    yield "incompatible_transitivity", sparse.assign(Z=range(4)), {"limits": {"max_candidates": 3}}
    yield (
        "ranking_reversal",
        pd.DataFrame(
            {
                "X": [1, 1, 2, 2, 3, 3, 4, 4],
                "Z": ["a", "b", "a", "b", "c", "d", "c", "d"],
                "W": ["a", "b", "a", "b", "c", "d", "c", "d"],
                **{f"Y{i}": [10, None, 20, None, 30, None, 40, None] for i in range(3)},
            }
        ),
        {"limits": {"max_candidates": 6}},
    )
    yield (
        "rare_complete_context",
        pd.DataFrame(
            {"X": [1] * 6, "Y": ["a", "a", "b", "c", "d", "e"], "C": ["rare"] * 2 + ["other"] * 4}
        ),
        {"by": ["C"]},
    )


def report(source_commit=None):
    records = []
    for name, frame, overrides in cases():
        limits = {"max_candidates": 1, **overrides.get("limits", {})}
        options = {"max_key_size": 1, **overrides, "limits": limits}
        result = fw.discover_dependencies(frame, **options)
        overview = {
            "kind": "overview",
            "sections": {"dependencies": result.to_dict(), "missingness": {}, "paths": {}},
        }
        parameters = {
            k: list(v.positions) if isinstance(v, fw.Scope) else v for k, v in options.items()
        }
        records.append(
            {
                "case": name,
                "parameters": parameters,
                "scope": result["scope"],
                "coverage": result["coverage"],
                "candidate_order": [c["columns"] for c in result["candidates"]],
                "presentation_order": [
                    c["columns"] for c in fw.visualization_data(overview)["overview"]["grains"]
                ],
                "candidates": [
                    {
                        **c,
                        **{
                            k: c.get(k)
                            for k in (
                                "determines_with_repeated_support",
                                "global_targets_tested",
                                "global_targets_possible",
                            )
                        },
                    }
                    for c in result["candidates"]
                ],
                "dependencies": [
                    {
                        "determinant": d["determinant"],
                        "target": d["target"],
                        "context": d["context"],
                        **{k: d.get(k) for k in DEPENDENCY_FIELDS},
                    }
                    for d in result["dependencies"]
                ],
                "graph_sha256": hashlib.sha256(
                    json.dumps(
                        {k: result[k] for k in ("grain_views", "graph_selection")},
                        sort_keys=True,
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
            }
        )
    return {
        "report_version": 1,
        "fixture_harness_version": VERSION,
        "seed": 721,
        "source_commit": source_commit
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "cases": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-commit", help="Explicit revision when importing an archived source tree"
    )
    args = parser.parse_args()
    args.output.write_text(json.dumps(report(args.source_commit), indent=2, allow_nan=False) + "\n")
