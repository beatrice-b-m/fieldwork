"""Compare default discovery JSON across revisions on deterministic semantic cases.

Run this same script with each source checkout on PYTHONPATH and the same dependency
versions. No progress or new budget options are used, so the baseline can run it.
Outputs are intentionally separate from timing measurements.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import fieldwork as fw

# This allowlist is intentionally limited to the additive discovery extension.
# In particular candidate repeated_rows and every foundation graph field remain
# part of the oracle. Presentation ordering is tested separately; to_dict keeps
# its original analytical order.
SUPPORT_FIELDS = {
    "determinant_evaluated_rows",
    "target_observed_rows",
    "target_coverage",
    "target_missing_excluded_rows",
    "repeated_rows",
    "repeat_coverage",
    "repeat_modal_accuracy",
}
CANDIDATE_SUPPORT_FIELDS = {
    "determines_with_repeated_support",
    "global_targets_tested",
    "global_targets_possible",
}


def without_support_extension(value):
    value = deepcopy(value)

    def normalize(result):
        if result.get("kind") == "dependencies":
            for record in result.get("dependencies", []):
                for key in SUPPORT_FIELDS:
                    record.pop(key, None)
            for candidate in result.get("candidates", []):
                for key in CANDIDATE_SUPPORT_FIELDS:
                    candidate.pop(key, None)
        for finding in result.get("findings", []):
            if finding["pattern"] in {"exact_dependency", "approximate_dependency"}:
                for key in SUPPORT_FIELDS:
                    finding["measurements"].pop(key, None)
        if result.get("kind") == "overview":
            for section in result["sections"].values():
                normalize(section)

    for case in value:
        normalize(case["result"])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--allow-evidence-support-extension", action="store_true")
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
                (
                    "availability",
                    fw.suggest_paths,
                    {"objective": "availability", "max_candidates": 50},
                ),
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
    if args.compare:
        before, after = json.loads(args.compare.read_text()), json.loads(encoded)
        if args.allow_evidence_support_extension:
            before, after = without_support_extension(before), without_support_extension(after)
        if before != after:
            raise SystemExit("Analytical result parity failed")
    print(f"{len(results)} cases: {fw.__file__}")


if __name__ == "__main__":
    main()
