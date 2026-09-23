"""Generate synthetic HTML stress cases for scripts/check_output_ux.cjs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import fieldwork as fw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from wide_table import investigate


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(21)
    count = 20_000
    wide = pd.DataFrame({"entity": np.arange(count) // 4, "site": np.arange(count) % 12})
    for index in range(64):
        wide[f"measurement_{index:02d}_" + "long_label_" * 6] = rng.choice(
            [None, "recorded", "pending", "unavailable"], count
        )
    long = "A_very_long_unbroken_label_" * 12 + '</script><img src=x onerror="alert(1)">\x00'
    awkward = pd.DataFrame(
        {
            long: [long, "東京", "A", None] * 8,
            "group": ["one", "one", "two", "two"] * 8,
            "entity": [1, 1, 2, 2] * 8,
        },
        index=[7] * 32,
    )
    small = pd.DataFrame({"key": [1, 1, 2, 2], "value": ["a", "a", "b", None]})
    _, laboratory = investigate()
    availability = fw.missingness(
        wide,
        limits={"max_pairs": 24, "max_signatures": 8, "example_limit": 2},
    )
    dependencies = fw.discover_dependencies(
        wide,
        limits={"max_candidates": 4, "max_dependency_tests": 60},
        include_grain=False,
    )
    cases = {
        "wide-missingness": fw.render_html(availability, max_findings=60),
        "wide-dependencies": fw.render_html(dependencies, max_findings=20),
        "wide-grain": fw.render_html(fw.grain(wide, ["entity", "site"])),
        "laboratory": fw.render_html(laboratory),
        "many-key-grain": fw.render_html(fw.grain(wide.iloc[:200], list(wide.columns)[:18])),
        "long-discovery": fw.render_html(fw.explore(awkward)),
        "long-levels": fw.render_html(fw.levels(awkward)),
        "long-grain": fw.render_html(fw.grain(awkward, [long, "entity"])),
        "long-joint": fw.render_html(fw.joint_counts(awkward, [long, "group"])),
        "deep-census": fw.render_html(fw.census(wide, list(wide.columns)[:6], max_nodes=160)),
        "pairs": fw.render_html(
            fw.profile(
                wide,
                list(wide.columns)[2:18],
                pairs={"pair_contexts": [{"site": 1}], "limits": {"max_pairs": 24}},
            ),
            section="pairs",
        ),
        "overview": fw.render_html(fw.explore(small)),
        "comparison": fw.render_html(
            fw.compare(fw.missingness(small), fw.missingness(small.iloc[:2]))
        ),
        "empty": fw.render_html(fw.discover_dependencies(small.iloc[:0])),
        "zero-limit": fw.render_html(fw.explore(small), max_findings=0),
        "topology": fw.render_html(availability, detail="topology", max_findings=12),
    }
    for name, markup in cases.items():
        (output / f"{name}.html").write_text(markup, encoding="utf-8")
    (output / "fixtures.json").write_text(
        json.dumps({"rows": count, "columns": len(wide.columns), "cases": list(cases)}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
