"""Generate full and topology SVG/HTML examples without optional dependencies.

Run: python examples/observed_grain_graph.py /tmp/fieldwork-figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fieldwork import KeySpec, explore, joint_counts, render_html, render_svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "exam_id": [1, 1, 2, 2, 3, 3],
            "side": ["L", "R", "L", "R", "L", "L"],
            "finding": ["clear", "scar", "clear", "clear", "scar", "scar"],
            "site": ["North", "South", "North", "South", "North", "South"],
        }
    )
    result = explore(
        frame,
        ["side", "finding"],
        features=["side", "finding"],
        candidate_keys=["exam_id", KeySpec("exam_side", ("exam_id", "side"))],
        top_n=1,
        pair_contexts=[{"site": "North"}],
    )
    for detail in ("full", "topology"):
        for section in ("grain", "levels", "census", "pairs"):
            for extension, renderer in (("svg", render_svg), ("html", render_html)):
                (args.output / f"{section}-{detail}.{extension}").write_text(
                    renderer(result, section=section, detail=detail),
                    encoding="utf-8",
                )
        selected = joint_counts(frame, ["side", "finding"])
        (args.output / f"joint-{detail}.svg").write_text(
            render_svg(selected, detail=detail),
            encoding="utf-8",
        )
    (args.output / "grain-exceptions.svg").write_text(
        render_svg(result, show_exceptions=True),
        encoding="utf-8",
    )
    (args.output / "grain-matrix.svg").write_text(
        render_svg(result, view="matrix"),
        encoding="utf-8",
    )
    print(f"Wrote figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
