"""Render executable examples through the package's current presentation code."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from dependency_support import investigate as investigate_support
from investigation import investigate
from wide_table import investigate as investigate_wide_table

import fieldwork as fw


def generate(output: Path):
    import resvg_py

    output.mkdir(parents=True, exist_ok=True)
    df, availability, exceptions, paths, tree = investigate()
    _, wide_graph = investigate_wide_table()
    assert exceptions["exam_id"].tolist() == ["S01"]
    _, dependency_support = investigate_support()
    results = {
        "dependency-support": dependency_support,
        "wide-table": wide_graph,
        "availability": availability,
        "census": tree,
        "grain": fw.grain(df, ["site", "exam_id", "modality"]),
        "paths": paths,
        "availability-topology": availability,
    }
    manifest = {"package_version": fw.__version__, "schema": "1", "files": {}}
    for name, result in results.items():
        detail = "topology" if name.endswith("-topology") else "full"
        svg = fw.render_svg(result, detail=detail)
        files = {
            f"{name}.svg": svg.encode(),
            f"{name}.png": resvg_py.svg_to_bytes(
                svg_string=svg,
                skip_system_fonts=True,
                font_files=[
                    str(ROOT / "scripts/fonts/Lato-Regular.ttf"),
                    str(ROOT / "scripts/fonts/Lato-Bold.ttf"),
                ],
                font_family="Lato",
                sans_serif_family="Lato",
            ),
            f"{name}.html": fw.render_html(result, detail=detail).encode(),
            f"{name}.json": (
                json.dumps(fw.visualization_data(result, detail=detail), indent=2, allow_nan=False)
                + "\n"
            ).encode(),
        }
        for filename, content in files.items():
            (output / filename).write_bytes(content)
            manifest["files"][filename] = hashlib.sha256(content).hexdigest()
    example = (ROOT / "examples/wide_table.py").read_bytes()
    (output / "wide-table.py").write_bytes(example)
    manifest["files"]["wide-table.py"] = hashlib.sha256(example).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    readme = (ROOT / "README.md").read_text()
    readme = readme.replace(
        "](docs/assets/",
        f"](https://raw.githubusercontent.com/beatrice-b-m/fieldwork/v{fw.__version__}/docs/assets/",
    )
    for prefix in ("docs/", "examples/"):
        readme = readme.replace(
            f"]({prefix}",
            f"](https://github.com/beatrice-b-m/fieldwork/blob/v{fw.__version__}/{prefix}",
        )
    for filename in ("LICENSE", "NOTICE"):
        readme = readme.replace(
            f"]({filename})",
            f"](https://github.com/beatrice-b-m/fieldwork/blob/v{fw.__version__}/{filename})",
        )
    (output / "README.pypi.md").write_text(readme)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            generate(Path(directory))
            expected = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            stale = [
                name
                for name, content in expected.items()
                if not (args.output / name).exists() or (args.output / name).read_bytes() != content
            ]
            if (ROOT / "README.pypi.md").read_bytes() != expected["README.pypi.md"]:
                stale.append("root README.pypi.md")
            if stale:
                raise SystemExit(
                    "Stale assets; run uv run python scripts/generate_assets.py: "
                    + ", ".join(stale)
                )
    else:
        generate(args.output)
        (ROOT / "README.pypi.md").write_bytes((args.output / "README.pypi.md").read_bytes())
