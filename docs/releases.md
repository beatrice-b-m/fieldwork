# Release and documentation workflow

## Prepare a release

1. Update the version in `pyproject.toml` and `src/fieldwork/__init__.py`; run `uv lock`.
2. Update README release wording, user docs, and developer contracts to match the
   intended release. Run the supported-Python test matrix.
3. Run `uv run python scripts/generate_assets.py`. It executes the worked example,
   renders public SVG/HTML/visualization-data APIs, rasterizes SVGs with pinned
   resvg and bundled OFL-licensed Lato fonts, and writes a SHA-256 asset manifest.
   The generated `README.pypi.md` uses version-tagged absolute image/link URLs.
4. Visually inspect the generated wide-table hero, availability, census, grain,
   path and topology images. Run the generator with `--check` to detect stale
   assets or README.
5. Run `uv build`, then smoke-test the wheel in an isolated environment with
   `uv run --no-project --isolated --with ./dist/*.whl python -I examples/investigation.py`.
   This exercises the worked investigation using the built package and its runtime
   dependencies. Commit the release's source and generated assets.
6. Tag the commit `v<version>` and publish a GitHub release after CI passes.

Publishing a release triggers `.github/workflows/release.yml`: verify tag/version,
run tests, regenerate images and reject an uncommitted difference, build sdist/wheel,
smoke-test the installed wheel, attach distributions plus the documentation asset
archive, and publish via PyPI trusted publishing. CI runs the same wheel smoke test
on each supported Python version.

If publishing fails after attachments have uploaded, rerun the workflow for the
same unchanged tag. The attachment upload uses `--clobber` to replace existing
files instead of failing on duplicate names.

## Configure trusted publishing before the first release

Create the GitHub environment `pypi`. On PyPI, add a GitHub Actions publisher with
the following values (use a pending publisher if the project does not exist yet):

| Field | Value |
| --- | --- |
| PyPI project name | `fieldwork` |
| GitHub owner | `beatrice-b-m` |
| Repository | `fieldwork` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

The workflow filename is entered without `.github/workflows/`. The environment
name must match exactly. No PyPI API-token secret is needed: the workflow already
requests `id-token: write` and uses `uv publish --trusted-publishing always`.
See [PyPI's pending-publisher setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
Project-name availability must be established by the maintainer.

After preparation and CI pass on the exact release candidate, publish a GitHub
release for its `v<version>` tag to trigger publication. Creating a trusted
publisher alone does not publish the package. Never move a published tag.

## Two documentation audiences

`docs/` in this repository is for human/agent developers, organized from map to
architecture to contracts and algorithms. `fieldwork-docs` contains the user-facing
Astro/Starlight site: orient, choose, inspect, refine, and save.

The user site tracks GitHub's latest non-draft, non-prerelease release and records
its exact tagged source commit in `docs-source.json` under channel `stable`.
Prepare synchronization on a separate branch, update every provenance field
together, and merge its PR only after the corresponding release is published
and the documentation checks and deployment preview have been reviewed.
Do not describe a later development commit as the stable release.

Copy generated assets from the release archive into `public/generated/` in
fieldwork-docs in the same synchronization change as the content and source
metadata. Run `npm run format` followed by `npm run validate`. Keep these updates
in a reviewable PR before publishing the documentation site. No site deployment is
part of package builds. Do not substitute developer docs for the user journey.

Fonts are build-only assets under `scripts/fonts/`; their OFL license is included
in source distributions. Runtime SVG/HTML remain dependency-free and use the
client's sans-serif font. Bitmap baselines use explicit fonts to avoid platform
font drift.
