# Architecture

`src/fieldwork/_explore/` contains the extracted analytical foundation: typed scalar
encoding, population accounting, independent levels, ordered census, exact grain
graphs, pairs, role suggestions, and saved-evidence renderers. Public functions
are exported from `fieldwork`; no bea-tools compatibility accessor is installed.

Run `uv sync --locked` and `uv run pytest`. Build distributions with `uv build`.
The runtime imports only pandas, NumPy and the standard library.

Extraction provenance is recorded in `NOTICE`; foundation regression and
differential tests live in `tests/foundation/`.
