# Development and validation

```bash
uv sync --locked
uv run pytest
uv run ruff check src tests scripts examples
uv run ruff format --check src tests scripts examples
uv run python examples/investigation.py
uv run python scripts/generate_assets.py --check
uv build
```

The project uses the [uv build backend](https://docs.astral.sh/uv/concepts/build-backend/)
with a `src` layout. The package metadata caps supported Python at `<3.15`; CI tests
3.11, 3.12, 3.13 and 3.14. pandas 2.2+ permits Python 3.11; uv selects compatible
versions for each interpreter. pandas 3.x requires Python 3.11 or newer.

Foundation tests preserve typed scalar, scope accounting, exact graph, omission,
rendering and strict-JSON semantics, including Hypothesis differential oracles.
Discovery fixtures check co-absence, sentinels, unequal entities, duplicate indexes,
modal exceptions, conditional/composite dependencies, constraints and saved exports.
The notebook executes as part of validation with IPython; no external data is used.

## Profile representative workloads

```bash
uv run python benchmarks/foundation.py --suite smoke --repeats 3 --output /tmp/fieldwork-foundation-benchmarks
uv run python benchmarks/discovery.py --output /tmp/discovery.json
```

Timings and serialized result bytes are measured separately from correctness checks.
Do not assert wall-clock thresholds in unit tests. Fingerprinting traverses the
entire source frame; dependency graphs compare pairs of candidate keys. For wider
frames, narrow features and candidate budgets before introducing an optional engine.
Use the reported coverage to distinguish a budget boundary from a negative finding.

## Change discipline

Keep coherent changes in scoped commits. Update developer contracts here and the
user journey/reference pages in fieldwork-docs when interfaces change. Regenerate
assets and inspect PNGs for clipping, layout, and useful evidence. Treat intentional
presentation changes separately from analytical regressions. Plans belong in
`temp-docs/`; retire them once behavior is described in durable documentation.
