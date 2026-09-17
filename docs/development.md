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
Its checks also verify the numerical conclusions in the narrative and agreement
with the Python companion's initial and corrected deliveries. After editing notebook
cells, refresh the checked-in text, table, and SVG outputs with:

```bash
uv run python scripts/execute_notebook.py
```

The output refresher uses the existing IPython dependency and the notebook's explicit
`display`/`print` calls; it does not need a Jupyter server. Review the saved outputs
and adjacent interpretations together. A failed cell leaves the previous file intact.

`tests/discovery/test_journeys.py` follows scoped recommendation handoffs, sparse
grain views, typed conditional contexts, complete signature/context/entity
selection, unequal entity weights, candidate classification, diverse path reasons,
and linked feature relationships through saved exports. The executable example
also saves/loads a recipe in a temporary directory, compares the initial and corrected
deliveries using newly selected scopes, and restores saved exception evidence.

After graph or path search changes, run the seeded discovery benchmark. Its
population-compatible views retain more evidence than a single intersected graph;
inspect result bytes as well as runtime. Narrow candidate and feature budgets for
wide frames. At most one view is built per distinct supported candidate mask.

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

The README and documentation homepage use `wide-table` assets generated from
`examples/wide_table.py`. This seeded synthetic laboratory table has 960 rows and
43 columns: participant, specimen, instrument, assay, run and specimen–assay
attributes are flattened alongside technical replicates. Six explicit candidate
keys yield a branching observed grain graph; the renderer's edges and attribute
placements come from `grain()`, not a hand-drawn schema. Replicate numbers and raw
signals intentionally remain unplaced by those candidates. To inspect it alone:

```bash
uv run python examples/wide_table.py /tmp/fieldwork-wide-table
```

Asset generation also bundles the executable source as `wide-table.py`, so the
documentation site can offer the exact example alongside its SVG/HTML/JSON exports.
