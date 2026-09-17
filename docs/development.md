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
uv run python benchmarks/scaling.py --rows 10000 --columns 150 --operations missingness paths dependencies explore --output /tmp/scaling.json
```

`scaling.py` measures each operation in a fresh process, with a per-process
deadline (`--timeout`, default 300 seconds). It records fixture construction,
analysis, JSON serialization, result bytes, and absolute process peak RSS
separately. Peak RSS includes fixture construction and serialization; it is not
incremental analysis memory. Timeouts include startup and fixture construction.
Use `--repeats 3` for multiple measurements, `--fixture dense` or `--fixture mixed`
to vary data characteristics, and `--rows 300000` for a larger case. The default
sparse fixture uses 70% missingness and eight populated values per column.
`--fixture structured` includes unique IDs, repeated entities/contexts, and nested
missingness masks. `--progress` records callback counts and inclusive phase
durations (nested phases overlap; do not sum them). `--compact` measures the
optional shared-container JSON envelope, separately from analysis.

For comparable overview components, its `dependencies` workload uses 20
single-column candidates and `patterns` uses 20 pairs. `--candidates` changes
standalone dependency/path budgets; for `explore` it changes path search only,
matching the backward-compatible `discovery` configuration. New `section_options`
can configure dependency work independently in application code. `--features` restricts analytical features but
does not project the source frame. `--profile /tmp/overview.prof` supports one
operation/repeat with cProfile; keep these diagnostic timings separate from
unprofiled measurements. The harness uses Unix `resource` RSS reporting (macOS
bytes, Linux KiB converted to bytes).

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

## Performance regression workflow

[Current performance controls](performance.md) document runtime APIs and cache
bounds; [implementation measurements](performance-results.md) record representative
before/after runs. `tests/discovery/test_performance_contracts.py`, `test_runtime.py`,
and `test_scaling_controls.py` check canonical fingerprint bytes, vectorized group
oracles, scoped/entity selections, cache populations/lifetimes/bounds, cancellation,
callback errors, ETA/throttling, omitted work and compact round-trips.

For a revision-to-revision default-output check, run the same parity script with
both source trees and identical pandas/NumPy versions:

```bash
PYTHONPATH=/tmp/baseline/src .venv/bin/python benchmarks/parity.py --output /tmp/before.json
.venv/bin/python benchmarks/parity.py --output /tmp/after.json --compare /tmp/before.json
```

The corpus covers 100 seeded scoped/unscoped cases, duplicate indexes, sentinels,
entity/context summaries, composite/conditional dependencies, all scored path
objectives and full overviews. It compares all exported analytical fields, not
only top-level counts. Keep parity runs separate from performance measurements.
