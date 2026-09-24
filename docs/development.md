# Development and validation

```bash
uv sync --locked
uv run pytest
uv run pyright --warnings
uv run ruff check src tests scripts examples benchmarks
uv run ruff format --check src tests scripts examples
uv run python examples/investigation.py
uv run python scripts/generate_assets.py --check
uv build
```

The project uses the [uv build backend](https://docs.astral.sh/uv/concepts/build-backend/)
with a `src` layout. The package metadata caps supported Python at `<3.15`; CI tests
3.11, 3.12, 3.13 and 3.14. pandas 2.2+ permits Python 3.11; uv selects compatible
versions for each interpreter. pandas 3.x requires Python 3.11 or newer.

## Principles

- Prefer deleting code to adding configuration; every option needs a real analyst
  use case.
- Keep the guarantees an analyst relies on: analyses never mutate the source,
  budgets never silently change the analyzed population, and every finding can be
  traced back to its source rows ([contracts](contracts.md)).
- Tests assert analytical behavior (counts, relationships, populations,
  selection) or valuable contracts (round trips, source-mismatch rejection), not
  prose, CSS or docstring layout.
- Regenerate the assets (`scripts/generate_assets.py`) and the notebook outputs
  (`scripts/execute_notebook.py`) whenever output changes, and look at the images.

## Test suite

Tests assert analytical behavior (counts, relationships, populations, selections)
and structural contracts (round trips, source-mismatch rejection, bounded
exports). They never assert sentence text, CSS or docstring layout, so wording
and presentation can change without editing tests. `uv run pytest` runs the
suite and every docstring example in `src/fieldwork` in under about 10 seconds;
`uv run pytest -m "not slow"` skips the realistic-scale smoke test.

- **Independent oracles.** `tests/oracle.py` recomputes levels, census prefixes
  and dependency tests with plain Python and pandas groupby, importing no
  production code. Hypothesis frames mix types and every native missing
  spelling (None, NaN, `pd.NA`, NaT): `foundation/test_differential.py` checks
  levels and census, `discovery/test_dependency_oracle.py` checks every
  dependency record, exception group, finding and candidate summary with
  composite keys, `dropna` both ways, contexts and scopes.
- **Known answers and properties.** `foundation/test_pairs.py` (relation classes,
  Cramér's V, absence classes), `foundation/test_census.py` (all census options as
  properties), `discovery/test_value_patterns.py` and `discovery/test_compare.py`.
- **Relations across tables.** `discovery/test_relate.py` checks coverage,
  cardinality, joined rows, agreement and both sides' selections against the
  enumeration oracle in `tests/oracle.py`, over Hypothesis frames with mixed key
  types, composite keys and both match modes. Reciprocity has its own edge oracle.
- **Invariances.** `discovery/test_native_dtypes.py` requires categorical,
  nullable, pyarrow-backed, tz-aware and float32 columns to give the same evidence
  as their object equivalents. `discovery/test_permutation.py` requires row order
  never to change evidence, only relabel saved positions.
- **Rendering contracts.** `tests/test_rendering_contracts.py` checks every result
  kind in every medium once: valid self-contained output, no mutation, escaping,
  unique HTML IDs, and topology output that depends on structure, not quantities.
- **Scale.** `discovery/test_scale.py` (marked `slow`) runs the overview on a
  50,000-row wide export with planted structure and bounded saved positions.

Use private kernels in tests only where a property oracle checks them (as for
`modal_groups` and `exact_pair_ids`); otherwise test through public functions.
Prefer a mutation check (temporarily breaking the code) when a new property test
passes on its first run.

The notebook executes as part of validation with IPython; no external data is used.
Its checks verify the conclusions stated in the narrative, not incidental numbers,
and agreement with the Python companion's initial and corrected deliveries. After
editing notebook cells, refresh the checked-in text, table, and SVG outputs with:

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

## Qualify output interactions and responsive layouts

The optional browser harness requires a development installation of Playwright
with Chromium. It does not add a package runtime dependency. Use the package's
Python environment for fixtures and Node for browser checks:

```bash
uv run python scripts/output_ux_fixtures.py /tmp/fieldwork-output-ux
node scripts/check_output_ux.cjs /tmp/fieldwork-output-ux
```

If Playwright is installed outside Node's module search path, set
`FIELDWORK_PLAYWRIGHT` to its absolute package directory. The harness opens only
the generated local files. It writes screenshots and `browser-results.json` beside
them, fails on viewport overflow, clipped SVG text, script errors, or external
requests, and exercises filters, fragment links, matrix scrolling, keyboard
selection, reset, census branches, and JavaScript-disabled disclosures.

See the [output UX qualification](evaluation/output-ux.md) for fixtures, viewports,
scales, results, and limitations. Inspect screenshots as well as automated bounds:
geometry alone cannot establish that a sticky header is painted above its rows.

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
Because each operation runs in a subprocess, patching fieldwork in the calling
process has no effect on the measurement; to compare implementations, run the
script from two source trees (`PYTHONPATH=<tree>/src`).
`--fixture structured` includes unique IDs, repeated entities/contexts, and nested
missingness masks. `--progress` records callback counts and inclusive phase
durations (nested phases overlap; do not sum them).

For comparable overview components, its `dependencies` workload uses 20
single-column candidates and `patterns` uses 20 pairs. `--candidates` changes
standalone dependency/path budgets; for `explore` it changes path search only
(`options={"paths": {...}}`). Per-section `options` can configure dependency work
independently in application code. `--features` restricts analytical features but
does not project the source frame. `--profile /tmp/overview.prof` supports one
operation/repeat with cProfile; keep these diagnostic timings separate from
unprofiled measurements. The harness uses Unix `resource` RSS reporting (macOS
bytes, Linux KiB converted to bytes).

Timings and serialized result bytes are measured separately from correctness checks.
Do not assert wall-clock thresholds in unit tests. Fingerprinting traverses the
entire source frame; dependency graphs compare pairs of candidate keys. For wider
frames, narrow features and candidate budgets before introducing an optional engine.
Use the reported coverage to distinguish a budget boundary from a negative finding.

## Inline documentation and typing

Follow the [inline API standard](inline-api.md) for public entry points, members,
returned objects, and option dictionaries. The regular pytest run executes public
docstring examples (`--doctest-modules` over `src/fieldwork`) and a smoke test that
every public callable has a docstring and annotated, underscore-free parameters.
`uv run pyright --warnings` checks the strict consumer examples in `tests/typing`,
including expected invalid calls and result navigation.

After packaging, check the installed wheel independently of the source import:

```bash
uv run --no-project --isolated --with ./dist/*.whl --with pytest python -I -m pytest tests/test_public_api.py --doctest-modules --pyargs fieldwork -q
```

CI runs this check on each supported Python version in addition to the existing
installed investigation example. Update overloads and `fieldwork.typing` option
fields together with implementation signatures and docstrings.

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

[Current performance controls](performance.md) document runtime APIs and what is
reused within a call; [implementation measurements](performance-results.md) record representative
before/after runs. `tests/discovery/test_performance_contracts.py`, `test_runtime.py`,
and `test_scaling_controls.py` check fingerprint change detection, vectorized group
oracles, scoped/entity selections, population-exact graph comparisons, progress
ordering, cancellation, callback errors, session cleanup, omitted work and
saved-export round trips. Internal call sequences are deliberately not tested.

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

The [dependency support qualification](evaluation/dependency-support.md) records
versioned before/after semantic cases, graph digests and cost samples. Run
`benchmarks/evidence_support.py --output /tmp/support.json` with the current
Python environment. For comparisons across the additive support extension only,
`benchmarks/parity.py --allow-evidence-support-extension` removes the explicit new
fields from discovery records; all legacy fields, graphs, selectors and analytical
orders still have to match. Default parity remains exact. The standalone
`examples/dependency_support.py` supplies the generated dependency-support assets.
