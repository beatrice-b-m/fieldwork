# What you can rely on

Fieldwork describes one delivery of a table. Its results are observations about
the rows it was given, not claims about the process that produced them. This page
lists what every result guarantees; [algorithms](algorithms.md) defines each
measurement and [performance controls](performance.md) the budgets and runtime
controls.

## Every result has the same shape

- Every analysis returns a `fieldwork.Result`: a read-only mapping over its
  payload with a `kind` and schema version `2.0`. `to_dict()` is plain JSON
  (`json.dumps(..., allow_nan=False)` works) and `Result.from_dict` restores it;
  a restored result renders, inspects and selects exactly like the live one.
  Nested payload containers are ordinary mutable lists and dictionaries.
- Every result records its `source` (`dataset_id`, `table_id`, `rows`,
  `columns`), its `scope` (below), its `missing_convention` and, for single
  analyses, the `parameters` that produced it, with work budgets under
  `parameters["limits"]`.
- `findings` are the statements worth reading. Each has an `id`, a `pattern`, the
  `features` it involves, `measurements`, bounded `examples` and `exceptions`,
  and a `selector` that finds its rows again. `to_frame()` tabulates findings;
  `to_frame("availability")`, `to_frame("dependencies")` and similar tabulate
  other lists.
- An overview (`explore`) keeps each section's result under `sections` and ranks
  their findings as `leads`. `overview.findings` resolves the leads, in rank
  order, to the section findings with overview IDs `f0`, `f1`, ...; `inspect` and
  `select` accept those IDs, and `section(name)` returns one section as a
  `Result`. A profile (`profile`) keeps its levels, census, grain and pairs
  results under `sections` the same way.

## Values and columns

- Exported values are plain JSON: `null` for every native missing spelling,
  numbers as numbers, infinities as `"inf"`/`"-inf"`, and dates, datetimes
  (aware ones as UTC instants) and timedeltas as text that pandas parses back.
- Booleans, numbers and strings are never the same value. In object columns the
  integer `1` and the float `1.0` are different levels: mixed numeric
  representations are themselves worth seeing. Displays quote a string only when
  it could be read as a number, boolean or missing value (`code='1'`).
- Columns are identified by `str(label)`. Integer labels (a headerless CSV) and
  tuple labels are analyzed and reported by their string form, which must be
  unique. When an analysis chooses columns itself (`features=None`), columns with
  unsupported cells (lists, dicts, `Decimal`) are skipped and listed in
  `skipped_features`; a column you name explicitly raises `TypeError` instead.

## Which rows were counted

- `scope` states the analyzed population: `input_rows`, `evaluated_rows`,
  `restriction_excluded_rows` and, for a `Scope`, its `name`, `parent` and
  `selection_positions`. Create one with `Scope.from_positions(df, positions)`
  or `result.select(...)`; `refine` narrows it and records the parent. Scopes
  store row positions, never cell values.
- Every nested measurement (a level, the census tree, a pair, a dependency test,
  a grain graph) states its own `evaluated_rows` and `missing_excluded_rows`,
  plus `restriction_excluded_rows` where a context or a census pre-selection
  removed rows. Every fraction names its denominator; availability's are listed
  below.
- Missing means native missing values plus the sentinels you declare with
  `missing={column: [values]}`; the source is never modified. Integer and float
  sentinels match numerically (`-999` matches `-999.0`); others match by their
  exported value, so a saved convention applies again unchanged.
- Budgets (`limits`) and display limits bound work and output; they never sample
  rows or change a population. Work skipped by a budget is counted in `coverage`
  (or marked `not_requested`) and shown by every renderer. An untested pair or
  context is not evidence of anything.
- With `top_n_mode="pre"`, a profile's census pre-selection is an explicit Scope
  named "census top_n cohort" that pairs (and, with `top_n_applies_to="both"`,
  grain) analyze; the census tree counts the rows it removed.

## Going back to the rows

- A result is bound to its source by `dataset_id`, a SHA-256 fingerprint of the
  ordered column labels, the index and every value, with dtype as part of the
  identity. Duplicate index labels are fine. Reordering rows or changing a value,
  label or dtype makes `inspect`, `select`, `recompute` and `Path.census` refuse
  the frame. Fingerprints from 0.1.x do not match.
- `examples` and `exceptions` hold at most `example_limit` source positions
  (default 5), first in source order, with `total` and `omitted`.
  `result.inspect(df, finding_id)` returns those rows;
  `all_matches=True` returns every matching row.
- `result.select(df, finding_id, exceptions=False)` evaluates the finding's saved
  predicate on the verified source, under the saved scope and missing convention,
  and returns the matching rows as a `Scope`. No display limit changes that
  population. Entity findings select **all rows** of the matching entities,
  including rows where the feature itself is absent.
- `result.recompute(df, **overrides)` reruns a single analysis on its verified
  source with its saved scope and conventions; overviews and comparisons are
  recomputed section by section.
- `paths.best.census(df)` (or `paths.path(i).census(df)`) keeps a recommended
  path's source, scope and sentinels, also after JSON restoration, and rejects
  context overrides. `path.dimensions` alone is only a tuple of names.

## Availability: units and denominators

- `missingness` counts rows by default (`unit="rows"`), even with `entity`
  given. `unit="entities"` gives each distinct populated entity key equal weight;
  `entity_presence="any"` counts an entity as populated when any of its rows is,
  `"all"` when every row is. Rows with an incomplete key are excluded and
  counted. `analysis_unit` records the unit, aggregation, denominator and
  exclusions, and every availability measurement uses that unit.
- "A implies B" is measured over units where A is populated; similar presence
  (Jaccard) over units where A or B is. Signatures count units and report
  omitted units with their source rows. Vacuous evidence (always-present
  targets, identical columns) is not reported as a finding.
- Context analyses aggregate within each joint context value; a missing context
  value is its own category. Each context availability finding states whether
  all, some or none of the context's units are populated
  (`structure["presence"]`), so topology exports keep that state. Entity
  summaries classify each entity as any/all/one/some/none of its rows populated;
  for a single-row entity, `all` and `one` overlap. The summary counts every
  pattern; a pattern that matches no entity is not a finding.

## Dependencies

- A dependency test is exact when every determinant group has one target value,
  and approximate when the modal value covers at least `min_accuracy` of rows.
  Approximate findings are stated as "X approximately determines Y", so every
  output, including plain-text topology, keeps the distinction.
  With `dropna=True` (default) each test uses its own complete cases; with
  `False`, missing values are a category. Each test reports its populations,
  its target coverage and how consistent repeated determinant groups are, since
  singleton groups satisfy exactness trivially
  ([definitions](algorithms.md#dependency-target-coverage-and-repeated-support)).
- `dependencies` keeps every completed test, including those below the finding
  threshold. Candidate counters describe work done, not completeness.
- Topology exports keep candidates' structural roles in canonical order, without
  measurements.

## Reusing settings on a new delivery

- `Recipe(operation, parameters)` stores strict-JSON arguments; `save`/`load`
  round-trip them and `run(df)` applies them to another frame. Run overrides
  (`scope`, `missing`, `table_id`, `features` and runtime controls) replace saved
  values without changing the recipe. A `Scope` belongs to one source, so it is
  a run override, never a saved parameter.
- `compare(before, after)` matches two missingness results by feature name and
  keeps both sources, scopes and analysis units. Its delta is after minus before
  populated fraction; it is undefined for empty populations and added or removed
  features. Both results must use compatible units.

## Sharing results

- `render_plaintext`, `render_svg`, `render_html` and `visualization_data` read
  only the saved result: no dataframe, server or network access is needed, and
  HTML/SVG and terminal control characters are escaped.
- `detail="topology"` keeps feature names, structural labels, qualitative
  relations and analysis units, and drops measurements, row positions, dataset
  identifiers and scope counts. It is a disclosure filter, not anonymization:
  names and value labels can still identify people or sites.
- Display truncation is reported separately from search coverage.
- Fieldwork's own error messages name columns, arguments and value types, never
  source cell values.
