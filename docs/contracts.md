# Evidence contracts

## Results and serialization

Foundation `ExplorerResult` uses schema 0.3, retaining the migration contract.
Discovery `InvestigationResult` uses schema 1.0. Both are mappings with `to_dict()`
exports compatible with `json.dumps(..., allow_nan=False)`. Foundation scalar values
are tagged; integer and float identities remain distinct, nonfinite values use
string encodings, and native missing scalars share a missing token. Results have
frozen top-level attributes, but nested payload containers are mutable.

Discovery currently requires unique string column names. Foundation operations
also support integer and recursively tuple-valued labels. No arbitrary object
stringification is used to merge values. When discovery selects columns
automatically (`features=None`), columns containing unsupported cell types (such
as lists, dicts or `Decimal`) are skipped and listed in `skipped_features` with
their value type; the other columns are analyzed normally. A column named
explicitly (in `features`, `by`, `entity`, or a foundation operation) still raises
`TypeError`.
Foundation context (`scope`, `missing`, `table_id`) preserves those labels. When
columns include typed labels, `analysis_context.missing_convention` stores
`sentinels_by_column` records with tagged `column` identities and sentinel `values`,
so JSON exports preserve integer and tuple labels without converting them to strings.

`InvestigationResult.from_dict` restores schema 1.0 saved evidence. `to_frame()`
normalizes findings; pass a section such as `availability`, `dependencies`,
`candidates`, or `changes` to project another list. `PathResult.best` is `None` if
no path exists, otherwise use `best.census(df)` to preserve recommendation context.

## Population, source and scope

Each discovery source has a SHA-256 fingerprint over canonical ordered column
labels, index labels and all cell values. Dataframe dtype metadata is not part of
the identity. Duplicate indexes are allowed. Inspection compares the fingerprint
and selects with `.iloc`, returning a copy. Reordering or changing source values
invalidates inspection. Identical rows are analytically interchangeable.

Use `Scope.from_positions(df, positions, name=...)`; positions are unique,
nonnegative, in bounds, and normalized to source order. `refine` requires a subset
of its parent's source positions and records the parent name. Saved scopes include selected source positions, not source cell values. Analysis reports
input rows, evaluated rows and restrictions separately. A scope is not an implicit
sample: search budgets and display limits never change its row population.

Examples and exceptions contain at most `example_limit` positions (default 5),
selected in source order. `total`, `omitted`, `limit`, and selection method accompany
each. `inspect` returns those saved representative rows, not every matching row;
`result.inspect(df, finding_id, all_matches=True)` recovers the entire matching
source population. `result.select(df, finding_id, exceptions=False, name=...)`
returns that population as a reusable `Scope` with parent lineage. Selection
evaluates the saved predicate directly on the verified source, with saved scope and
missing conventions; no display limit changes the matching population. Overview
findings resolve through their owning section. Unsupported future selector types
can fall back to section recomputation. Recomputation applies to individual discovery sections, not an overview/comparison. Dependency exception groups
also report omitted groups and rows. Findings never embed entire source rows.

## Missing conventions

Native missing values count as absent by default. `missing={column: [sentinels]}`
adds column-specific sentinels without altering the source. Integer and float
sentinels compare by numeric value (so `-999` matches a float column containing
`-999.0`); booleans remain distinct. Other sentinel types use canonical identity.
The declared conventions are saved in results. Dependency discovery defaults to
pair-specific complete cases (`dropna=True`); false treats native and declared
missing values as a shared category. Path previews apply the same missing convention.

Row-level presence metrics always name their denominator. Entity summaries group
by populated entity keys; incomplete keys are excluded and counted. Each distinct
entity has equal weight. `all` and `one` overlap for a singleton populated entity;
`some` means populated on at least one but fewer than all rows. Conditional
availability groups include missing context values as categories.

## Presentation and disclosure

Full exports retain numerical evidence. Topology projections allowlist structural
labels and qualitative relations, discard measurements, row positions, dataset
identifiers, scope counts, and distribution values, and canonicalize ordering of
new findings. Topology does not anonymize feature names or value labels already
present in structural foundation outputs. Renderers escape HTML/SVG and terminal
control characters. Display truncation is reported separately from search coverage.
Topology retains qualitative analysis units: rows or entities, entity keys, and
any/all presence aggregation. Each finding and feature-network connection carries
its own unit, so an overview can distinguish entity availability from row-based
dependencies and value patterns without disclosing quantities.

Standalone HTML uses native disclosure controls for discovery findings and the
migrated interactive graph/census controls for foundation results. No server,
network resources, or dataframe are needed to view an export.

## Recipes and comparison

`Recipe(operation, parameters, notes)` stores strict JSON arguments and version 1.0.
`save`/`load` round-trip JSON; `run` calls an allowlisted operation. Recipes reapply to
new deliveries, while source-bound positions remain in results/scopes. Supply a
scope as a run override, not as a persisted recipe parameter. Composite grain
`KeySpec` objects and non-JSON sentinels cannot be stored directly in recipes.
Automatic overview recipes accept top-level `scope`, `missing`, `table_id`, and
`features` run overrides. These replace the corresponding `discovery` settings
while preserving other discovery configuration, including search budgets,
constraints, entity aggregation and context grouping. The recipe is not modified.
Search-only path options still belong in `discovery`; `sections` and
`section_options` configure independent overview components. Runtime controls are
run overrides, never saved recipe parameters.
`compare` compares two missingness results by feature name, preserving both source
identities and conventions. Its delta is after minus before populated fraction;
empty populations and added/removed features have an undefined delta.
Saved comparisons retain `before_scope`, `after_scope`, `before_analysis_unit`,
and `after_analysis_unit`, including each scope's name, positions and lineage and
each unit's denominator and exclusions. The existing `scope` and `analysis_unit`
fields continue to describe the after population. Presentations label both sides;
topology keeps scope names/parent names and qualitative units, omitting positions
and quantities.

## Recommendation handoff

Use `paths.best.census(df)` (or `paths.path(i).census(df)`) to retain the exact
source, scope and sentinel conventions of a recommendation, including after JSON
restoration. `path.dimensions` is only an ordered tuple: it does not carry context.
The handoff validates source identity and rejects context overrides. To choose a
new population, rerun discovery. `census(df, dimensions, scope=..., missing=...)`
also supports explicit context. Derived foundation scopes account against the
original source, separating scope restrictions from missing-value exclusions.
Context adaptation updates dataset metadata only at result roots and analytical
section roots; graph `source`/`target` references retain their node identities.
Shared population records are rebased once by object identity. In pre-filter
exploration, census scopes reused by pair or grain lineage retain the original
input total, disjoint exclusions, and a single added scope-lineage entry.

With explicit dimensions, `explore` accepts common `scope`, `missing`, `table_id`
and `features` settings in `discovery`; search-only settings raise `ValueError`.
Duplicate settings in `discovery` and explicit options are rejected.

## Availability units and selectable patterns

`missingness(..., unit="rows")` remains the default, even with `entity=` supplied.
`unit="entities"` requires entity keys. Each distinct populated key has equal
weight. `entity_presence="any"` (default) aggregates presence across its rows;
`"all"` requires presence on every row. Incomplete keys are excluded and counted.
All availability metrics, families, signatures, similarity and implications use
the chosen unit. `analysis_unit` records aggregation, denominator, eligible source
rows and missing-key exclusions. Context analyses aggregate within each context;
an entity spanning contexts contributes once to each relevant context.

The denominator of conditional presence is antecedent-populated units; Jaccard
uses either-populated units. Signature counts use units; omission metadata reports
both omitted units and their source rows. Entity summaries always classify raw-row
presence as any/all/one/some/none. Singleton all/one overlap intentionally.

Signatures and whole context summaries reference finding IDs. Selecting a context
summary returns every eligible source row in that context. Whole entity summaries
select all eligible entities; individual patterns narrow to matching entities.
Context availability and each entity presence
pattern are also findings, with typed structural predicates, bounded examples and
selectors. Entity selections return **all source rows** belonging to matching
entities (within the analyzed scope/context), including rows where the feature is
absent. Examples and exception totals always count source rows, separately from
unit support. Comparisons require compatible units and entity aggregation; recipes
persist these settings. HTML renders named evidence tables and source selections.

## Runtime controls and optional envelopes

[Performance controls](performance.md) specify progress event ordering, phase ETA,
cooperative cancellation, call-scoped cache lifetimes, independent overview sections,
and dependency/graph budgets. Defaults preserve the analytical payload. Explicit
omissions carry coverage or `not_requested` status and remain visible in presentations.

`to_dict(compact=True)` returns a versioned `fieldwork.compact` envelope preserving
all evidence through shared-container references. Both result classes accept their
ordinary and compact exports in `from_dict`; ordinary schemas 0.3/1.0 are unchanged.
Compact exports do not apply topology disclosure filtering.

## Editor-visible Python API

Public functions and members carry NumPy-style docstrings and explicit type
annotations. Runtime controls are declared in source; census handoff signatures
exclude replacement source context, and grain cache parameters are private.
`explore` overloads distinguish automatic discovery from explicit dimensions.
`fieldwork.typing` provides documented dictionary types for reusable configurations.
See the [inline API standard](inline-api.md) for the public boundary and validation.

### Additive dependency support fields (discovery schema 1.0)

Dependency tables and finding measurements retain all existing meanings and add
observed target coverage and consistency within repeated determinant groups.
See [the population definitions](algorithms.md#dependency-target-coverage-and-repeated-support).
Counts are nonnegative Python integers; undefined ratios serialize as JSON null.
Candidate global test counters and supported-exact lists disclose evaluated work,
not completeness. Graph budgets remain independent. Source identity, selection,
exception repair rows, finding IDs and analytical candidate enumeration are unchanged.
`to_frame('dependencies')` includes tests below the finding threshold; compact and
ordinary exports retain these fields. No new public parameters, exports, foundation
schema, or result type hierarchy are introduced.

Full presentations adapt saved schema-1.0 records without a source or mutation.
When available, evaluated rows minus (evaluated groups minus repeated groups)
recovers repeat rows, and repair rows recover repeat-only accuracy. Global records
recover supported-exact lists. Missing determinant eligibility never becomes
invented target coverage: legacy coverage is displayed as unavailable. If even one
candidate's supported-exact ranking input cannot be recovered, the entire collection
retains legacy ranking. The adapter does not change ordinary or compact exports.

Full text/HTML/SVG explanations distinguish observed exactness, observed target
coverage and repeat-only consistency, and disclose missing-category evaluation.
These qualifications never enter shared statements or structures. Topology keeps
existing structural roles/relationships, adds no measurements or evidence pointers,
and orders candidates canonically, independently of support ranking.
