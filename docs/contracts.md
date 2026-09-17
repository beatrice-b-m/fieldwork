# Evidence contracts

## Results and serialization

Foundation `ExplorerResult` uses schema 0.3, retaining the migration contract.
Discovery `InvestigationResult` uses schema 1.0. Both are mappings with `to_dict()`
exports compatible with `json.dumps(..., allow_nan=False)`. Foundation scalar values
are tagged; integer and float identities remain distinct, nonfinite values use
string encodings, and native missing scalars share a missing token. Results have
frozen top-level attributes, but nested payload containers are mutable.

Discovery currently requires unique string column names. Foundation operations
also support integer and recursively tuple-valued labels. Unsupported cell types
raise `TypeError`; no arbitrary object stringification is used to merge values.

`InvestigationResult.from_dict` restores schema 1.0 saved evidence. `to_frame()`
normalizes findings; pass a section such as `availability`, `dependencies`,
`candidates`, or `changes` to project another list. `PathResult.best` is `None` if
no path exists, otherwise its `dimensions` can be passed directly to `census`.

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
`result.recompute(df, example_limit=len(df))` replays saved parameters, missing
conventions and scope to recover all examples before calling `inspect` again.
Recomputation applies to individual discovery sections, not an overview/comparison. Dependency exception groups
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

Standalone HTML uses native disclosure controls for discovery findings and the
migrated interactive graph/census controls for foundation results. No server,
network resources, or dataframe are needed to view an export.

## Recipes and comparison

`Recipe(operation, parameters, notes)` stores strict JSON arguments and version 1.0.
`save`/`load` round-trip JSON; `run` calls an allowlisted operation. Recipes reapply to
new deliveries, while source-bound positions remain in results/scopes. Supply a
scope as a run override, not as a persisted recipe parameter. Composite grain
`KeySpec` objects and non-JSON sentinels cannot be stored directly in recipes.
`compare` compares two missingness results by feature name, preserving both source
identities and conventions. Its delta is after minus before populated fraction;
empty populations and added/removed features have an undefined delta.
