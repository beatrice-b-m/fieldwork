# Fieldwork implementation specification

Status: initial specification, 2026-09-16.

This document captures the project direction and current-term implementation
proposal discussed with the project owner. It is temporary development
documentation. Update it as implementation decisions are made, move descriptions
of delivered behavior into `docs/`, and retire superseded planning material.

## Purpose and product principles

Fieldwork helps researchers build a working map of previously unseen or unmapped
data. It identifies patterns in populated values, feature availability, topology,
entity grain, and relationships, and makes those patterns easy to investigate.
The Python package and import name is `fieldwork`.

The central workflow is: discover patterns, inspect evidence, refine the scope or
question, and pursue explanations. Users bring domain knowledge and research
judgment. Output should state findings and their evidence directly, without
repetitive warnings about whether observed patterns establish real-world semantics.
Method definitions and interpretive context belong in durable documentation.

Operational information remains part of the result: evaluated populations,
exclusions, computation coverage, undefined measurements, and output truncation
must remain inspectable. Express these as useful facts rather than generic caution.

Prioritize individual-table exploration first, followed by related-table analysis.
Maintain the convenient Python/notebook experience of the existing explorer.

## Starting point and migration boundary

The existing implementation is in the sibling `bea-tools` repository under
`bea_tools/_explore/`, with tests in `bea_tools/testing/explore/`, benchmarks, and
worked examples. No source code is to be copied during repository initialization.
Migration is a subsequent implementation task.

Existing capabilities to preserve and migrate deliberately:

- `levels()`: independent feature counts and distributions.
- `census()`: observed prefixes along explicitly ordered dimensions, with bounded
  expansion and reported omitted mass.
- `grain()`: exact functional dependencies for supplied single/composite candidates,
  including support, violations, equivalence, and coarse-to-fine grain graphs.
- Pair relationships, contextual pair analysis, joint counts, and absence summaries
  using empirical or caller-declared domains.
- `infer_schema()`: role suggestions with supporting evidence.
- `explore()`: composition of the analytical operations.
- Typed, versioned, strict-JSON-compatible results, explicit population accounting,
  and scope lineage.
- Plaintext, SVG, standalone interactive HTML, and visualization-data projections.
- Full and topology-only presentation modes, including existing export semantics.

Retain the analytical distinctions and evidence contracts, and carry over relevant
tests, benchmarks, and examples. The standalone core should use pandas and NumPy
without inheriting unrelated DICOM or sampling dependencies. Audit packaging and
source attribution during extraction. Migration compatibility for existing
`bea_tools` imports/accessors is a separate decision; no compatibility shim is
specified here.

## Current-term capabilities

### 1. Availability and missingness patterns

Treat feature availability and populated values as related analytical views.
Implement results for:

- Per-feature availability and repeated row availability signatures.
- Groups of fields with identical or similar availability.
- Directional presence relationships: A populated implies B populated, including
  non-exact relationships and their exceptions.
- Mutually exclusive fields or feature groups.
- Conditional availability by selected contexts such as site, modality, source,
  version, or period.
- Availability within entities: populated on all, one, or some rows per entity.

Report counts, denominators, pattern-specific measurements, and representative
examples and exceptions. Distinguish co-presence from agreement caused primarily
by co-absence. Keep always-populated and always-missing features visible even when
a particular association measurement is undefined.

Support column-specific missing-value conventions, including declared sentinel
values. Preserve source values and record the analysis convention. Analyze native
missing values by default; do not automatically reinterpret arbitrary sentinel
values as missing.

Counting units should include rows, distinct entities, and within-entity summaries.
Define the aggregation and denominator explicitly for each measurement so that
entities with many rows do not silently determine every ranking.

### 2. Populated-value patterns and feature families

Extend the existing dependency and pair evidence incrementally with:

- Exact and approximate mappings, aliases, and feature equivalence.
- Features constant within an entity or selected context.
- Common combinations and subgroup concentrations.
- String format, prefix, and length patterns.
- Repeated/indexed column families.
- Numeric ranges, quantization, and simple offset or ratio relationships.

Feature families may draw on availability, value overlap, dependencies, and names.
Expose which evidence contributes to each family. Naming similarities are useful
inputs alongside data evidence. Preserve distinctions between family membership
and equivalent analytical behavior.

Prioritize missingness and existing dependency extensions before broad numeric or
format discovery. The latter are incremental expansion areas, not prerequisites
for the first usable milestone.

### 3. Candidate grain and dependency discovery

Extend supplied-key evaluation with discovery of single and bounded-size composite
candidates. Report uniqueness, repetition, determined features, repeated-group
support, and alternative candidates. Distinguish a unique row identifier from a
repeated grouping useful for understanding entity structure.

Add approximate and conditional dependencies with named measurements and directly
inspectable exception groups. Preserve exact relationships in the exact grain
graph; present approximate relationships distinctly. Retain equivalent and
cross-cutting candidates rather than forcing all structure into a single tree.

Discovery must have explicit work limits, such as maximum determinant size and
candidate count. Record what was searched and evaluated. Select precise algorithms
and approximation metrics during implementation and document their definitions.

### 4. Census path recommendations

Add a dedicated path-recommendation operation, provisionally `suggest_paths()`.
Allow `explore(df)` to use discovery when dimensions are omitted. Explicit
dimensions continue to provide a direct, predictable census path.

Support different browsing objectives:

| Objective | Desired behavior |
| --- | --- |
| Entity structure | Place coarse groupings before finer entities and attributes |
| Availability families | Expose major populated/missing record configurations |
| Compact overview | Favor useful early splits with manageable branching |
| Target investigation | Bring forward dimensions distinguishing a target's values or availability |
| Context comparison | Start from a selected site/source/time or other context |

A proposed first implementation:

1. Identify constants, equivalent features, candidate identifiers, and families.
2. Derive a partial order from supported nesting relationships.
3. Search a bounded set of candidate paths using a deterministic heuristic.
4. Rank paths using observed prefix structure, branching, redundant steps, and the
   requested objective and display budget.
5. Return several distinct paths with explanations, component measurements, and
   small previews.

For example, when `exam_id` determines `site`, a coarse-to-fine path normally puts
`site` before `exam_id`. Equivalent features can be attached as aliases;
cross-cutting dimensions can yield alternative paths.

Reordering the same dimensions preserves the full joint distribution and complete
observed combination count. Optimize the intermediate prefixes and visibility,
not an order-invariant total-information score. Do not use cardinality sorting as
the entire recommendation algorithm.

Provide lightweight steering constraints such as `start_with`, `before`, `exclude`,
and `target`. Keep one consistent dimension order per recommended census in the
first implementation; branch-specific adaptive trees are a separate extension.

## User journey and API direction

1. **Orient:** a compact overview presents feature families, candidate grains,
   major availability signatures, and suggested census paths.
2. **Choose:** select a concrete finding or path for investigation.
3. **Inspect:** show evidence, typical examples, exception groups, and access to
   relevant source rows.
4. **Refine:** compare contexts or narrow the scope and reuse it across analyses.
5. **Save:** export results, reproducible analysis recipes, and optional research
   notes; reapply recipes to later dataset deliveries.

Preserve dataframe-in/result-out functions as the primary interface. Illustrative
API only; these names and arguments are proposals, not implemented contracts:

```python
import fieldwork as fw

overview = fw.explore(df)
availability = fw.missingness(df, by=["site", "modality"])
paths = fw.suggest_paths(df, objective="structure")
tree = fw.census(df, dimensions=paths.best.dimensions)
dependencies = fw.discover_dependencies(df, max_key_size=2)
```

An optional session object may later manage reusable scopes and cached encodings;
it should delegate to the same analytical functions. Avoid expanding the combined
entry point into an unstructured collection of every operation's options.

Separate population selection, discovery/computation budgets, and display budgets.
Retain direct access to individual analyses. Readable feature names and values
should be the default for interactive inspection, with compact references
available for serialization and linking.

## Result and inspection contracts

Continue typed, versioned, JSON-compatible results and add convenient tabular
projections. Findings should include:

- Pattern type, readable statement, and involved features.
- Scope, missing-value convention, counting unit, and evaluated population.
- Support and named pattern-specific measurements.
- Typical examples and exceptions, with their selection method and limits.
- A reproducible selector or recipe for further inspection.
- Method parameters and search/computation coverage where relevant.

Do not replace these measurements with a universal confidence score. Coverage,
exception rate, repeated-group support, and contrast against a baseline answer
different questions.

Row inspection must handle duplicate dataframe indexes. Define whether selectors
refer to positions, supplied row keys, or predicates, and associate them with the
source dataset/version used by the analysis. Exporting a finding should not require
embedding all source rows. Renderers should continue to operate on saved evidence
without access to the dataframe.

Preserve topology-only exports for the existing context-sharing workflow. New
finding types need corresponding presentation behavior and tests.

## Related-table extension

Include table identity in internal feature references from the outset, while
keeping single-table calls simple. A later interface should accept named frames.
Extend evidence with:

- Candidate relationships from value inclusion and overlap.
- Single and composite join candidates.
- One-to-one, one-to-many, and many-to-many multiplicity.
- Unmatched values/records and predicted join expansion.
- Grain and feature availability before and after joins.

Automatic multi-table discovery is outside the first single-table milestone.

## Implementation sequence and acceptance

### A. Extract the exploration foundation

Migrate the existing explorer, relevant tests, examples, and benchmarks into the
standalone package. Establish package metadata and installation instructions.
Verify analytical result parity on representative fixtures and preserve typed
values, scope accounting, grain graphs, and full/topology rendering. Deliberate
presentation changes, including removal of repeated disclaimers, should be tested
and documented rather than treated as analytical regressions.

### B. Availability discovery and inspection

Implement signatures, availability groups, directional relationships, contextual
summaries, and exception inspection. Verify against small hand-checkable fixtures
covering co-presence, co-absence, mutually exclusive groups, sentinel conventions,
duplicate indexes, and unequal entity sizes. Demonstrate the workflow in a notebook.

### C. Connect discovery to census navigation

Add bounded candidate discovery and deterministic path recommendations. Test known
nesting, equivalence, cross-cutting dimensions, constraints, and branching tradeoffs.
Document objective-specific ranking measurements and show multiple useful paths.

The first substantial user milestone is: open an unfamiliar dataframe, discover a
feature family, inspect its exceptions, and generate a useful census without
having to supply dimensions first.

### D. Expand investigative depth

Add conditional and approximate dependency discovery, richer populated-value
patterns, saved recipes, and comparisons across contexts or deliveries. Introduce
related-table exploration after the single-table workflow is established.

Use synthetic examples with known structure, differential checks where applicable,
and representative performance fixtures. Measure end-to-end analyses and result
size, including wide and missingness-heavy data. Output limits and computation
limits should remain separately observable.

## Decisions to resolve during implementation

- Final public names and configuration objects for discovery, scopes, and recipes.
- Exact metrics, ranking priorities, thresholds, and default search budgets.
- Selector persistence and dataset identity for saved investigations.
- Packaging versions, license/attribution details, and migration compatibility.
- Whether optional discovery engines are useful after profiling the native core.

Relevant references include [missingno](https://github.com/residentmario/missingno)
for availability visualizations and
[Desbordante](https://github.com/Desbordante/desbordante-core) for dependency and key
discovery. These are references, not selected runtime dependencies.
