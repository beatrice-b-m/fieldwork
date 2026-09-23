# Architecture

Fieldwork is a dataframe-in/result-out library. Runtime dependencies are pandas
and NumPy. Optional rendering tools used by release builds are development
requirements only. Python 3.11–3.14 are supported, with a committed universal uv lock.

## Follow one investigation

`explore(df)` calls path search, availability analysis, and a bounded single-key
dependency search, plus bounded value-pattern analysis. It returns an overview containing individual saved results and
a census preview. Its compact text view presents families, signatures, candidate
grains and recommended paths together. `explore(df, dimensions)` delegates to the original composition
API; explicit dimension order remains authoritative. Common context settings are
applied to a private normalized frame and source accounting is retained in all
derived scopes; incompatible search options are rejected. `discovery={...}` steers the
implicit path search; common `features`, `scope`, `missing`, and `table_id` parameters
also flow into overview evidence.

Discovery modules call `evidence.prepare` to identify the ordered dataset, apply a
scope, encode needed values, and compute native/sentinel availability without modifying
source values. A private call-scoped runtime shares prepared data and fingerprints
across nested components, reports optional progress, and checks cancellation.
No cache survives the top-level call; see [performance controls](performance.md). Findings carry table-qualified feature references and bounded source
positions. Presentation consumes saved evidence, never the original dataframe.

## Module map

| Module | Responsibility |
| --- | --- |
| `_explore/encoding.py`, `_kernels.py` | Canonical scalar identity and counting |
| `_explore/census.py` | Independent levels, bounded ordered observed prefixes |
| `_explore/grain.py`, `grain_graph.py` | Exact FDs, scope compatibility, equivalence, DAG |
| `_explore/relations.py`, `roles.py` | Pair contexts/absence, joint counts, schema suggestions |
| `_explore/result.py` | Foundation result model |
| `_explore/visual_data.py`, `render.py`, `graphics.py` | Foundation projections and renderers |
| `evidence.py`, `_selection.py` | Discovery results, scopes, fingerprints, targeted source selection |
| `_runtime.py`, `progress.py` | Call-scoped reuse, progress events/display, cancellation |
| `availability.py` | Presence signatures, families, implications, entity summaries |
| `discovery.py` | Supplied search bounds, exact/approximate/conditional FDs |
| `navigation.py` | Deterministic beam search and objective-specific prefix costs |
| `families.py` | Typed feature relationships linked to section findings |
| `patterns.py` | Populated strings/numbers, indexed families, context constancy |
| `workflow.py` | Composition, recipes, delivery comparisons |
| `presentation.py` | Discovery projections and dispatch to foundation renderers |
| `_html.py` | Shared offline document, responsive styles, filtering and evidence-link controls |

## Extension boundaries

New patterns should add named measurements, population accounting, an example
selection limit, and a presentation test for full and topology exports. Add an
operation to the explicit Recipe allowlist only when its arguments serialize to
JSON and can be safely reapplied to a new frame. No arbitrary Python evaluation is
used for recipes.

Related-table automatic discovery, adaptive branch-specific census orders, persistent user-managed
sessions, and external discovery engines remain future extensions. Table identity
is present in new feature references; foundation references are local to the
source's `table_id`. The current public APIs analyze one dataframe at a time.

## Extraction

The foundation, regression tests, benchmark and grain example came from bea-tools
commit `4e4f1704cbca91a314652b0c99f86d8f4a2c4f90`; see `NOTICE`. Imports were
rewritten, accessors omitted, branding updated, and repeated interpretive disclaimers
removed. Quantitative evidence, populations, and topology disclosure behavior remain.
No DICOM, sampling dependencies, or compatibility shims were migrated.

Overview discovery also accepts `by`, `entity`, `unit`, and `entity_presence`.
Context columns flow to availability, dependency and value-pattern sections.
Entity settings apply to availability; census path scores and exact/approximate
value dependencies continue to count rows. Each linked finding identifies its unit.
