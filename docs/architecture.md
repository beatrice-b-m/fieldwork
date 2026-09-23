# Architecture

Fieldwork is a dataframe-in/result-out library. Runtime dependencies are pandas
and NumPy. Optional rendering tools used by release builds are development
requirements only. Python 3.11–3.14 are supported, with a committed universal uv lock.

## Follow one investigation

`explore(df)` calls path search, availability analysis, and a bounded single-key
dependency search, plus bounded value-pattern analysis. It returns an overview containing the individual saved results
(the paths section includes census previews of its recommendations). Its compact text view presents families, signatures, candidate
grains and recommended paths together. Shared parameters (`features`, `by`,
entity settings, `scope`, `missing`, `table_id`) flow to every section, and
`options={"paths": {...}, ...}` configures sections independently. The overview
ranks the sections' findings as leads that reference them rather than copying them.
`profile(df, dimensions)` composes levels, census, grain and pairs for chosen
dimensions, each receiving the same scope, sentinels and table ID.

Every analysis calls `evidence.prepare` to identify the ordered dataset, apply a
scope, encode needed values, and compute native/sentinel availability without modifying
source values. A private call-scoped runtime shares prepared data and fingerprints
across nested components, reports optional progress, and checks cancellation.
No cache survives the top-level call; see [performance controls](performance.md). Findings carry table-qualified feature references and bounded source
positions. Presentation consumes saved evidence, never the original dataframe.

## Module map

| Module | Responsibility |
| --- | --- |
| `_explore/encoding.py`, `_kernels.py` | Column labels, native value codes, exported JSON values, counting kernels |
| `_explore/census.py` | Independent levels, bounded ordered observed prefixes |
| `_explore/grain.py`, `grain_graph.py` | Exact FDs, scope compatibility, equivalence, DAG |
| `_explore/relations.py`, `roles.py` | Pair contexts/absence, joint counts, schema suggestions |
| `_explore/profile.py` | `profile`: levels, census, grain and pairs for chosen dimensions |
| `result.py` | The `Result` model shared by every analysis: export, sections, findings, inspection, selection, recomputation |
| `evidence.py`, `_selection.py` | Scopes, fingerprints, source preparation, findings, targeted source selection |
| `_runtime.py`, `progress.py` | Call-scoped reuse, progress events/display, cancellation |
| `availability.py` | Presence signatures, families, implications, entity summaries |
| `discovery.py` | Supplied search bounds, exact/approximate/conditional FDs |
| `navigation.py` | Deterministic beam search and objective-specific prefix costs |
| `families.py`, `leads.py` | Typed feature relationships and lead ranking |
| `patterns.py` | Populated strings/numbers, indexed families, context constancy |
| `overview.py` | `explore`: overview sections, ranked leads, feature network |
| `workflow.py` | Recipes and delivery comparisons |
| `presentation.py` | Public renderers: project once, then draw |
| `_present/project.py` | One allowlisted projection per result kind (full or topology) |
| `_present/text.py`, `svg.py`, `html.py` | One renderer per medium, one function per kind or figure |

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
