# Architecture

Fieldwork is a dataframe-in/result-out library. Runtime dependencies are pandas
and NumPy. Optional rendering tools used by release builds are development
requirements only. Python 3.11–3.14 are supported, with a committed universal uv lock.

## Follow one investigation

`explore(df)` calls path search, availability analysis, and a bounded single-key
dependency search. It returns an overview containing individual saved results and
a census preview. Its compact text view presents families, signatures, candidate
grains and recommended paths together. `explore(df, dimensions)` delegates to the original composition
API; explicit dimension order remains authoritative. `discovery={...}` steers the
implicit path search; common `features`, `scope`, `missing`, and `table_id` parameters
also flow into overview evidence.

Discovery modules call `evidence.prepare` to identify the ordered dataset, apply a
scope, encode values, and compute native/sentinel availability without modifying
source values. Findings carry table-qualified feature references and bounded source
positions. Presentation consumes saved evidence, never the original dataframe.

## Module map

| Module | Responsibility |
| --- | --- |
| `_explore/encoding.py`, `_kernels.py` | Canonical scalar identity and counting |
| `_explore/census.py` | Independent levels, bounded ordered observed prefixes |
| `_explore/grain.py`, `grain_graph.py` | Exact FDs, scope compatibility, equivalence, DAG |
| `_explore/relations.py`, `roles.py` | Pair contexts/absence, joint counts, schema suggestions |
| `_explore/result.py`, `resolved.py` | Foundation result model and readable references |
| `_explore/visual_data.py`, `render.py`, `graphics.py` | Foundation projections and renderers |
| `evidence.py` | Discovery results, scopes, fingerprints, source inspection |
| `availability.py` | Presence signatures, families, implications, entity summaries |
| `discovery.py` | Supplied search bounds, exact/approximate/conditional FDs |
| `navigation.py` | Deterministic beam search and objective-specific prefix costs |
| `patterns.py` | Populated strings/numbers, indexed families, context constancy |
| `workflow.py` | Composition, recipes, delivery comparisons |
| `presentation.py` | Discovery projections and dispatch to foundation renderers |

## Extension boundaries

New patterns should add named measurements, population accounting, an example
selection limit, and a presentation test for full and topology exports. Add an
operation to the explicit Recipe allowlist only when its arguments serialize to
JSON and can be safely reapplied to a new frame. No arbitrary Python evaluation is
used for recipes.

Related-table automatic discovery, adaptive branch-specific census orders, cached
sessions, and external discovery engines remain future extensions. Table identity
is present in new feature references; foundation references are local to the
source's `table_id`. The current public APIs analyze one dataframe at a time.

## Extraction

The foundation, regression tests, benchmark and grain example came from bea-tools
commit `4e4f1704cbca91a314652b0c99f86d8f4a2c4f90`; see `NOTICE`. Imports were
rewritten, accessors omitted, branding updated, and repeated interpretive disclaimers
removed. Quantitative evidence, populations, and topology disclosure behavior remain.
No DICOM, sampling dependencies, or compatibility shims were migrated.
