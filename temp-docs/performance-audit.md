# Large-dataframe performance and progress audit

Date: 2026-09-17. Baseline: `039d88c2ec33bedc11572348c43fb5118f41d641`.
Scope: the public analytical tools, overview composition, evidence inspection,
memory use, search budgets, and runtime feedback. This audit adds reproducible
measurements and recommendations; it does not change analytical behavior.
Retire or refresh this snapshot when the identified paths are refactored; keep
the reusable benchmark instructions in `docs/development.md` synchronized.

## Assessment

Fieldwork has an efficient foundation for bounded, explicitly selected analyses,
but its newer discovery orchestration repeatedly performs expensive preparation
and Python-level row processing. A 300,000-row, 150-column frame contains 45 million
cells. Several seemingly bounded operations still traverse all of them, sometimes
many times. Search/output budgets are not general runtime or memory budgets.

There is no progress callback, runtime progress display, cancellation API, or ETA
in the analytical implementation. Existing text, HTML, and SVG renderers consume
completed results; they cannot indicate progress during a synchronous call.
Both the computational bottlenecks and observability need attention.

## Measurement protocol

`benchmarks/scaling.py` runs each operation in a fresh subprocess, records analysis
time separately from fixture construction and JSON serialization, and applies a
process deadline. A timeout covers interpreter startup, fixture construction,
analysis, and serialization; it is not an exact lower bound on analysis time alone.
Peak RSS includes the interpreter, fixture, computation, and serialization, rather
than measuring incremental analytical allocation. Runs are sequential to avoid
benchmark contention. cProfile measurements are collected separately because they
add overhead. These are local diagnostic measurements, not universal latency SLAs.

The main fixture uses seed 721, 150 float64 columns, eight populated values per
column, and independent 70% missingness. It intentionally exercises distinct
availability masks and signatures. A dense fixture removes missingness; a mixed
fixture alternates high-cardinality numeric values, repeated strings, and low-
cardinality numbers. No real user data is inspected. The benchmark's dependency
operation uses the overview's 20 single-column candidates; the standalone API's
default is 100 candidates and a maximum key size of two.

Measurements and environment details are retained in `performance-results/`.

Environment: Python 3.11.14, pandas 3.0.5, NumPy 2.4.6, macOS 26.5.1 ARM64,
Fieldwork 0.1.0. CPU model and RAM size were not available through the sandbox's
hardware query. Initial measurements below are single fresh-process runs, rounded
to avoid implying more precision than the protocol supports.

| Operation | 10,000 × 150 | 300,000 × 150 | Configuration |
| --- | ---: | ---: | --- |
| Full-source fingerprint | 2.23 s | 66.32 s | All values and ordered labels |
| Preparation, including fingerprint | 2.21 s | — | All columns |
| Missingness | 3.86 s | 122.19 s | Default row units, 200 pairs |
| Value patterns | 2.26 s | 67.82 s | 20 pairs, matching overview |
| Path suggestions | 10.16 s | 180 s timeout | Default search, three previews |
| Dependency discovery | 20.81 s | 180 s timeout | 20 single-column candidates |
| Default overview | 37.69 s | 180 s timeout | `fw.explore(df)` |
| Levels | — | 0.36 s | All 150 columns, five displayed levels |
| Census | — | 0.02 s | First four dimensions, 40 nodes, eight levels |
| Pairs | — | 0.07 s | First four dimensions, six pairs |
| Joint counts | — | 0.02 s | First pair |
| Schema inference | — | 0.24 s | All columns, no candidate keys |
| Explicit grain | — | 1.37 s | Two supplied single-column keys, all 150 columns |

The separate fingerprint and preparation runs differ slightly due to normal
run-to-run variation; preparation includes the hash and is not intrinsically
cheaper. Subsecond foundation timings apply to this low-cardinality fixture and
the stated bounded queries, not arbitrary continuous/string data or the same
analytical work as a complete overview.

The three large discovery timeouts were separate runs, not one shared deadline.
No full completion time or peak RSS is claimed for those interrupted operations.
They were terminated after 180 seconds of total child-process wall time. In the
overview, paths run first, so the timeout does not demonstrate that its later
sections were reached. Seven hashes at the isolated large-frame throughput would
alone represent roughly 7.7 minutes; this is a cost projection, not a measured
overview runtime or a reliable ETA for another dataframe.

The 1,000 × 150 overview took 5.67 s. At 10,000 × 150 the overview serialized to
11.92 MB, of which the standalone dependency result was 9.71 MB with 20 grain
views. Serialization itself took only 0.05 s for that overview, so JSON encoding
was not its main time bottleneck. At 300,000 × 150 missingness peaked at 1.64 GB
process RSS and retained 300,000 distinct signatures while displaying 50. Its
serialized output was only 0.865 MB. Small output does not imply small working
memory. Foundation peak RSS was about 0.80 GB, mostly reflecting construction of
the 360 MB fixture plus the constructor's transient storage.

### Profile and optimization experiments

A separate cProfile run of the 1,000 × 150 overview confirmed **seven** calls to
both `fingerprint` and `prepare_context`, **20** calls to `grain`, **4,056** calls
to `encode_series`, and **5,960** foundation `_fd_record` calls, in addition to
2,980 discovery dependency tests. Of the encoder calls, **3,006** used its per-row
fallback. The profiled run took 16.92 s versus 5.67 s without profiling; use its
call counts to understand repetition, not its elapsed time as normal performance.
`profile-summary.json` retains cumulative timings, which overlap and must not be
added together.

On the same seeded 10,000-row data, restricting overview `features` to the first
20 columns while retaining the 150-column input took **20.03 s**. Constructing
only those same first 20 columns and running the default overview took **6.22 s**.
The initial full-width/all-feature overview took 37.69 s. The projected and
feature-restricted analyses use the same selected values, but source identity
and source metadata differ; projection is not an identity-preserving replacement.

The isolated kernel experiment measured the following at 10,000 × 150:

| Fixture | Current fingerprint | Dictionary/chunk prototype | Exact hash equality |
| --- | ---: | ---: | --- |
| Sparse, eight populated values | 2.245 s | 0.066 s | Yes |
| Dense, eight values | 2.920 s | 0.068 s | Yes |
| Mixed repeated strings/low-cardinality numbers/continuous numbers | 2.785 s | 1.417 s | Yes |

This is approximately 34× and 43× for the low-cardinality hashing kernels, but
only 2× for the mixed fixture. It is **not** a measured end-to-end speedup or a
guarantee on high-cardinality data. Replacing the singleton row-mask reductions
with reuse of the precomputed arrays removed about 1.56–1.58 seconds per fixture;
all arrays compared equal. The dictionary construction itself takes microseconds,
but that is not a useful whole-operation speed claim. Neither experiment modifies
the library, and full semantic regression coverage is still required before
adopting these changes.

## Bottlenecks and recommended changes

### 1. Repeated full-source fingerprinting and preparation

[`evidence.fingerprint`](../src/fieldwork/evidence.py) normalizes every cell and
index/column label, creates a typed dictionary, serializes it to JSON, and updates
SHA-256. This is Python work per cell even for a column with only eight values.
The homogeneous-column fast path in `encode_series` does not accelerate this hash.

`prepare_context` fingerprints the original frame before applying a scope, then
selects rows and encodes **every column**, regardless of the operation's `features`
or `max_features`. Restricting features reduces downstream tests but does not
reduce full-source hashing or full-width preparation. A small source-bound scope
also retains the full-source hash cost.

[`workflow.explore`](../src/fieldwork/workflow.py) independently invokes paths,
missingness, dependency discovery, and value patterns. Each prepares again.
[`suggest_paths`](../src/fieldwork/navigation.py) additionally calls `census` for
each returned preview with `missing=missing or {}`. Even the empty dictionary
activates `foundation_context`, which fingerprints and encodes the entire source,
copies a normalized frame, and converts every column to object dtype. With three
previews, one overview performs **seven full fingerprints and preparations**.
That is 315 million data-cell visits for hashing alone at the target size, plus
index/label visits, encoding, normalization, and the actual analyses.

**Priority:** introduce a private prepared-analysis context reused within a single
top-level call: source identity, original positions, canonical column codes,
dictionaries, native/sentinel masks, and population metadata. Encode only the
union of required columns. Build previews directly from that context. Preserve
independent result dictionaries so one section cannot mutate another's findings.

Optimize the fingerprint independently: for repeated values, serialize each
canonical unique value once and stream its bytes in original row order, in
bounded chunks. Preserve the existing digest exactly if feasible. Changing hash
representation requires an explicit version/migration policy because saved scopes
and inspection depend on identity. Do not substitute an untyped hash that conflates
`True`, `1`, `1.0`, missing values, or reordered/duplicate indexes. Do not cache by
`id(df)` across calls: pandas frames are mutable. A later reusable session must
define snapshot ownership or explicit invalidation.

### 2. Missingness performs per-cell reductions on singleton arrays

[`availability.missingness`](../src/fieldwork/availability.py) creates one Python
list per row, then `masks_for` invokes `np.any(present[c][rows])` for every unit and
column. With the default `unit='rows'`, each array has only one element. At the
target size this means **45 million tiny NumPy calls** to reconstruct masks that
preparation already computed. Context summaries repeat the same pattern.

Use the existing presence arrays directly for row units, and indexed views/copies
for context rows. For entity units, factorize entity keys once and use grouped
counts/reductions; implement `any` as count > 0 and `all` as count == entity size.
Retain missing-key exclusions, entity weights, and context-local aggregation.

Signatures currently build a 150-element Python boolean tuple for every row,
retain all distinct tuples, and sort every signature before showing at most 50.
For independent sparse columns, almost every row has a different signature. The
tuple reference slots alone can approach 360 MB at 300,000 × 150, before tuple
headers, dictionaries, and row lists. Packed bits need about 5.7 MB for the same
raw 150-bit signatures. Factorize packed signatures and retain counts and dense
group IDs; preserve the current frequency and lexical tie ordering. Packing must
be chunked or planned to avoid an unnecessary second full boolean matrix.

`emit` also builds and sorts complete source-position lists before `finding`
retains only five examples. Count from masks, and retrieve just the first bounded
source positions while preserving total/omitted accounting. Full selectors should
have a dedicated retrieval path, rather than forcing example construction to
materialize every match.

### 3. Dependency discovery repeats grouping for every target

[`discover_dependencies`](../src/fieldwork/discovery.py) groups eligible rows in
Python dictionaries for each candidate × target × population, builds a `Counter`
per group, then constructs lists of typical, exceptional, and affected rows.
The same determinant is grouped repeatedly for different target columns.

At 150 columns, the overview's 20 single-column candidates produce **2,980 tests**
globally; standalone defaults produce **14,900**. With complete data that is
894 million and 4.47 billion eligible row visits, respectively, before the inner
counter/list work. Missingness reduces eligible rows, but adds distinct population
masks and grain views. Contexts multiply the number of test records by up to 33;
their rows are disjoint partitions, so their total row visits should not be
described as 33 full-frame scans. Many tiny groups still add considerable overhead.

Factorize determinants once, use dense group/target pair counts, compute modal
counts and violations in arrays, and reuse grouping for matching eligibility
masks. Preserve smallest-canonical-code modal ties, target-specific complete-case
denominators, singleton evidence, and representative exceptions. Avoid retaining
all typical/exception positions when only counts and bounded examples are needed.
Unique IDs are especially costly in the current Python implementation: they can
create one `Counter` and several lists per row per target, rather than eight
groups as in the main fixture. Proven unique determinants and constant targets
also permit exact shortcuts, provided their evaluated-population and group
metrics remain correct. The sparse low-cardinality timing is not a worst case.

Graph construction repeats exact FD work already measured by discovery. Each
distinct supported candidate mask can trigger another `grain` call; each call
re-encodes all selected columns. There can be up to one view per candidate, with
overlapping membership. Reuse compatible exact evidence and intern population masks;
do not collapse views with different evaluated populations. An explicit lazy or
optional graph phase would let users request dependency evidence first.

**Confirmed dtype-inference penalty:** on the installed pandas 3.0.5,
`infer_dtype(series.array, skipna=True)` returns `floating` for a float64 column
but `unknown-array` after the graph/preview normalization converts it to object
dtype. `infer_dtype(series.to_numpy(), skipna=True)` recognizes the latter's
actual floating values. Since `unknown-array` is outside the encoder's homogeneous
allowlist, normalization defeats its factorization fast path. The profile's
3,006 fallback calls are 3,000 graph-column encodes plus six preview dimensions.
This is avoidable per-cell canonicalization even on repeated numeric data.
Prefer passing existing typed codes into these consumers; also correct container
inference with differential tests for truly mixed object values, where ordinary
pandas factorization can conflate boolean/integer/float identities. Do not simply
add `unknown-array` to the homogeneous allowlist.

The foundation [`grain`](../src/fieldwork/_explore/grain.py) retains an N-element
boolean evaluated mask per key/target pair. With 300,000 rows and complete data,
20 keys × 149 targets is roughly **894 MB of boolean masks alone**, even though the
masks are identical; 100 keys is roughly 4.47 GB. This is a structural allocation
estimate, not a measured peak. Intern repeated masks/use population IDs and avoid
unnecessary copies. Key-to-key comparisons and target assignment can add quadratic
candidate work, especially when many keys determine the same targets.

More generally, the target-sized numeric dataframe occupies about 360 MB before
analysis. One full-width int64 code dictionary adds another 360 MB; one boolean
presence array per column adds 45 MB. Preparation selects a full row-position
array and creates a private frame; missing-code remapping allocates another set
of integer arrays transiently. Object conversion for graph/preview preparation
can add substantial boxed-value storage. Optimizations should therefore reduce
simultaneously live representations as well as CPU time. These byte counts use
decimal MB and exclude Python/container overhead.

### 4. Path scoring repeats prefix work and rich explanations

[`navigation.suggest_paths`](../src/fieldwork/navigation.py) limits search to 20
features and 200 extensions by default, but each extension constructs Python
tuples across all rows for every prefix. Its cache stores the final measurements
for an entire path, not the grouping arrays for shared prefixes. Availability
impurity is computed at every prefix even for the default structure objective,
where it contributes explanations but does not affect ranking. Pair inference
also builds Python sets from full-column `.tolist()` conversions.

Cache dense prefix group IDs or bounded beam states, extend them using exact pair
factorization, and compute group counts from arrays. Reuse prefix measurements
across paths; compute explanatory-only impurity for retained recommendations
after ranking. Target/availability objectives still need their respective impurity
during scoring. Avoid caching every N-row state indefinitely: the search cache
itself needs a memory bound. Reuse prepared context for the selected previews.

### 5. Patterns, inspection, and exports have additional traps

[`value_patterns`](../src/fieldwork/patterns.py) runs regexes, lengths, and prefixes
over every populated string. Use unique strings and frequencies when values
repeat, preserving first-observed ties. Numeric summaries sort distinct values,
so genuinely continuous columns cost more than the low-cardinality main fixture.
Its context-constancy code uses `rows not in constant` where `constant` is a list
of row lists. For many singleton context groups this can do quadratic group-list
comparisons per feature. Replace list membership with a constant-group boolean
array or set of group IDs. This path has no `max_contexts` limit.

[`InvestigationResult.inspect/select`](../src/fieldwork/evidence.py) rehashes the
source even to retrieve five saved examples. `select` validates again through
`recompute`, which prepares again, and replays the whole section with
`example_limit=len(df)`. That can materialize full examples for unrelated findings.
`inspect(all_matches=True)` introduces another validation before calling `select`.
Implement selector-specific evaluation, reusing one validated context per call.
Source mutation detection must remain intact.

Grain views serialize full source-position lists despite the usual example limit.
`exact_grain` also aliases the first view's grain result: sharing saves live object
memory but JSON writes it twice. The overview serializes both section findings
and its combined findings, and duplicates the best census preview. Consider an
explicit compact export with references, plus an independently versioned schema
if the existing format changes. Display truncation is not a result-memory bound.

## What is already efficient

- `encode_series` factorizes homogeneous columns before canonicalizing distinct
  values; this is a useful building block for the discovery refactor. Its mixed-
  object fallback intentionally normalizes each value to preserve scalar types.
- `levels` encodes selected features, and `census` encodes active dimensions and
  expands a bounded observed-prefix tree. They avoid the discovery identity pass
  unless the contextual census adapter is requested. Large cardinalities still
  require counting/sorting; `top_n` primarily limits output, not input scanning.
- `pairs` and `joint_counts` use exact integer pair IDs and `bincount`, with an
  overflow-safe fallback. Pair work is bounded; absence counts avoid constructing
  a full Cartesian product under default example limits. `joint_counts` explicitly
  rejects a supported-domain product above `max_cells`.
- `infer_schema` is column-wise and uses the encoder; supplying candidate keys
  additionally invokes full grain analysis. `Recipe.run` inherits its operation's
  costs, whereas `compare` works on saved availability summaries.
- Rendering and relationship filtering use saved results. They do not rescan the
  dataframe, although large serialized graphs/findings can still be expensive.

## Runtime feedback design

Add a common optional progress callback to public entry points, with a small
structured event rather than a dependency on a particular UI library:

```text
operation, phase, completed, total, unit, elapsed_seconds,
estimated_remaining_seconds (nullable), detail, terminal_status
```

Propagate one reporter through nested operations. An overview should show stages
for source validation/fingerprinting, encoding, availability, path search,
dependency tests, compatible grain views, patterns, and assembly. Emit a start
event before hashing: otherwise the first long silent phase remains unresolved.
Examples of useful counters are columns encoded, candidate/target tests completed,
extensions evaluated, and views built. A spinner with elapsed time is appropriate
while totals are unknown. Report search-budget exhaustion separately from
completion of all possible tests.

Provide an opt-in terminal/notebook adapter (a progress extra could use a mature
progress library), keeping the core callback dependency-free and silent by default.
Refresh at most about 4–10 times per second, update one display, and skip per-row
callbacks. Notebook display must update before blocking work starts. All exit
paths should close the display and distinguish completed, cancelled, and failed.
Keep progress events out of saved analytical evidence and recipe JSON.

**ETA must be calibrated, not a fixed row-count formula.** Start with elapsed time
and counts; estimate a phase only after observing representative work. Use rolling
rates stratified by determinant size, eligible rows, and observed group count for
dependency tests. Estimate hashing/encoding by processed cells or bytes, rather
than treating a short integer column like a long-string column. Search depth and
population views make remaining work data-dependent. Show an approximate range or
“estimating” until sufficiently stable, and reset/qualify the estimate when the
phase changes. Do not show 95% complete merely because 19 of 20 columns are done
if expensive graph work has not started. A full-run ETA needs explicit phase cost
weights; a stage bar plus elapsed time is more honest initially.

Check a cancellation/deadline token at bounded chunk or work-item boundaries.
Cooperative cancellation cannot interrupt a single long unchunked pandas/NumPy
call; document that latency. Default cancellation should raise a distinct outcome,
not return a result that appears complete. If partial results are later supported,
record completed/omitted work and incomplete sections explicitly.

## Prioritized implementation and acceptance

1. **Make work observable:** common reporter, stage counters, terminal/notebook
   adapters, elapsed time, and cancellation cleanup. No ETA until calibrated.
2. **Remove avoidable work:** row-unit missingness fast path, bounded example
   extraction, shared within-call context, and preview reuse. Keep these as small
   independent changes with output parity checks.
3. **Replace Python row grouping:** reusable determinant IDs, packed signatures,
   prefix grouping, vectorized modal counts, and the context-constancy membership
   fix. Add mask/cache memory budgets and preserve population contracts.
4. **Control expensive optional work:** independently configurable overview
   sections/graph building, graph-view/candidate-target budgets, and compact result
   export. New limits must expose coverage rather than silently omit evidence.
5. **Re-measure before choosing another engine or parallelism.** Python-heavy
   loops and redundant work come first. Process parallelism would otherwise copy
   large frames and multiply memory consumption; threads do not remove Python
   per-cell work. Consider an optional engine only behind exact semantic parity.

Use the existing differential/property tests for typed scalar identity, missing
sentinels, population-compatible graphs, duplicate indexes, stable ties, source
positions, scopes, and strict JSON. Add focused parity cases for the refactored
kernels, read-only source behavior, high-cardinality contexts, and repeated masks.
Progress tests should check ordering, monotonic counters within each phase,
nested propagation, throttling, cancellation/error closure, and absence from
serialized results, rather than asserting a particular UI string or wall time.

Performance acceptance should use several fresh-process repeats on 10k/100k/300k
rows and 20/150 columns, including sparse, dense, mixed, unique IDs, continuous
values, shared/nested missingness masks, and entity/context workflows. Record
median and spread, peak RSS, output size, and phase times. Do not put fragile
wall-clock thresholds in ordinary unit tests. Set concrete latency and memory
targets from the baseline and available hardware before implementing each phase.

The existing benchmark coverage is narrower: the foundation acceptance fixture
defaults to 150,000 rows but only six columns; discovery defaults to 2,000 rows
and 25 columns including an entity key, and tests just 12 dependency candidates.
It does not run the complete implicit overview. Those workloads can remain fast
while the 150-column discovery workflow becomes expensive.

## Immediate workarounds with the existing API

Call the specific operation you need. For an initial scan, use explicit features
with `levels`, and build a bounded census for known dimensions. For discovery,
physically project the frame before calling Fieldwork if full-source identity is
not required: `features=` alone does not eliminate preparation of other columns.
Keep using that projected frame for inspection; its source identity differs from
the original full dataframe.

```python
small = df.loc[:, relevant_columns]  # retain IDs/context columns you need
overview = fw.explore(
    small,
    discovery={"max_features": 12, "max_candidates": 40, "n_paths": 1},
)

# Control dependency work separately when this is the only question:
dependencies = fw.discover_dependencies(
    small, max_key_size=1, max_candidates=5,
)
```

In `explore`, `discovery['max_candidates']` controls **path extensions only**;
dependency discovery still uses its hard-coded 20 candidates, and missingness/
patterns have their own budgets. `min_accuracy`, `example_limit`, `max_signatures`,
and `max_patterns` mainly control output, not the full computation. Candidate and
pair coverage follows input-column order, so order meaningful candidate columns
first and inspect coverage. On 150 columns, the first 100 dependency candidates
are still single-column keys, despite `max_key_size=2`.

A representative sample is useful for initial suggestions if clearly labeled as
sample evidence. A dependency observed in a sample is not proven on the full
table; validate final candidates on the full relevant population. A `Scope`
preserves lineage but does not avoid hashing the full source; passing a sampled
frame does, at the cost of identifying a different source. No existing setting
enables an analytical progress bar.

## Reproduction

Run sequentially in the project's environment, with no other benchmarking jobs
competing for CPU/memory. The large suite deliberately records timeouts rather
than running expensive discovery without a bound.

```bash
uv run python benchmarks/scaling.py --rows 1000 --operations explore --timeout 180 --output /tmp/fieldwork-perf/overview-1000.json
uv run python benchmarks/scaling.py --rows 10000 --operations fingerprint prepare missingness paths dependencies patterns explore --timeout 180 --output /tmp/fieldwork-perf/discovery-10000.json
uv run python benchmarks/scaling.py --rows 300000 --operations levels census pairs joint_counts infer_schema fingerprint missingness patterns paths dependencies explore --timeout 180 --output /tmp/fieldwork-perf/large-300000.json
uv run python benchmarks/scaling.py --rows 1000 --operations explore --profile /tmp/fieldwork-perf/overview.prof --output /tmp/fieldwork-perf/profile-1000.json
uv run python benchmarks/scaling.py --rows 10000 --features 20 --operations explore --output /tmp/fieldwork-perf/features20-10000.json
uv run python benchmarks/scaling.py --rows 10000 --columns 20 --operations explore --output /tmp/fieldwork-perf/projected20-10000.json
uv run python benchmarks/scaling.py --rows 300000 --operations grain --timeout 180 --output /tmp/fieldwork-perf/grain-300000.json
uv run python temp-docs/performance-kernels.py > /tmp/fieldwork-perf/kernels.json
```

`performance-kernels.py` is an audit-only experiment. It compares the current
fingerprint against dictionary serialization with bounded byte chunks, and the
current singleton-array presence reduction against direct reuse. Equality checks
cover the three synthetic fixtures, not the complete supported scalar/index
contract; it is not a production-ready patch or an end-to-end speedup claim.
