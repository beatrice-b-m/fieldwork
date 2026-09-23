# Performance implementation measurements

Measured on 2026-09-17 with Python 3.11.14, pandas 3.0.5, NumPy 2.4.6,
and macOS 26.5.1 ARM64. The baseline is commit `d0a4d9f`; implementation
measurements cover the performance/runtime implementation committed as `4578127`.
Package version remains 0.1.0; no release or publication is implied.

Every timed operation runs in a fresh process. Fixtures use seed 721. After-change
figures are medians of three runs, with minimum/maximum and peak process RSS below.
The historical baseline has one run per case. Analysis excludes fixture creation
and JSON serialization; the raw files retain each separately. Process timeouts
include startup, fixture creation and export, so a timeout is not an exact analysis
duration. Peak RSS includes those stages and the input itself; it is **not**
incremental analysis memory. GB and MB here are decimal.

## Same sparse workloads, before and after

150 columns, eight populated float values, independently 70% missing cells.
Default overview budgets are preserved: 20 single-column dependency candidates,
2,980 candidate/target tests, 20 grain views on this fixture, 200 path extensions,
and 20 numeric-pattern pairs. Standalone component workloads use the matching
overview budgets described in `benchmarks/scaling.py`.

| Operation | 10k rows before | 10k rows after | 300k rows before | 300k rows after |
| --- | ---: | ---: | ---: | ---: |
| `fingerprint` | 2.235 s | 0.073 s | 66.325 s | 1.933 s |
| `prepare` | 2.213 s | 0.087 s | — | 2.194 s |
| `missingness` | 3.863 s | 0.093 s | 122.191 s | 2.289 s |
| `dependencies` | 20.807 s | 0.968 s | 180 s cutoff | 7.488 s |
| `paths` | 10.161 s | 0.128 s | 180 s cutoff | 3.003 s |
| `patterns` | 2.259 s | 0.093 s | 67.820 s | 2.137 s |
| `explore` | 37.689 s | 1.063 s | 180 s cutoff | 9.023 s |

The 10k default overview improves by about 35× in this fixture. At 300k,
missingness improves by about 53×; overview, dependency discovery and path search
now complete within the former 180-second cutoff. These are measurements of this
synthetic workload, not promises for arbitrary tables.

| 300k sparse operation | After median [min–max] | Maximum peak RSS | Ordinary JSON bytes |
| --- | ---: | ---: | ---: |
| `fingerprint` | 1.933 [1.930–1.934] s | 0.799 GB | — |
| `prepare` | 2.194 [2.180–2.196] s | 0.856 GB | — |
| `missingness` | 2.289 [2.279–2.292] s | 0.799 GB | 864,950 |
| `dependencies` | 7.488 [7.457–7.499] s | 1.351 GB | 23,217,493 |
| `paths` | 3.003 [2.993–3.017] s | 0.799 GB | 108,577 |
| `patterns` | 2.137 [2.136–2.140] s | 0.799 GB | 172,029 |
| `explore` | 9.023 [9.021–9.028] s | 1.455 GB | 25,447,388 |

Missingness peak process RSS falls from 1.637 GB to approximately 0.799 GB.
The fixture alone occupies 360 MB and construction briefly retains additional
arrays. The complete overview still has substantial memory and output costs:
its ordinary JSON is 25.45 MB. At 10k rows, every component's ordinary result byte
count matches the corresponding baseline count; semantic equality is separately
checked by the 100-case revision corpus.

## Foundation checks

These operations already avoided most discovery overhead. On the same 300k × 150
sparse input, their performance is retained. `levels` covers all columns;
`census` uses the first four dimensions and a 40-node budget; `pairs` uses those
four dimensions; `joint_counts` uses the first two; `grain` uses two candidate keys.
Peak process RSS remains 0.799–0.824 GB, dominated by fixture construction.

| Operation | Baseline seconds | After median [min–max] seconds |
| --- | ---: | ---: |
| `levels` | 0.355 | 0.354 [0.349–0.357] |
| `census` | 0.016 | 0.017 [0.017–0.018] |
| `pairs` | 0.067 | 0.065 [0.065–0.066] |
| `joint_counts` | 0.015 | 0.016 [0.016–0.016] |
| `infer_schema` | 0.244 | 0.243 [0.240–0.244] |
| `grain` | 1.368 | 1.309 [1.304–1.316] |

## Different data characteristics

All rows in this table run the same default `explore` workflow.

| Fixture | Shape | Analysis median [min–max] | Maximum peak RSS |
| --- | --- | ---: | ---: |
| Dense categorical | 300k × 20 | 3.792 [3.775–3.795] s | 0.318 GB |
| Mixed | 10k × 150 | 4.180 [4.149–4.184] s | 0.240 GB |
| Mixed | 100k × 150 | 39.299 [39.027–39.447] s | 0.817 GB |
| Mixed | 300k × 150 | 130.161 [129.950–130.434] s | 2.166 GB |
| Structured | 300k × 150 | 38.798 [38.757–38.880] s | 1.532 GB |

Dense categorical data has eight populated values and no missing cells. Mixed
has equal numbers of continuous normally distributed floats, eight repeated
strings, and eight-valued numeric columns, without missing cells. Structured has
a unique ID, an entity ID repeated every four rows, an eight-valued context, and
columns with shared/nested missingness masks and repeated values. The structured
overview treats all columns as candidate features; it does not automatically
activate entity aggregation or conditional analysis.

Continuous/unique values remain expensive: canonical source identity still visits
all cells, and value encoding needs exact identities and deterministic tie order.
The mixed results show why row count alone cannot provide a reliable whole-run
ETA. One additional bottleneck found during implementation—linear missing-code
lookups in every FD dictionary—was replaced with constant-time lookup. Shared
scalar dictionaries are now bounded; grain only retains equivalence codes and
missing-code metadata rather than every scalar object.

## Progress and compact exports

On the 300k sparse overview, enabling a collecting progress callback gives
9.154 s median analysis [9.089–9.162], versus
9.023 s without a callback. These separate runs do
not establish a precise overhead percentage. Each produced 208 events
including phase boundaries. Raw results include inclusive phase durations;
nested durations overlap and must not be summed.

The same runs exported compact JSON (an envelope removed after 0.1.2):
24,219,506 bytes versus 25,447,388 ordinary bytes, a 4.8% reduction. Median compact serialization was
0.473 s. Compact export is optional, preserves all evidence,
and does not change analytical work or solve input-memory costs.

One representative instrumented run shows the benefit of sharing preparation:

| Phase | Inclusive seconds |
| --- | ---: |
| fingerprinting | 2.015 |
| paths | 3.038 |
| missingness | 0.337 |
| dependencies | 5.573 |
| value patterns | 0.137 |

Fingerprinting occurs once, inside paths. Subsequent overview sections reuse it;
standalone calls include their own source validation. The dependency phase contains
both modal tests and grain-view work. These are nested durations, not additive
independent benchmarks.

## Validation and reproduction

- All **208 tests** pass locally. The suite covers typed scalar/fingerprint parity, exact and
  modal grouping, scope/sentinel/entity populations, source-order selections,
  missing masks and graph cache compatibility.
- New runtime checks cover event hierarchy, monotonic counts, throttling and
  conservative phase ETA, cancellation/deadline/error cleanup, bounded cache
  lifetimes, terminal/notebook adapters, omitted work and compact round-trips.
- `benchmarks/parity.py` compares 100 complete default JSON results against
  `d0a4d9f` using identical dependencies. All match, including conditional and
  composite dependencies, graph views, scoped source positions and path rankings.
- Lint/format checks, generated-asset verification, the seeded discovery benchmark,
  wheel/source builds, and the wheel investigation smoke check pass.
- Timing assertions are excluded from unit tests. Ordinary outputs and the
  default row population are unchanged; no sampling is introduced.

```bash
# Representative large baseline-compatible workload, after the changes:
.venv/bin/python benchmarks/scaling.py --rows 300000 --columns 150 --fixture sparse --operations fingerprint prepare missingness dependencies paths patterns explore --repeats 3 --timeout 180 --output /tmp/sparse-300k.json
# Vary cardinality and data types:
.venv/bin/python benchmarks/scaling.py --rows 300000 --columns 150 --fixture mixed --operations explore --repeats 3 --timeout 180 --output /tmp/mixed-300k.json
.venv/bin/python benchmarks/scaling.py --rows 300000 --columns 150 --fixture structured --operations explore --repeats 3 --timeout 180 --output /tmp/structured-300k.json
# Collect inclusive phase times (the historical run also passed --compact, removed after 0.1.2):
.venv/bin/python benchmarks/scaling.py --rows 300000 --columns 150 --operations explore --progress --repeats 3 --timeout 180 --output /tmp/progress.json
```

The raw JSON measurements (fixture definitions, environment versions, every
repeat, memory and export sizes) were retired after 0.2.1; they remain in git
history at tag `v0.2.1` under `docs/performance-results/`. Use the baseline
revision's harness to reproduce the historical runs, since it predates runtime
and compact options. See [performance controls](performance.md) for the public
APIs and remaining memory/latency limitations.
