# Dependency support qualification (v0.1.1)

The implementation distinguishes observed exactness, consistency supported by
repeated determinant groups, and semantic entity validity. It measures the first
two; it does not establish the third. Discovery schema remains 1.0. This is an
additive analytical extension and an intentional full-presentation ranking change.

## Reproducible evidence

The original baseline was captured before production edits at
`16cca830837f50b5c9140585ae2b3f6cdf7df878` (the worktree's actual starting revision,
newer than the plan's reviewed source). Commit `3bb1110` preserves the initial
12-case harness, pre-change report, and genuine old schema-1.0 fixture. Harness
version 2 expands to 18 cases, including an ordering reversal and incompatible
transitivity. The expanded baseline was rerun against an archive of that same
starting revision. The implementation source is
`f857e369f4a67a4d34a318825218f021f8e0bca3`; subsequent qualification work adds
validation, documentation, and regenerated notebook output without changing the
measurements or ranking.

- [Before report](dependency-support-before.json) and [after report](dependency-support-after.json)
  record fixture/harness version, seed, Python/pandas/NumPy versions, parameters,
  scope, candidate and presentation order, dependencies and graph digests.
- [Parity report](dependency-support-parity.json) records the exact extension
  allowlists, 100-case corpus hashes, and graph/order comparisons.
- [Performance samples](dependency-support-performance.json) keep timings separate
  from correctness evidence.
- [Executable example](../../examples/dependency_support.py) and
  [rendered HTML](../assets/dependency-support.html) demonstrate the sparse case.
  The generated PNG was visually checked for readable, unclipped qualifications.

Use the same harness and dependency environment for both revisions:

```bash
# Archive the baseline without changing the working checkout.
mkdir -p /tmp/fieldwork-evidence-baseline
git archive 16cca830837f50b5c9140585ae2b3f6cdf7df878 | tar -x -C /tmp/fieldwork-evidence-baseline
PYTHONPATH=/tmp/fieldwork-evidence-baseline/src .venv/bin/python benchmarks/evidence_support.py --source-commit 16cca830837f50b5c9140585ae2b3f6cdf7df878 --output /tmp/support-before.json
.venv/bin/python benchmarks/evidence_support.py --output /tmp/support-after.json
PYTHONPATH=/tmp/fieldwork-evidence-baseline/src .venv/bin/python benchmarks/parity.py --output /tmp/parity-before.json
.venv/bin/python benchmarks/parity.py --output /tmp/parity-after.json --compare /tmp/parity-before.json --allow-evidence-support-extension
```

The parity adapter removes seven new fields only from discovery dependency records
and dependency finding measurements, and three new fields only from discovery
candidate summaries. It does **not** remove the pre-existing candidate
`repeated_rows`, foundation graph fields, selectors, coverage, or any ordering.
A sensitivity test deliberately changes legacy measurements and graph populations
and requires those changes to fail parity. Full-presentation changes are checked
separately; analytical payload order is preserved.

## Findings

| Case | Qualified outcome |
| --- | --- |
| Sparse target: X=[1,1,2,2], Y=['a',None,'b',None] | Exact on E=2; Q=4, observed coverage 0.5; R=0, repeat coverage 0, repeat accuracy null. The target remains globally exact but has no repeated support. |
| 98 singletons plus one conflicting pair | Row modal accuracy 0.99; R=2/100, repeat-only accuracy 0.5. The default finding threshold retains it and all full renderings qualify it. |
| Empty Q or E | No supported exact finding; undefined fractions stay null. Observed coverage is zero only for nonempty Q with no observed target. |
| Missing-as-category | E=Q; target exclusions are zero; native/sentinel absence still lowers observed coverage. Full explanations disclose category-inclusive consistency. |
| Scoped, composite and conditional cases | Q/E/R use the corresponding scope/context; duplicate indexes select the same source positions. A rare internally complete repeated mapping retains local coverage 1. |
| Equal-role ranking reversal | X has three singleton-only exact targets; Z has one repeated-supported exact target. Both have eight determinant repeated rows. Full presentation moves Z ahead of X; raw exact-target count supplies no extra tie-breaker. |
| Budget and permutation | At two tests, column order X,Y,C tests X→Y and X→C; order Y,C,X tests Y→C and Y→X. Global counters expose these omissions. Graph work does not count as target tests. |
| Saved results | Legacy repeat arithmetic is recovered without source data or mutation; target coverage stays unavailable. Insufficient ranking inputs trigger collection-wide legacy ordering. Ordinary/compact JSON and selections round-trip. |

All 18 graph digests, scoped populations, search coverage and analytical candidate
orders match. Full ordering changes in the report are intentional consequences of
using repeated-supported exact targets. Topology has no new measurements, evidence
pointers, or support-based ordering. Candidate roles remain descriptive; one repeated
group qualifies for the count, and unique identifiers remain valid row-grain leads.
No permutation invariance or ground-truth entity recovery is claimed.

## Existing graph invariant

Each exact graph view uses a compatible population for its key relationships and
attribute assignments. A dependency true on a determinant/target-specific subset
does not become an assignment on a broader view. Views are never composed.

The journey tests use X=[1,1,2,2], sparse Y=['a',None,'b',None], and Z=[0,1,2,3].
X→Y and Y→Z hold on rows 0 and 2, but X→Z fails globally. The four-row graph leaves
Y unplaced with `different_target_population`; the two-row graph may merge the
three observed partitions. This difference is correct: their populations differ.
Disjoint left/right optional grains likewise never become members of the same
view. The singleton-inflation case also reaches the overview and dependency detail
renderings, where 0.99 overall accuracy is accompanied by 0.5 repeat-only accuracy.
A feature-network link points to its actual finding and selectable population;
network reachability is not logical entailment.

These are existing safeguards, unchanged by this work. Exact restriction and
transitivity on a fixed compatible population are not new theorems. Empty
populations remain unsupported. No general evidence algebra, statistical reliability
claim, or entity identification guarantee is introduced.

## Cost and validation

On Python 3.11.14, pandas 3.0.5 and NumPy 2.4.6, the seeded 2,000-row/25-column
workload performed 288 dependency tests in 0.0698 seconds before and 0.0688 seconds
after. Serialized dependency results grew from 982,236 to 1,049,915 bytes (+6.9%),
consistent with retaining the additional fields. Missingness/path result sizes
and all workload coverage remained unchanged. These are single wall-clock samples,
not evidence of a speedup or a stable runtime bound. Q is computed once per
candidate/context; target masks and existing modal group sizes supply the new
measurements, without another discovery pass or retained per-test masks.

Validation passed all 346 tests on supported Python 3.11–3.14, pyright,
ruff lint/format, the investigation and dependency examples, executed notebook,
generated asset check, wheel/sdist build, and installed-wheel documentation/editor
checks (125 tests per interpreter) and investigation smoke tests. Supported Python versions were exercised
locally on macOS; the repository's CI continues to cover Linux. No production
dependency was added. [Development](../development.md) lists the commands.

## Boundaries and follow-on work

[Algorithms](../algorithms.md#dependency-target-coverage-and-repeated-support) and
[contracts](../contracts.md#additive-dependency-support-fields-discovery-schema-10)
are the durable behavioral specification. The
[v0.1.1 release record](../releases/v0.1.1.md) records publication and the matching
documentation synchronization.

Future experiments remain separate: bias-adjusted AFD ranking versus this corrected
baseline, closed availability bundles, entropy prefix losses, explanation diversity,
and adaptive candidate allocation under fixed budgets. Any entity validity or
systems-paper novelty claims need task-matched evaluation and further formalization;
subsampled exactness and unique IDs in independent deliveries are not generalization
evidence. This implementation includes no new dependency engine, macro averaging,
entity-weighted FD analysis, or resampling.
