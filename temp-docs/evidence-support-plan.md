# Implementation plan: dependency coverage and repeated support

Status: planned; implementation has not started.

Reviewed source baseline: `aac50800966c8bb7e5aa16e22c5306446ee5d841`.
This plan records the conclusion of the literature review and repository review.
It supplements the delivered milestone in `implementation-spec.md`.

## Outcome and boundary

Make dependency details, candidate summaries, and presentation ranking distinguish:

1. Exactness on the population actually evaluated.
2. Consistency supported by repeated determinant groups on that population.
3. A meaningful entity interpretation, which these measurements do not establish.

The first phase ends when these distinctions are visible and consistent through
saved results, overview presentations, and graph handoffs. Preserve observed
exactness, source selection, search budgets, and population-compatible graph
construction. No new dependency engine, evidence algebra, entity-weighted FD
analysis, or statistical reliability claim is required.

## Decisions to implement

### Populations and dependency measurements

For each determinant `X`, target `Y`, and global or conditional context:

- `P` is the scoped context population before dependency missing exclusions.
- `Q` is the determinant-eligible population: complete cases of `X` when
  `dropna=True`, otherwise all of `P`.
- `O` is the subset of `Q` with observed `Y`, using the existing native/sentinel
  presence policy.
- `E` is the evaluated population: `O` when `dropna=True`, otherwise `Q`.
- Group `E` by the determinant using the existing encoding. `R` contains rows
  in groups of size at least two **within E**.

Add these fields to every dependency record and its finding measurements:

| Field | Type | Definition |
| --- | --- | --- |
| `determinant_evaluated_rows` | `int` | `len(Q)` in this context |
| `target_observed_rows` | `int` | `len(O)` |
| `target_coverage` | `float \| None` | `len(O) / len(Q)`; undefined for empty Q |
| `target_missing_excluded_rows` | `int` | `len(Q) - len(E)` |
| `repeated_rows` | `int` | `len(R)` after all test exclusions |
| `repeat_coverage` | `float \| None` | `len(R) / len(E)`; undefined for empty E |
| `repeat_modal_accuracy` | `float \| None` | `1 - repair_rows / len(R)`; undefined for empty R |

All repair rows belong to repeated groups, so the last formula uses the existing
repair total. Compute the new quantities from existing masks and modal-group
sizes; do not introduce another full discovery pass or retain full masks per test.
Counts are nonnegative Python integers; undefined values serialize as JSON null.

Keep existing fields and meanings: `exact`, `modal_accuracy`, `repair_rows`,
`evaluated_rows`, `missing_excluded_rows`, `repeated_groups`, group violation rate,
and bounded exception groups. In particular, existing `missing_excluded_rows`
remains `len(P) - len(E)`, not just target exclusions.

With `dropna=False`, missing values participate as a category, so E equals Q and
`target_missing_excluded_rows` is zero. `target_coverage` still reports observed
target availability and can be below one. Label it as observed target coverage,
not as the fraction of rows evaluated. Determinant eligibility in this mode does
not imply that the determinant values are observed.

Do not add macro accuracy in this phase. It is a separate weighting choice and
does not remove singleton inflation. Keep counting unit, aggregation, weighting,
and resampling unit distinct in the documentation. Dependencies remain row-counted
and row-weighted, with no new resampling operation.

### Candidate summaries and ranking

Keep candidate `determines` as the ordered list of globally observed exact
targets, including those supported only by singleton groups. Add:

- `determines_with_repeated_support: list[str]`: global exact targets whose own
  evaluated population contains at least one repeated determinant group.
- `global_targets_tested: int`: completed global candidate/target tests, including
  tests with no evaluated rows. Excludes conditional tests and graph work.
- `global_targets_possible: int`: selected non-key targets for this candidate.

Do not aggregate conditional exact targets into either global list. Their coverage
and support remain attached to their typed context. Empty lists with incomplete
test coverage mean no such target was established in the work performed, not that
none exists. Candidate-level `repeated_rows` retains its determinant-population
meaning; it is never substituted for a dependency's `repeated_rows`.

Full candidate summaries show both exact-target counts, global test coverage, and
the existing determinant-group counts. Detailed per-target rows expose the new
coverage and repeat measurements, including tests below the finding threshold.
Use the existing `dependencies` table as the source of those rows rather than
duplicating full records into candidates.

For full overview presentation, preserve the current role ordering: repeated
groupings, unique identifiers, constants, then candidates without support. Within
each role replace the current raw exact-target-count tie-breaker with the count of
`determines_with_repeated_support`, followed by the existing repeated-row count,
key size, and lexical tie-breakers. Do not use raw exact-target count as an extra
tie-breaker or multiply scores by global target coverage. This removes one specific
unsupported ranking advantage without claiming statistically reliable ranking.

This ordering is explicitly based on evaluated evidence; show incomplete test
coverage. It remains sensitive to budgets and is not a completeness guarantee.
Preserve candidate enumeration order in analytical payloads and finding IDs.
One repeated group qualifies for the descriptive count, not for a reliability
certificate. Unique identifiers remain valid row-grain candidates.

### Serialization and presentation compatibility

Keep discovery schema `1.0`: this is an additive payload extension with unchanged
existing analytical field meanings and no new function parameters or exports.
Document the intentional overview ranking/presentation change.

New renderers must accept old saved 1.0 results without a source dataframe or
recomputation. Where old dependency records contain sufficient measurements,
derive repeat rows as `evaluated_rows - (evaluated_groups - repeated_groups)` and
repeat accuracy from those rows and `repair_rows`. Derive the supported-exact list
from old global records. Use a private presentation adapter; do not mutate saved
payloads or fabricate target coverage when determinant eligibility is unavailable.
Display unavailable legacy measurements as unavailable, not zero. If records are
insufficient to recover the ranking inputs, retain the legacy ranking for that
whole candidate collection rather than mixing incompatible scores.

Keep source validation, selectors, example totals, repair-row exception semantics,
and compact round-trips intact. No change to `grain()`'s foundation schema is
needed; graph evidence retains its own population and meanings.

Use full-presentation explanations for numerical qualifications. Do not put
counts, coverage, or support classifications into shared `statement`/`structure`
fields that also appear in topology exports. Topology must not acquire new
measurement-derived fields; order its candidate entries canonically rather than
using the new support ranking. Preserve existing structural roles and relationships.

## Sequence and commit boundaries

### 1. Capture the pre-change baseline and semantic cases

Suggested commit: `test(discovery): capture dependency support baseline`.

Add `benchmarks/evidence_support.py` with deterministic named cases and a small
versioned JSON report. Record the source commit, fixture/harness version, seed,
Python/pandas/NumPy versions, analysis parameters, evaluated populations, and
ordering. Missing new fields in the baseline are unavailable, not measured zero.
Record pre-change output before production edits; store compact reviewed reports
in `docs/evaluation/`. Keep timings separate from correctness evidence.

Add readable fixtures/characterization tests in a focused new
`tests/discovery/test_dependency_support.py`, including an old schema-1.0 export
for compatibility checks. This commit should pass on the old implementation;
add new expected-behavior assertions with the implementation commits.

Use the cases in the acceptance matrix below. Include a fixed column permutation
under the same candidate/test budget to document current search-order sensitivity.
Do not require permutation invariance or claim ground-truth entity recovery.

### 2. Compute target-specific support and additive candidate fields

Suggested commit: `feat(discovery): expose dependency coverage and repeated support`.

Primary files: `src/fieldwork/discovery.py`, producer/result docstrings in
`src/fieldwork/evidence.py`, `docs/algorithms.md`, `docs/contracts.md`, and focused
tests from step 1.

Compute Q once per determinant/context, reuse existing target availability and
modal-group results, then add the specified fields to every computed record.
Update candidate lists and test counts only from completed global tests. Keep
`min_accuracy` as a finding-emission threshold and retain below-threshold records.
Exercise zero budgets, conditional populations, sentinels, and missing-as-category
behavior. Update NumPy-style Returns/Notes/examples in the same commit.

Keep existing exactness, selections, graph views, caches, and execution budgets
unchanged. Result payloads remain documented versioned mappings; maintain precise
public signatures/types and existing editor navigation without introducing a new
public result-type hierarchy solely for this extension.

### 3. Make summaries and ranking reflect the new evidence

Suggested commit: `fix(presentation): qualify dependency support in grain summaries`.

Primary file: `src/fieldwork/presentation.py`. Check propagation through
`workflow.py`, `families.py`, and `evidence.py`; change them only if needed to expose
the same evidence, not to reinterpret network edges.

Implement the private legacy adapter, full overview ordering, candidate summary
counts/test coverage, and readable dependency explanations in text/HTML/SVG.
Expose a full dependency/candidate projection for standalone dependency results,
so evidence below `min_accuracy` can be inspected without manufacturing findings.
Keep verbose per-target details out of the compact overview.

Example wording for the sparse-target case: "Exact on 2 evaluated rows; target
observed on 2/4 determinant-eligible rows; no repeated groups after exclusions;
repeat-only consistency not assessable." Missing-as-category analyses must also
say that missing values participated in consistency measurements.

Extend `tests/discovery/test_journeys.py` and saved rendering tests for overview,
standalone results, legacy results, numerical disclosure, and topology. Exercise
`to_frame`, source inspection/selection, JSON, compact exports, and feature-network
references. Regenerate any affected assets and review their rendered appearance.

### 4. Qualify the corrected baseline and document the invariant

Suggested commit: `test(evaluation): qualify dependency evidence summaries`.

Run the same harness from step 1 on the completed implementation. Save the
identified source revision, before/after reports, and interpretation in
`docs/evaluation/dependency-support.md`; link it from `docs/index.md`.

Document the existing exact-graph invariant: key relationships and assignments in
each view use its compatible population; pair-specific truth does not transfer to
a broader population; views are not composed. Exact restriction/transitivity is
not claimed as a new theorem. No general evidence algebra or reliable entity
identification guarantee is introduced. Empty populations remain unsupported.

Confirm unchanged graph payloads for this work and include end-to-end tests of
both counterexamples. A correct detail table is insufficient if an overview still
implies broader support. Explain legitimate differences between graph and test
populations rather than requiring their support counts to match.

Use `benchmarks/parity.py` on the same deterministic corpus and dependency versions.
For cross-revision comparison, use an explicit narrow allowlist of added fields
and intended presentation changes; require unchanged legacy analytical fields,
selectors, graph outputs, and search coverage. Do not replace the parity oracle
with a blanket acceptance of changed output. Measure the existing discovery
workload before/after for added runtime and result size; investigate unexpected
changes without brittle wall-clock assertions.

### 5. Prepare user documentation for release synchronization

Source documentation and executable examples accompany steps 2–4. Prepare the
user-facing narrative/example and a synchronization checklist in this repository's
temporary planning area. Do not publish or describe unreleased behavior as stable.

The separate `fieldwork-docs` repository currently records channel `stable`, release
`v0.1.0`, source `657555f44d29960a3fe5d11eb9276634cd73f5a8`. Its AGENTS.md requires
release-matched documentation and synchronization through a PR. That work follows
a release containing this implementation; verify the then-current stable release
and update all source metadata together. Release/deployment is not part of this
implementation phase.

The follow-on documentation PR should update `reference/discover-dependencies.md`,
`reference/measurement-details.md`, `reference/results.md`, `guides/dependencies.md`,
the relevant overview/grain explanation, executable example checks, and generated
assets. Run that repository's `npm run format` and `npm run validate`.

## Acceptance matrix

| Case | Required behavior |
| --- | --- |
| `X=[1,1,2,2]`, `Y=['a',None,'b',None]` | Q=4, E=2, target coverage=0.5, repeated rows/groups=0, repeat coverage=0, repeat accuracy=null, exact=true; target remains in `determines` but not its repeated-support counterpart |
| 98 singleton groups plus one two-row group with conflicting targets | E=100, repeated rows=2, repeat coverage=0.02, modal accuracy=0.99, repeat accuracy=0.5; visible with default threshold |
| Empty context or wholly missing target with `dropna=True` | No supported exact finding; empty-E accuracy/repeat coverage undefined; zero target coverage only when Q is nonempty; empty-Q target coverage undefined |
| `dropna=False`, native/sentinel missing targets | Missing category evaluated; E=Q, target exclusions=0, observed coverage still reflects absence; repeat metrics use category-inclusive groups |
| Scoped, conditional, and composite determinants | All fractions use the corresponding Q/E/R; missing context values retain current category semantics; duplicate indexes do not change source-position selection |
| Rare context with an internally complete repeated mapping | Local coverage remains complete; no global coverage penalty suppresses its interpretation |
| Equal-role candidates, one with singleton-only exact targets | Full ranking uses repeat-supported exact count, not raw exact count; test an actual ordering reversal with other tie-breakers controlled |
| Zero/partial dependency-test budgets and independent graph budgets | Candidate test counters expose omissions; no unsupported zero/negative inference; graph work does not inflate these counters |
| Incompatible populations, sparse targets, overlapping grains | Existing graph safeguards and placements remain unchanged; no inference across views or conflation of network reachability with entailment |
| Old saved JSON and compact/new round-trips | Loading/rendering works source-free; unavailable coverage is not fabricated; strict JSON contains no NaN; selectors recover the same rows |
| Full versus topology presentations | Full details qualify support; topology contains no new counts, fractions, evidence pointers, or support-based ordering |

## Validation and completion

During each code commit, run focused tests for the touched behavior. At integration:

```bash
uv run pytest
uv run pyright --warnings
uv run ruff check src tests scripts examples
uv run ruff format --check src tests scripts examples
uv run python examples/investigation.py
uv run python scripts/generate_assets.py --check
uv build
```

Also run the new evaluation harness, existing parity comparison with its explicit
change allowance, and the existing discovery performance benchmark. Refresh
notebook outputs only if affected, and visually inspect changed generated assets.
Run installed-wheel documentation/editor checks from `docs/development.md` and the
supported-Python CI matrix. No new production dependency is needed.

Completion requires passing acceptance cases, accurate public docstrings, saved
result compatibility, a versioned before/after baseline, source documentation,
and the prepared release-documentation handoff. It does not require publishing a
release, deploying the site, running an analyst study, or implementing the research
backlog. Track each coherent change in a separate commit. Mark this plan complete
and retire implementation detail into durable docs once delivered.

## Subsequent research, outside the first phase

| Experiment | Failure mode and matched comparison | Outcome to measure |
| --- | --- | --- |
| Reliable-AFD ranking | Cardinality/sparsity artifacts; corrected baseline versus added bias-adjusted score on identical candidates/populations | False leads under independence, supported-grain ranking, cost |
| Closed availability bundles | Full signatures fragment when unrelated optional fields vary; compare on identical presence matrices | Planted partial-bundle recovery, redundancy, rare-pattern retention |
| Entropy prefix losses | Poor early separation; keep beam, constraints, budgets and display penalties matched | Separation by depth and analyst task performance |
| Explanation diversity | Different feature sets answer effectively the same question | Distinct questions/structural components covered at matched path count |
| Adaptive candidate allocation | Input-order omissions under fixed budget | Candidate recall and ranking sensitivity under column permutations; distinguish proved pruning from heuristic allocation |

Systems-paper preparation may proceed alongside the first phase: describe current
contracts and mechanisms, report this baseline honestly, and design a bounded
analyst evaluation. Separate observed exactness, repeated support, and semantic
entity validity. Do not treat subsampled exactness or unique IDs in independent
deliveries as generalization evidence. Methods or novelty claims require further
formalization, task-matched comparisons, and successful experiments.
