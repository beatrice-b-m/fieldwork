# Review remediation plan

Status: steps 1–3 complete and released in v0.2.0 (2026-09-23); step 4
complete on branch `test-suite-rebalance` (unreleased). Step 5 is scoped for a
separate, self-contained session. Read the status log at the end first: it
records what each step changed and measured, and facts later sections depend on. Update the status log as each step lands, and retire this
document when all five steps are complete (move anything durable into `docs/`).

## Why this plan exists

Fieldwork is meant to help an analyst who receives a raw, complex export with
little context. It surfaces structural evidence and implied data topology, such as
availability patterns, grains and dependencies, and then helps the analyst follow
leads back to source rows. The analytical core does this well. A review on
2026-09-23 found that the layers around that core cost more than they return for a
single-analyst exploration tool:

- a guarantee that saved evidence matches the exact source frame (a SHA-256
  fingerprint of every cell)
- versioned schemas, legacy adapters and a compact export format
- two parallel result, encoding and rendering stacks
- a docstring regime and a test suite that mostly check wording and layout rather
  than analytical behavior

The review also found real bugs, and output noise that buries the leads the tool
exists to surface.

The whole repository was written between 2026-09-16 and 2026-09-18 and is an alpha
(`0.1.x`). Nobody depends on long-lived saved formats, so **changes in this plan may
break saved-result compatibility without migration shims**. Record every such break
in `docs/release-notes/unreleased.md`.

### Principles for every step

- Prefer deleting code to adding configuration. Every new option needs a real
  analyst use case.
- Tests assert analytical behavior (counts, relationships, populations, selection)
  or genuinely valuable contracts (round trips, source-mismatch rejection). Do not
  pin prose, CSS or docstring layout.
- Keep the core guarantees an analyst relies on:
  - analyses never mutate the source
  - search and display budgets never silently change the analyzed population
  - every finding can be traced back to its source rows
- Keep `docs/` synchronized, as required by `AGENTS.md`. Generated assets
  (`scripts/generate_assets.py --check`) and the executed notebook
  (`examples/investigation.ipynb`, checked by `tests/discovery/test_examples.py`)
  must be regenerated whenever output changes.

### Validation commands (all steps)

```bash
uv run pytest -q
uv run pyright --warnings
uv run ruff check src tests scripts examples benchmarks
uv run ruff format --check src tests scripts examples
uv run python examples/investigation.py
uv run python scripts/generate_assets.py --check   # regenerate without --check when output changes
```

Before the review, the baseline was 358 tests passing in about 4.4 s, pyright clean,
and 2 ruff import-order errors in `benchmarks/`. After steps 1–3: 356 tests pass in
about 3.6 s, pyright is clean, and `ruff check` is clean including `benchmarks/`.
CI (`.github/workflows/ci.yml`) still does not lint `benchmarks/`.

---

## Step 1: fix verified bugs (this session)

1. **A MultiIndex with float or datetime levels breaks every discovery analysis.**
   `_fingerprint` in `evidence.py` normalizes tuple index entries with
   `normalize_scalar(value, label=True)`, which accepts only str/int. The tuple-cell
   fallback has the same problem. This is the normal shape after
   `groupby([...]).agg()`. Fix: hash index values as values, not labels. Add a
   regression test running missingness, explore and inspect on such a frame.
2. **Default text output escapes non-ASCII, including Fieldwork's own labels.** Both
   renderers default to `unicode_mode="safe"`, so `print(fw.explore(df))` shows
   `Fieldwork \xb7 overview` and paths print as `a \u2192 b`. Fix: default to
   `"display"`. Measure width with `wcwidth` when installed, otherwise with a
   standard-library fallback (`unicodedata.east_asian_width` plus combining marks).
   Keep `"safe"` as an explicit option. Control and bidi characters stay escaped in
   both modes.
3. **Timedeltas sort as strings and display as raw nanoseconds.**
   `ScalarIdentity.sort_key` falls through to comparing the stored string for
   `timedelta`, giving the order `-5s, 10s, 100s, 9s`. `display_scalar` prints
   `timedelta(-5000000000 ns)`. Fix: sort numerically and display via
   `pd.Timedelta`.
4. **One unsupported cell type aborts the whole analysis.** Lists, dicts, `Decimal`
   and similar objects raise `TypeError` from `encode_series`. JSON-derived exports
   commonly contain such columns. Fix for discovery analyses:
   - When columns are selected automatically (`features=None`), skip unsupported
     columns and report them in the result (`skipped_features` with a reason), in
     the text overview, and in coverage.
   - When the caller names the column explicitly, keep raising `TypeError` with a
     clear message.
   - Fingerprinting must accept any cell. Step 3 does this naturally.
5. **A cosmetic progress miscount can abort an analysis.** `Phase.advance` in
   `_runtime.py` raises `ValueError("Progress exceeds phase total")`. Fix: clamp the
   displayed progress instead of raising.

Acceptance: each bug has a behavioral regression test, and existing suites pass
with only intended expectation updates.

## Step 2: make the overview useful (this session)

The overview is the analyst's entry point, and it currently buries the leads. On
the bundled 960×43 laboratory table (`examples/wide_table.py`, no missing values),
`explore` produced 573 findings:

- **400 were vacuous presence implications** ("A populated implies B populated"
  where B is populated everywhere). The overview's default `max_pairs=200` visits
  200 pairs, and each emits both directions.
- **43 were "X: populated values" at 100%.**
- **The rest were buried.** Findings are listed in section order, not by interest.
- **The text summary is capped at 40 lines.** It never showed value patterns or
  any top leads.
- **The four "major signatures" printed as identical truncated lines.** Each listed
  the present columns, which share a long common prefix.
- **Candidate grains listed `assay_id`, `analyte` and `detection_limit`
  separately.** They are the same partition, and another finding already reports
  the equivalence.
- **The feature network collapsed into one connected component** because vacuous
  implications link every column.

Changes:

1. **Suppress vacuous availability evidence in `missingness`.**
   - Skip implications whose target is populated in every unit.
   - Skip implications between identically available features; the family finding
     already covers them.
   - Skip the per-feature `availability` finding for fully populated features.
   - Every feature still appears in the `availability` table, so the section keeps
     its complete measurements.
2. **Rank overview findings by how useful they are as leads.**
   - Apply a small, documented heuristic score per pattern:
     - near-rules with exceptions
     - mutual exclusion
     - partial or empty availability
     - multiple string formats
     - equivalent partitions
     - exact dependencies with repeated support
   - Down-rank trivially true or purely descriptive findings: single-format
     strings, ranges, census paths.
   - Reorder the overview's finding list and assign IDs in rank order, so `f0` is
     the top lead.
   - Section results keep their own order and IDs.
   - Save the score and a short `lead` reason with each overview finding.
3. **Collapse equivalent candidate grains.**
   - Candidates with the same partition (same group count and mutually exact
     dependencies) appear once, with their aliases listed.
   - This is a presentation grouping. The saved candidate records are unchanged.
4. **Signature lines show what differs.**
   - Show the columns absent from the signature, relative to the columns populated
     in every signature.
   - Show the count, and "all populated" for the complete signature.
5. **Add a "Leads" block to the text overview** with the top-ranked finding
   statements and their IDs, so `print(fw.explore(df))` answers "where should I
   look first?".

Acceptance:
- The lab table overview has no vacuous implications, its top findings are
  non-trivial, and the feature network has more than one component.
- Selection and inspection of every remaining finding still work.
- The notebook, assets and docs are regenerated and updated.

## Step 3: cheap fingerprint, bounded exports, remove the compact format (this session)

1. **Replace the fingerprint.**
   - `evidence._fingerprint` JSON-serializes every distinct value, and every
     analysis call and source-bound method recomputes it. On a 500k-row frame it
     took 4.9 s of `missingness`'s 5.2 s. The foundation review measured
     `census(df, dims, scope=s)` at 16.7 s against 0.03 s without a scope.
   - Replace it with SHA-256 over the column labels plus
     `pd.util.hash_pandas_object` of the index and each column.
   - Consequences to document:
     - Dtype becomes part of the identity (an object column of ints differs from
       int64).
     - Fingerprints saved by 0.1.x no longer match, so saved scopes and results
       from earlier versions cannot be inspected against a source.
     - Any cell type can be fingerprinted.
2. **Bound grain-view populations.**
   - `discover_dependencies` stores every row position of each grain view's
     population (`grain_views[].population.positions`). That contradicts the
     "bounded positions" contract, and it dominates export size on large frames.
   - Replace it with a bounded example `selection(...)` record plus the anchoring
     candidate ID. The population is fully determined by the anchor candidate's
     complete cases within the saved scope.
3. **Delete the compact export format.**
   - `_serialization.py` (the `fieldwork.compact` `$ref`/`$dict` envelope) mostly
     deduplicates copies the payload creates itself. It saved about 15–30% in
     measurements, and the ordinary export is still large for other reasons.
   - Remove `to_dict(compact=...)`, compact handling in both `from_dict` methods,
     `_serialization.py` and its tests and docs.
   - Remove the largest self-made duplicate at the same time: `exact_grain` (an
     alias of `grain_views[0]["grain"]`, about 1.1 MB of the 4.8 MB lab-table
     overview). Use `grain_views[0]` instead.

On the lab table, the overview export was 4.8 MB for a 241 KB CSV. The remaining
size comes mostly from:
- the foundation grain result (tagged scalar identities)
- every completed dependency test record, with exception groups
- overview findings duplicating section findings

Step 5 addresses these.

Acceptance:
- Fingerprint time on the 500k-row benchmark frame is a small fraction of the
  analysis time.
- Grain-view positions are bounded.
- No compact envelope code remains.
- The export size of the lab-table overview is recorded in the status log.

---

## Step 4: rebalance the test suite (complete, see status log)

**Goal:** tests that protect analytical correctness, so step 5 can refactor safely.
Remove tests that freeze wording, docstring layout or editor behavior. **Do step 4
before step 5.**

### Current inventory

Counts are test functions, classified as a/b/c/d:
- **a:** analytical behavior
- **b:** valuable contract
- **c:** wording, layout, docstring or editor pinning
- **d:** redundant

The 358 collected items include parametrizations. **About 125 items (35%) only
check docstrings and editor behavior.**

| File | a/b/c/d | Notes |
| --- | --- | --- |
| foundation/test_census.py | 5/0/0/0 | conservation and permutation canonicality: keep |
| foundation/test_differential.py | 1/0/0/0 | Hypothesis oracle for levels and census prefixes: best test, but narrow |
| foundation/test_joint_counts.py | 2/0/0/0 | keep |
| foundation/test_contracts.py | 5/2/1/1 | imports private kernels |
| foundation/test_grain_relations.py | 5/0/1/0 | `:50` only asserts a scope ID `"s2"` |
| foundation/test_scopes.py | 4/0/0/0 | population accounting: keep |
| foundation/test_grain_graph.py | 7/0/0/0 | includes a row-partition oracle at `:121`: keep |
| foundation/test_resolved.py | 1/5/5/0 | label and repr pinning |
| foundation/test_graphics.py | 2/3/6/2 | mostly string/markup |
| foundation/test_render.py | 1/3/13/0 | almost entirely exact wording |
| discovery/test_context_adapter.py | 2/1/0/0 | pins object identity with `is` |
| discovery/test_dependency_support.py | 6/2/3/0 | also tests `benchmarks/parity.py` |
| discovery/test_examples.py | 0/1/1/0 | runs the examples; pins notebook numbers |
| discovery/test_journeys.py | 9/1/4/1 | real selection/entity behavior mixed with UI copy |
| discovery/test_performance_contracts.py | 1/3/1/0 | fingerprint differential is valuable; pins internal call sequences |
| discovery/test_presentation_ux.py | 1/1/10/0 | HTML copy and attributes |
| discovery/test_runtime.py | 2/3/3/1 | cancellation and cleanup good; pins ETA maths and cache thresholds |
| discovery/test_scaling_controls.py | 4/3/1/0 | `modal_groups` property test and selection equivalence are good |
| discovery/test_workflow.py | 12/3/4/1 | most useful discovery file |
| test_inline_docs.py | 0/0/5/0 (103 items) | docstring sections, parameter-name parity, AST checks of `typing.py`, doctests |
| test_editor_api.py | 0/0/3/0 (22 items) | jedi completions, hover and goto; depends on the jedi version |
| typing/public_api.py | pyright `assert_type` checks | reasonable, not behavioral |

Line numbers are from the 2026-09-23 baseline; re-locate before editing. Steps
1–3 changed the suite as follows:

- **Removed:** the byte-stream fingerprint parity tests
  (`test_performance_contracts.py`, formerly `original_fingerprint` and three
  tests) and the four compact-envelope tests in `test_scaling_controls.py`.
- **Added behavioral tests:**
  - `test_workflow.py`: groupby-style MultiIndex, unsupported cells
    skip/reject, vacuous availability omission, similarity with always-present
    features, lead ranking and trivial-edge exclusion, overview summary
    structure
  - `test_render.py`: Unicode default and width with and without `wcwidth`
  - `test_contracts.py`: timedelta ordering
  - `test_runtime.py`: progress overrun
  - `test_performance_contracts.py`: fingerprint change detection
    (Hypothesis) and order/labels/index/dtype coverage
  - `test_scaling_controls.py`: bounded grain-view examples and a saved-export
    round trip
- **Adjusted fixtures:** in `test_journeys.py`, `test_presentation_ux.py` and
  `test_dependency_support.py`, where vacuous findings or unique determinants
  had been doing the work.
- **Legacy fixture:** `test_legacy_export_loads_without_source` now asserts that
  a 0.1.x fingerprint is *rejected* on `select`. The fixture
  (`tests/discovery/fixtures/dependency-schema-1.0.json`) still contains
  `exact_grain` and the old `population.positions`; loading ignores them. Step
  5.4 deletes the fixture with the legacy adapter.
- **Collected items:** 356, of which 125 are still in `test_inline_docs.py` and
  `test_editor_api.py`.

### Most brittle examples to replace or delete

- `foundation/test_render.py:22-36`: whole-output line-by-line comparison starting
  `"Fieldwork feature explorer v0.3"`.
- `test_render.py:249-264`: a substring blacklist (`" row"`, `" rows"`, `" cells"`,
  `"Cramer's V"`) that fires on any column named "rows".
- `test_render.py:110-125`: 15 wording assertions.
- `test_inline_docs.py:53-73` and `:85-100`: NumPy section and parameter parity
  across 103 objects, plus an AST requirement for a string literal after every
  TypedDict field.
- `test_editor_api.py:36-62`: jedi completion and hover content.
- `discovery/test_presentation_ux.py:28-69, 98-104, 145-159`: HTML copy,
  `svg.attrib["width"] == 900`, `height > 2000` and element IDs.
- `foundation/test_graphics.py:99-107`: copy such as `"5 groups · 6 rows"` and
  `'id="focus"'`.
- `test_resolved.py:165-178`: repr must be exactly 40 lines.
- `test_runtime.py:110-134, 177-188`: the ETA formula and cache thresholds
  (`60_000`, `100_001`).
- `test_performance_contracts.py:82-109`: the exact internal `encode_series` call
  list.
- `test_dependency_support.py:276-299`: tests a benchmark script's normalizer.

### Redundancy to consolidate

- **HTML ID uniqueness:** 3 tests, 2 with copy-pasted parser classes
  (`test_graphics.py:181,211`, `test_presentation_ux.py:90`).
- **"Topology leaks no quantities":** about 10 tests (graphics:76,
  presentation_ux:118, workflow:164/317/333, journeys:237/316,
  dependency_support:258, render:223/279).
- **Control-character escaping:** about 7 tests.
- **Renderer non-mutation:** 4 or more tests.
- **The singleton-inflation frame:** 2 tests.

Keep one strong parametrized test for each.

### Behavioral gaps to fill (highest value first)

1. **Oracle for dependency discovery.** Recompute modal accuracy, exactness,
   violating groups and repair rows independently (pandas groupby) on random
   Hypothesis frames, including composite keys, `dropna` both ways, and contexts.
2. **Known-answer tests for `pairs`.** Relation classes (1:1, 1:n, n:1, n:m) and
   Cramér's V against hand-computed values, including degenerate cases.
3. **NaN, None, `pd.NA` and `NaT` in the count oracle.**
   `test_differential` currently uses `allow_nan=False`. Also widen it beyond
   2 columns and 20 rows.
4. **A native-dtype matrix** across missingness, dependencies, value_patterns and
   census. Cover categorical (including unobserved categories), nullable
   `Int64`/`boolean`/`string`, pyarrow-backed, tz-aware datetime and float32.
   Assert counts agree with the object-dtype equivalent.
5. **`value_patterns` behavior.** Assert `string_patterns` formats and lengths,
   `numeric_range`, `indexed_family` and `context_constancy` values. Only
   `numeric_offset` is asserted today.
6. **Row-permutation invariance** for missingness, dependencies and
   `suggest_paths`: finding sets equal modulo positions. Only census has this.
7. **Census options** (`top_n`, pre/post, `per_parent`, `max_levels`,
   `min_count`) as property tests. Today they are checked on 2–5-row frames only.
8. **`compare`** with added and removed features and mismatched feature sets.
9. **One realistic-scale smoke test** (a wide frame of about 50k rows with
   high-cardinality keys), marked so it can be skipped locally if slow.

### Documentation and editor checks

`AGENTS.md` currently requires NumPy docstrings, editor-signature checks and
completion checks for every public API change. `test_inline_docs.py` and
`test_editor_api.py` enforce that policy. Replace them with:

- **One smoke test:** every exported callable has a docstring and annotated
  parameters.
- **Doctest execution:** `pytest --doctest-modules` on `src/fieldwork` replaces
  the custom doctest runner.
- **`tests/typing/public_api.py` under pyright**, as today.

Update the CI step in `.github/workflows/ci.yml` that runs those two files against
the installed wheel. The `AGENTS.md` wording itself is part of step 5, so it
matches the simplified API. Coordinate if step 4 lands first: leave a note in the
status log.

### Constraints

- Target a suite that runs in under about 10 s locally.
- Rendering tests should check structure: sections present, findings linked, no
  row positions in topology output, escaping. Do not check sentence text.
- Use private kernels in tests only where they have a property oracle
  (`modal_groups`). Otherwise test through public functions.
- The notebook test should check that the notebook executes and that its stated
  conclusions hold. It should not pin incidental numbers.

Acceptance:
- Every item in the gaps list has a test.
- The brittle examples are removed or rewritten structurally.
- The collected count may drop substantially; that is intended.
- CI is green.

---

## Step 5: simplify the architecture (separate session)

**Goal:** one result model, one value encoding, one projection and rendering path,
and smaller functions. Step 4's oracles are the safety net. Expect to break saved
formats; record breaks in release notes. Sequence the work so each item is its own
commit with a green suite.

### 5.1 Unify the two stacks

There are two parallel layers:

| | Foundation | Discovery |
| --- | --- | --- |
| Code | `src/fieldwork/_explore/` (extracted from bea-tools, see `NOTICE`) | `evidence.py`, `availability.py`, `discovery.py`, `navigation.py`, `patterns.py`, `families.py`, `presentation.py`, `workflow.py` |
| Result | `ExplorerResult` | `InvestigationResult` (subclass of `ExplorerResult`) |
| Schema | 0.3 | 1.0 |

`InvestigationResult` keeps the parent's default `schema_version="0.3"`, which is
documented as a caveat.

Symptoms:
- **Two renderers, chosen by result kind.** `presentation.py` (1177 lines) checks
  `kind not in KINDS` and otherwise sends results to the foundation renderers
  (`render.py`, `visual_data.py`, `graphics.py`). Output headers differ
  (`Fieldwork feature explorer v0.3` against `Fieldwork · paths`). The stacks share
  only `_SVG`, `_esc`, `_wrap` and `_clip`.
- **Different column-label rules.** Discovery requires string column labels
  (`evidence.columns`), while the foundation accepts int and tuple labels.
- **Four scope shapes:**
  - foundation `_scope` (`census.py:49`, where `retained_rows` always equals
    `evaluated_rows`)
  - discovery `base["scope"]`
  - grain-view `population`
  - `scope_metadata` lineage
- **Discovery payloads embed whole foundation results.**
  `grain_views[].grain` comes from `discovery.py` and `paths[].preview` from
  `navigation.py`. The overview copies `paths[0].preview` into
  `sections["census"]`. (Step 3 removed the `exact_grain` alias and the compact
  envelope. The overview still copies every section finding into its own
  ranked `findings` list, and each dependency finding's `measurements` repeats
  its full dependency record, exception groups included.)
- **Fragile glue.**
  - `contextual_result` (`evidence.py`) deep-copies a payload, then edits any
    dict that has both `scope_id` and `input_rows`.
  - `foundation_context` converts every column to object dtype to apply
    sentinels.
- **`explore` has two modes.** `explore(df)` returns an `InvestigationResult`,
  while `explore(df, dims)` returns an `ExplorerResult`. Overloads,
  `discovery=` versus keyword versus `section_options` merging, and
  duplicate/protected-key rules make up `workflow.py:274-329`.

Target:
- **One result class and one schema version.** Kind-specific sections reference
  sub-results rather than embedding copies.
- **One scope/population record shape.**
- **One projection layer and one renderer per medium** (text, SVG, HTML).
- **Consistent column-label support.** Decide whether discovery accepts non-string
  labels. Given the target users (raw exports), string labels plus automatic
  `str()` for display are probably enough.
- **Split `explore`.** Consider `explore(df)` for the overview and a separately
  named function for the explicit composition, instead of one overloaded entry
  point.

### 5.2 Replace the tagged scalar identity with native codes

`_explore/encoding.py` defines `ScalarIdentity` with 10 type families:
- ints are stored as strings
- floats are stored as `float.hex`
- values carry resolution metadata
- every output value becomes `{"type": …, "value": …}`

Consequences:
- **Three decoders:** `render._identity`, `evidence._restore_scalar` and
  `visual_data.label` (a near-copy of `render.py:90`).
- **Dict keys built from `str(dict)`:** `grain.py:194,205`,
  `visual_data.py:67,231` and `roles.py:178`.
- **Slow `levels()` on continuous columns.** It normalizes and sorts every unique
  value in Python even when it only reports 100: about 3 s for a 500k-unique float
  column.
- **Workaround code.** `MissingCode` and `normalized_encoding` in `evidence.py` (a
  per-column re-encoding with caches) exist to work around this encoding.

Measured after step 3 (500k rows × 14 columns: 10 low-cardinality ints, 3 unique
floats, 1 string), `explore` takes about 12.8 s. Phase timings:
- `encoding`: about 6 s, almost entirely `normalize_scalar`, `sort_key` and
  `ScalarIdentity` hashing, with about 1.5M calls for the three unique float
  columns
- `paths`: 7.7 s inclusive
- `dependencies`: 4.8 s

This encoding is the single largest remaining cost. Fingerprinting no longer
uses it (step 3).

Target: `pd.factorize` codes for all analytics. Convert to JSON-safe scalars only
for values that are emitted, using plain JSON numbers with NaN/inf as strings.
Keep the valuable parts of the contract:
- `1` and `1.0` stay distinct only if that matters in practice (decide
  explicitly)
- numeric sentinel matching stays numeric
- native missing values collapse to one token

### 5.3 Break up the very large functions

- `missingness`: about 480 lines with nested closures.
- `discover_dependencies`: about 430 lines.
- `suggest_paths`: about 400 lines. Its scoring uses hand-tuned weights (2, 3, 8,
  12) in `navigation.py` that should be named and documented as heuristics.
- `value_patterns`: about 280 lines.
- `render_plaintext`, `render_svg` and `render_html` in `presentation.py`: about
  650 lines of `if kind ==` branching.
- `visualization_data`: about 220 lines.
- `_explore/render.py:_section_lines`: about 300 lines.
- `_explore/graphics.py` (863 lines) is a grab-bag:
  - SVG primitives, five figure builders, an HTML matrix, a 90-line inline JS
    string and page assembly
  - post-processing SVG with `str.replace` to inject `data-features`
  - unreachable `_matrix_svg`/`view == "map"` branches after a `continue`

Target: one function per pattern family or figure, returning typed records (for
example dataclasses) rather than growing nested dicts.

### 5.4 Remove compatibility shims and dead parameters

Already gone (step 3): `_serialization.py`/`to_dict(compact=...)`, `exact_grain`.

- The legacy dependency-ranking adapter: `presentation.py` `candidate_priority(...,
  legacy=)`, `_dependency_measurements` backfills, and
  `tests/discovery/fixtures/dependency-schema-1.0.json` with its test.
- `grain(schema=)`, documented as a "Reserved compatibility argument… no effect".
- `engine_metadata` everywhere, `stability="unstable"`, and fields such as
  `claim: "support_description_only"` and
  `higher_order_constraints_ruled_out: False` (`relations.py`).
- `python_prefix_counts` (`_kernels.py`), which exists only for tests.
- Every public function declares `progress`, `cancel` and `timeout` and repeats the
  same 12-line docstring block. The `operation` decorator already strips them.
  Declare them once, and keep them visible to type checkers.
- Duplicated logic:
  - `_pre_cohort` (`orchestration.py`) against census pre-selection
    (`census.py`), where the first is slower (one pass per level code)
  - key comparison in `grain.py` against `grain_graph.py`
  - warning construction twice in `census.py`
- Explicit `explore` re-encodes the same columns up to five times.

### 5.5 Right-size runtime, progress and caching

- **Progress.** `_runtime.py` and `progress.py` keep a rate deque with a pstdev
  stability test for ETA, a 0.2 s throttle, phase/session stacks, and several
  bounded caches (`remember_encoding` at 100k tokens, the `prefix_cache` in
  `navigation.py` at 32 MB, `FDCache` at 512 entries / 16 MB). Keep cancellation,
  timeout and a simple phase/percentage progress. Drop ETA statistics unless
  there's a demonstrated need.
- **Caches.** Re-evaluate each once step 5.2 removes per-value normalization. Most
  exist to amortize that cost.

### 5.6 Reduce parameter and documentation weight

- **Parameter counts:** `suggest_paths` 20, `census` 19, `missingness` 18,
  `discover_dependencies` 18. Group rarely used budgets into one options object,
  or drop them.
- **Docstrings:** `explore`'s is 170 lines, and the public functions total about
  1,300 docstring lines. Relax `AGENTS.md` to require concise NumPy docstrings
  covering purpose, parameters, returns and one example, with population and
  denominator semantics documented once in `docs/`.
- **`docs/contracts.md`:** dense policy prose. Rewrite it around what an analyst
  can rely on.
- **Retire fulfilled records.** `temp-docs/evidence-support-plan.md` and
  `temp-docs/performance-audit.md` are fulfilled. `docs/releases/*.md` (CI and
  commit bookkeeping) duplicates `docs/release-notes/`. `docs/assets/README.pypi.md`
  is an identical copy of `README.pypi.md`, but `scripts/generate_assets.py`
  produces it, so check before deleting. `docs/performance-results/` and
  `docs/evaluation/*.json` are raw measurement dumps.

### Code added in steps 1–3 that step 5 should keep working

- `src/fieldwork/leads.py`: the lead heuristic and `rank()`, called from
  `workflow.explore`. `TRIVIAL` reasons also filter edges in
  `families.feature_network`. The scores are hand-tuned. Keep them in one
  place, and keep the behavioral test
  (`test_overview_ranks_leads_and_keeps_trivial_rules_out_of_network`).
- `evidence.prepare(..., optional=...)`, `analyzable()` and the
  `skipped_features` payload field: skip-and-report for automatically selected
  columns with unsupported cells.
- `evidence._value_hashes`: the vectorized fingerprint. Object columns hash
  `f"{type(v).__qualname__}:{v!r}"`.
- `presentation._collapse_equivalent`, `_grain_title` and `_signature_label`:
  overview grain merging and signature labels, used by text, SVG and HTML.
- `_explore/render.py:_width_function`: Unicode display width, using `wcwidth`
  when installed and `unicodedata` otherwise.

### Risks and guidance for step 5

- The user-facing docs live in a separate repository (`fieldwork-docs`; see
  `docs/releases.md`). Record public API changes so that repository can be
  synchronized.
- `scripts/generate_assets.py` renders the README hero and the documentation
  images from the public renderers. Regenerate them and review them visually after
  renderer changes.
- `scripts/check_output_ux.cjs` is an optional Playwright harness for the HTML
  output. Update or retire it with the HTML renderer.

Acceptance:
- One result class and schema.
- One renderer path per medium.
- No `ScalarIdentity`.
- Public functions under about 100 lines each.
- The suite from step 4 is green.
- Benchmarks (`benchmarks/scaling.py`) show no regressions.
- Docs and release notes are updated.

---

## Status log

- 2026-09-23: Plan written. Retired `temp-docs/implementation-spec.md`; its
  delivered content already lives in `docs/`.
- 2026-09-23, **step 1 complete.** Commits `9600bdc` (MultiIndex), `938db1d`
  (Unicode default), `4f23137` (timedelta), `ce48de9` (unsupported cells, new
  `skipped_features`) and `cf8e08b` (progress clamp).
  - Explicitly requested columns with unsupported cells now raise
    `TypeError("Column 'x': …")`.
  - The skip reason stores only the value type, so topology exports carry no
    cell values.
- 2026-09-23, **step 2 complete.** Commits `5f500b3` and `44a558b` (vacuous
  availability and similarity findings omitted), `eb17862` (lead ranking,
  `lead.score`/`lead.reason`, trivial edges out of the network) and `bf0c06d`
  (Leads block, merged equivalent grains, "Missing:" signatures, lead reason
  badges in HTML, notebook narrative).
  - Lab table (`examples/wide_table.py`): 573 findings reduced to 129.
  - The feature network separates into 2 components (14 and 7 features)
    instead of 1.
  - A messy variant (lowercased IDs, a stray reagent-lot format, partially
    populated columns) surfaces each planted issue in the top 10 leads.
  - One late fix: merging grains requires identical group counts and evaluated
    rows, because with `dropna` two columns can "determine each other" on
    different populations.
- 2026-09-23, **step 3 complete.** Commits `3238871` (fingerprint), `4b76ed2`
  (bounded grain-view positions, `anchor_candidate_id`) and `5dc3110` (compact
  envelope and `exact_grain` removed); `310ba67` fixes the benchmark import
  order.
  - 500k × 14 frame: fingerprint 4.8 s → 0.04 s; `missingness` 5.2 s →
    0.24 s; `explore` 18.3 s → 12.8 s; scoped `census` 0.08 s.
  - Lab-table overview export: 4.8 MB → 2.56 MB, for a 241 KB CSV.
  - Breaking changes are listed in `docs/release-notes/unreleased.md`:
    overview finding IDs follow rank, 0.1.x fingerprints no longer match, and
    the compact format and `exact_grain` are gone.
- 2026-09-23: **released as v0.2.0** at `eb45a2d` (release PR #4), published to
  PyPI, and `fieldwork-docs` was synchronized (PR #5). See
  `docs/releases/v0.2.0.md`. Steps 4 and 5 start from this release.
- 2026-09-23, **step 4 complete** on branch `test-suite-rebalance` (not yet
  merged or released).
  - **Oracles and gap tests, added first:** `fba8556` (shared
    `tests/oracle.py`; the count oracle now covers None/NaN/`pd.NA`/NaT and
    1–4 columns × 0–40 rows), `4cda52a` (dependency-discovery groupby oracle:
    composite keys, `dropna` both ways, contexts, scopes; records, exception
    groups, finding exceptions, candidates), `47ad37a` (pairs known answers),
    `1417cbc` (native-dtype matrix; adds pyarrow to the dev group), `c284d96`
    (value_patterns), `60cdfde` (row-permutation invariance for missingness,
    dependencies and suggest_paths), `7046597` (census options as properties),
    `9f2854f` (compare), `b1b3f39` (50k-row smoke test, marked `slow`).
  - **Docstring and editor checks replaced** (`bcf59e9`): `test_inline_docs.py`
    (103 items) and `test_editor_api.py` (22 jedi probes) are gone. In their
    place: one smoke test (`tests/test_public_api.py`: docstring, annotated
    parameters, return annotation, no underscore parameters for every export,
    its public members and `Path`), `--doctest-modules` over `src/fieldwork`
    (configured in `pyproject.toml`; the same 27 examples), and
    `tests/typing/public_api.py` under pyright, unchanged. The CI wheel step
    now runs `pytest tests/test_public_api.py --doctest-modules --pyargs
    fieldwork` without jedi, and jedi is no longer a direct dev dependency.
    **`AGENTS.md` was not changed:** it still asks for editor signature and
    completion checks, and `docs/inline-api.md` "Writing public
    documentation" still states the full NumPy policy. Step 5.6 should relax
    both together.
  - **Brittle and redundant tests** (`6dcd3e9`, `bab046b`, `091e5e4`,
    `e955f13`): `tests/test_rendering_contracts.py` now holds, once for 13
    result kinds × 4 media, renderer validity and non-mutation, live-versus-saved
    equality, escaping, HTML ID/link integrity, topology invariance to
    quantities and an injected-quantity allowlist. Every brittle example listed
    above was removed or rewritten structurally, and the duplicates were dropped.
    The benchmark-normalizer test is gone. The notebook test now checks only
    the narrative's stated conclusions (plus companion agreement).
  - **Counts and runtime:** before, 356 collected items in about 3.4–3.9 s, of
    which 125 were docstring and editor items. After, 337 items (310 tests plus
    27 doctests) in about 8.2 s, or 6.9 s with `-m "not slow"`. Most of the added
    time is Hypothesis oracles (about 2.5 s), the rendering contracts (1.4 s)
    and the smoke test (1.3 s). pyright, ruff, the example, `generate_assets.py
    --check` and the installed-wheel checks pass.
  - **Bug found and fixed** (`80aa4ba`, recorded in
    `docs/release-notes/unreleased.md`): `value_patterns` string summaries
    were `most_common()` tuples in memory but lists once saved. A live result
    rendered HTML differently from its own export, and `to_dict()` did not
    equal the saved payload. They are now lists, and the docstring example
    changed to `[['A9', 2]]`. Sync that example in fieldwork-docs if it is shown
    there.
  - **Observed, not changed (step 5 should decide):**
    - Numeric summaries (`numeric_range`, offset, ratio) require a numeric
      dtype, so an object column of numbers gets none. Every other analysis
      treats the same values identically in either dtype.
    - Inside a grain view, each key→target record keeps its own pair population
      (for example 4 rows in a 2-row view). Only key-to-key comparisons use the
      view's common rows. This matches `grain()` but is easy to misread.
    - `value_patterns(by=[...])` emits a trivially constant
      `context_constancy` finding for each context column itself.
    - Timedelta labels such as `-0 days 00:00:05` carry the sign as a prefix.
      `pd.Timedelta` parses them as +5 s.
    - `exact_pair_ids` numbers pairs in sorted order, not first-observed order.
  - **For step 5:**
    - The oracle, property, dtype, permutation, rendering-contract and scale
      tests are the safety net, and they compare exported values through
      `oracle.record_token`. When 5.2 replaces the tagged scalar encoding,
      update `record_token` and the literal `{"type": …, "value": …}`
      records in `test_contracts.py`, `test_resolved.py`,
      `test_grain_graph.py`, `test_context_adapter.py`, `test_journeys.py`,
      `test_workflow.py` and `test_dependency_support.py`, not the oracles.
    - The contracts file enumerates result kinds in `KINDS`. When the result
      model is unified (5.1), adapt that list; its assertions name no renderer
      wording.
    - Private imports remain only where justified: `modal_groups`/`group_ids`
      and `exact_pair_ids` with property oracles, `_runtime`/`evidence` session
      internals for cleanup, `render._width_function`/`_fallback_width` for
      Unicode width, and `resolve_result`.
    - The legacy fixture and its tests remain for 5.4.
