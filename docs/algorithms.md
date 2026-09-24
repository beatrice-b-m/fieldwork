# Discovery algorithms and budgets

Budgets written `name=default` below are keys of each analysis's `limits`
mapping, except `max_key_size`, `max_dimensions` and `n_paths`, which are
ordinary arguments.

## Availability

Needed value columns are encoded once per call context; native presence-only
columns use missing masks. Row presence or explicit any/all entity
aggregation provides the analysis masks. Repeated boolean availability signatures
are ranked by descending analysis-unit count with lexical signature ties. `max_signatures=50`
limits stored signatures, with omitted row mass reported. Identical masks form
families, including always-missing columns. Always-present columns form no family
and receive no per-feature availability finding; they remain in the `availability`
table with their counts.

Pairs are enumerated in input-column combination order, bounded by `max_pairs=200`.
For A and B, presence Jaccard is both-present / either-present. If neither is ever
present, it is undefined, never perfect similarity. Agreement additionally includes
co-absence and is reported separately. A implies B has conditional presence
both-present / A-present, an exception rate, and B's baseline presence. No antecedent
support means no implication finding. Implications are omitted when they are
vacuous: B is present in every analysis unit, or A and B have identical
availability (their family finding already states it). Similarity findings are
likewise omitted when either feature is always present, since the similarity then
only restates the other feature's populated fraction. Similarity and implication thresholds default
to 0.8 and 0.9. Mutually exclusive pairs require each field to have observed support
and no co-presence; exact families let users interpret exclusive field groups.

Context groups are joint combinations in first-observed order, bounded by
`max_contexts=32`. Per-feature entity counts operate over all eligible distinct
keys, without a display truncation affecting their denominator.

## Dependencies and candidate grain

Enumerate determinants by size then input-column order, up to `max_key_size=2` and
`max_candidates=100`. Report the combinatorial candidate space and tested count.
Every selected non-key feature is a target; all rows are evaluated. User-specified
contexts add up to `max_contexts=32` conditional populations plus the global population.

Modal accuracy = 1 − minimum rows needing target-value repair / evaluated rows.
Within each determinant group, choose its most frequent target; canonical code order
breaks ties. Nonmodal rows are representative exceptions. Also retain all violating
groups, affected rows, group violation rate, and repeated-group support. This
separates approximate mapping quality from coverage and singleton effects.
`min_accuracy=0.95` controls finding emission, not which tests are computed.

Candidates report groups, uniqueness, repeated groups/rows, complete-case exclusions,
and exactly determined targets. A unique row ID therefore remains distinguishable
from a useful repeated entity grouping. Conditional dependencies use their own
populations. The embedded foundation grain graph admits exact dependencies only,
collapses equivalent candidates, retains cross-cutting structure, and checks
population compatibility. Graph work is additionally bounded by the same candidate
set; pairwise candidate comparisons may dominate runtime.

## Census paths

`max_features=20` bounds the eligible input features (steering columns survive the
budget), and `max_pairs=200` bounds pair nesting inference. Constants are omitted
unless required. Equivalent pairs are retained as aliases. If fine determines coarse
but coarse does not determine fine, the soft nesting edge is coarse → fine.

A deterministic beam search extends ordered prefixes; it defaults to four dimensions,
beam width 12, 200 evaluated extensions, three returned paths, and display budget 40.
Input order defines feature coverage; lexical path order breaks score ties. If the
search budget ends, shorter valid paths may be returned. Coverage records returned
and requested depth; unsatisfied required columns can yield no recommendation.

For each path, count observed distinct prefixes at every depth. Base score is:

```text
sum(prefix_counts) / display_budget
+ sum(max(0, prefix_count - display_budget)) / display_budget
+ 2 * redundant_steps
+ 3 * equivalent_pair_steps
```

A redundant step leaves the prefix count unchanged. Lower scores rank first. The
weights (2, 3, and 8 and 12 below) are hand-tuned heuristics, named in
`src/fieldwork/navigation.py` (`REDUNDANT_STEP`, `ALIAS_STEP`, `NESTING_INVERSION`,
`SEPARATION`); they were chosen so that nesting and separation dominate prefix size,
not fitted to data.
Structure and context objectives add eight per reversed supported nesting edge.
Compact uses the base score. Target adds twelve times summed within-prefix modal
impurity of the target values. Availability adds twelve times the same impurity for
full boolean signatures. Impurity is rows outside each group's mode / input rows.
The target itself is excluded from browsing candidates unless explicitly required.

`start_with` fixes the leading dimensions. `before` enforces acyclic precedence and
includes its referenced features; `exclude` removes candidates. Conflicting or
unfittable steering raises `ValueError`. Context requires `start_with`; target
requires `target`. These are observed-prefix heuristics, not an exhaustive optimizer
or an order-invariant joint-information score.

## Value patterns

String formats replace digit runs with `9` and ASCII letter runs with `A`; report
three-character prefixes and lengths. `max_patterns=10` bounds displayed counts;
`min_count` omits formats, lengths and prefixes seen in fewer rows. A string
pattern finding's `structure` lists the reported formats alphabetically, without
counts, and whether others were omitted. Topology keeps that list and drops
lengths and prefixes, since prefixes can reveal identifier fragments.
Indexed-name families are explicitly name evidence, augmented by identical presence
when observed. A column is numeric when every populated value is a non-boolean
number, whatever its dtype, so an object column of numbers is summarized like its
numeric equivalent. Numeric summaries use finite values, observed minimum spacing,
and an allclose grid check. Offset and ratio checks require at least two finite paired
rows; ratios exclude zero denominators. Tolerances are rtol 1e-5 and atol 1e-8.
They are simple measured relationships, not fitted latent models. Context constancy
reports how many populated context groups have a single populated target value; the
context columns themselves are not tested.

### Population-compatible grain views

Discovery builds one view per distinct, nonempty candidate complete-case mask.
Each view includes candidates whose supported rows contain that mask. The anchor
candidate guarantees their common complete-case population is exactly that mask;
the foundation checks every combined relationship on it, reusing exact metrics
only for identical eligible populations.
Views are ordered by descending population, then candidate enumeration order.
`grain_views[0]` is the primary view; `grain_views` retains every view, candidate
IDs and population accounting. A view's population is the complete cases of its
`anchor_candidate_id` within the scope (or the whole scope with `dropna=False`), so
it stores bounded `examples` rather than every source position. No relation is composed across views.
All candidates remain in `candidates`, with view membership; unsupported candidates
also appear in `graph_selection.excluded` with `no_evaluated_support`. With
`dropna=False`, all candidates share the scoped population.

Inside a grain result (standalone or in a view) there are two populations. Each
key→target record in `dependencies` counts its own complete cases for that key and
target, as a standalone test would, so it can count more rows than the view. The
graph's `tests` table, nodes and placements compare keys on the graph's common
rows (`graph.evaluated_rows`). Read placements from the graph and per-pair support
from `dependencies`.

Presentation ranks supported repeated groupings before unique identifiers,
constants, and candidates without evaluated support. Within each class, more exact
determined targets and repeated rows rank first, with shorter keys and lexical
order breaking ties. A one-group candidate is a constant, including a singleton
population. Ranking changes presentation only; the complete evaluated set remains
available in enumeration order. Conditional findings carry typed context predicates
in `structure.context`; readable statements name the feature, value and scalar type.
Topology retains these predicates while suppressing measurements and selectors.

### Recommendation reasons and diversity

Every path stores `reasons` for observed prefix branching/overflow, supported
nesting, redundant dimensions and equivalent partitions, and availability
separation. Target searches also report target separation. Separation is the
within-prefix nonmodal row fraction at each depth; explanations quote these
measured values and the actual feature names. Numerical reasons stay in full
presentations; topology retains only structural path and alias findings.

At each search depth the beam keeps the best order per selected feature set,
reserving slots for different browsing choices. Future extensions depend on that
set, while accumulated prefix costs retain the ordering evidence. Returned paths
also collapse alias substitutions. Fewer than `n_paths` are returned when no
meaningfully different evaluated feature sets exist; reverse permutations are not
advertised as alternatives. Explicit start order and precedence remain binding.

### Connected feature evidence

The overview includes value-pattern discovery and a `feature_network` with
feature nodes, typed relationships, and connected components. Availability
identity/similarity/implication/exclusion, indexed names, equivalent value
partitions and exact/approximate dependencies remain separate relationship types.
Trivially true dependencies (a constant target, or a determinant that is unique
within the target's evaluated rows) remain findings but are not network edges;
otherwise they would connect every feature without describing structure.
A connected component means reachability through this evidence, not equivalence
or a composed functional dependency. Composite determinants and typed context
predicates remain explicit. Each relationship links to its section finding,
overview finding, counting unit and population reference. Pair-specific FD
populations stay in the referenced finding; the network does not replace the
compatible exact grain views. `overview.relationships(feature, kinds=[...])`
returns a dataframe for filtering and following evidence to `inspect`/`select`.
Saved HTML provides feature disclosures and links to the supporting findings.
Topology removes evidence pointers and populations while retaining relation types,
direction and context. Its relationship order is canonical.

### Overview lead ranking

Overview leads rank the sections' findings by a heuristic score
(`src/fieldwork/leads.py`) and are numbered in that order, so `f0` is the most
promising lead. Each carries `lead.score` and a short `lead.reason`, and refers to
its section finding instead of copying it (`Result.findings` resolves them). The score favors evidence an analyst would
want to explain: near-rules with repeated support and a few exceptions, mutually
exclusive or empty columns, presence rules with exceptions, mixed string formats,
equivalent encodings and partially populated columns. Trivially true or purely
descriptive findings (constant targets, unique determinants, uniform formats,
numeric ranges, census paths) rank last. After the first finding of a pattern on a
given leading column, further ones are halved (all equivalent-encoding findings
after the first, since they chain across columns), so one near-key determining
many targets does not crowd out other leads. Scores are for ordering only; they are not
probabilities or measurements. Section results keep their own order and IDs.

## Explicit work budgets and exact kernels

Overview `sections` and per-section `options` select and configure operations
independently. Dependency discovery adds `include_grain` and the `max_grain_views` and
`max_dependency_tests` limits; omitted graph views and candidate/target/context tests have
explicit coverage metadata. Defaults preserve previous work and result ordering.
See [usage and contracts](performance.md#choose-the-work-you-need).

Availability uses packed boolean signatures with frequency/lexical tie ordering.
Determinants and path prefixes use dense first-observed integer group IDs; modal
counts preserve canonical-code ties. Path search computes objective-relevant
impurity during search and complete explanation metrics for retained paths.
Context constancy uses group membership counts, and repeated string summaries
weight unique strings while preserving first-observed ties. Graph subset caches
key exact packed populations; matching row counts alone never equate unrelated
subsets. Bounded example extraction retains complete totals and first source rows.

## Dependency target coverage and repeated support

For each determinant/context, let P be the scoped context population, Q the
complete determinant cases (`dropna=True`) or all P (`False`), O the rows of Q
with an observed target, and E the evaluated population: O when dropping missing
values, otherwise Q. Native and declared sentinel missingness share the existing
presence policy. Group E by the encoded determinant; R contains rows in groups
of at least two **after target exclusions**.

Dependency records and finding measurements expose `determinant_evaluated_rows`
(Q), `target_observed_rows` (O), `target_coverage` (O/Q),
`target_missing_excluded_rows` (Q−E), `repeated_rows` (R), `repeat_coverage` (R/E),
and `repeat_modal_accuracy` (1−repair_rows/R). All repair rows belong to repeated
groups. Zero denominators give None/JSON null. Existing `missing_excluded_rows`
still means P−E. With `dropna=False`, missing categories participate in consistency
measurements: E=Q and target exclusions are zero, but observed target coverage
can be below one. Determinant eligibility does not then imply observed values.

Dependencies remain row-counted and row-weighted. There is no entity aggregation,
macro averaging, or resampling. Counting unit, aggregation rule, weighting, and
resampling unit are distinct choices. Exactness describes the evaluated rows;
repeated support describes consistency beyond singleton groups; neither establishes
entity meaning or statistical reliability. One repeated group qualifies descriptively.

Candidate `determines` retains ordered global exact targets, including singleton-only
tests. `determines_with_repeated_support` includes global exact targets with at
least one repeated group in that target's E. Conditional evidence stays attached
to its typed context. Candidate `repeated_rows` continues to refer to Q before
target exclusions. `global_targets_tested` counts completed global tests, including
empty tests; `global_targets_possible` counts selected non-key targets. Conditional
and graph work never inflate those counters. Empty lists under incomplete budgets
mean nothing was established by the completed work. `min_accuracy` only filters
findings: the `dependencies` table retains every completed test.

Full presentation retains the role order: repeated groupings, unique identifiers,
constants, then candidates without evaluated support. Within a role it sorts by
the count of global exact targets with repeated support, descending determinant
repeated rows, key size, and lexical columns. Raw exact-target count is no longer
a tie-breaker; global test coverage is disclosed rather than multiplied into a
score. This intentional presentation change does not reorder analytical candidates
or findings. Ranking remains budget-sensitive and is not a completeness guarantee.
Unique identifiers remain valid row-grain candidates.

Each dependency finding's `structure` records its `strength` (`exact` or
`approximate`, as its pattern does) and `repeated_support`: whether E has a
repeated group. An exact finding without repeated support holds only because
every determinant group is one row. Grain tests carry the same flag for their
key groups, and each candidate or grain key has a structural role: `unique
identifier` (no repeated group), `repeated grouping`, `constant` (one group) or
`no evaluated support`. Topology exports keep these qualitative fields and drop
the counts behind them.

Full candidate summaries show both exact-target counts, determinant group counts,
and global tests completed/possible. Standalone full dependency projections and
renderings expose all completed per-target records within display limits, including
below-threshold tests; the overview stays compact. Use `to_frame('dependencies')`
for the complete table or render the overview's dependency section for detail.

## Relations across tables

`relate(left, right, on=..., compare=...)` never joins the tables. Each side is
prepared under its own scope and sentinels, the key and compared columns are
mapped into identities shared by both sides, and every measurement comes from
per-key row counts, so the work is linear in rows plus distinct values.

**Matching values.** With `match="typed"` (default), booleans, numbers, strings,
dates, naive datetimes, aware datetimes (as UTC instants) and timedeltas never
match each other, and numbers match numerically (`1` matches `1.0`, also within
one side). Within one table these stay distinct levels, but across tables an
integer key often arrives as floats, because a missing value turns a pandas
integer column into a float column. `match="text"` also writes integer-valued
numbers as decimal text, so `123` matches `"123"`. Text is compared exactly:
`"0123"` does not match `123`, and non-integral numbers keep typed identity.
Compared attributes use the same rule as keys. When the value kinds seen on the
two sides of a key or compared pair do not overlap, the result carries a
`VALUE_KIND_MISMATCH` warning naming both columns and their kinds.

**Keys.** A row's key is its tuple of `on` columns. Rows with any missing key
column (native or declared sentinel) are excluded and counted per side as
`incomplete_key`. Keys are distinct complete tuples. A key is matched when it
occurs on both sides.

**Coverage** (each direction). Of the source side's distinct keys, `found` are
matched. The state is `all`, `some`, `none`, or `empty` when the source side has
no complete key. Row counts (`found_rows`, `not_found_rows`) are reported
alongside; the unit is keys.

**Relation.** Over matched keys, a side is "many" when any matched key has more
than one row on it, giving `1:1`, `1:n` (one left row, several right rows), `n:1`
or `n:m`. Unmatched keys do not affect the type. `joined_rows` is
the sum over matched keys of left rows × right rows, the size of an inner join.
With no matched key, the type is null and there is no finding.

**Agreement** (per compared pair, over matched keys). Collect each side's
distinct compared values for the key, ignoring missing values when
`dropna=True` or treating missing as one shared value when `False`. A key is
`unavailable` when a side has no value, `ambiguous` when a side has several, and
otherwise agrees or disagrees. The state is taken over agreeing plus disagreeing
keys (`empty` when there are none); `ambiguity` is `some` when any key is
ambiguous. Keeping ambiguity apart means that variation within one table is
never reported as disagreement between the tables.

**Self-references.** With `right=None`, both sides read `left`: the left
columns of `on` are references and the right columns are the key each row is
referenced by. Coverage then reads as "references resolve" and "keys are
referenced". Each left row with a complete reference and a complete own key
forms an edge own → reference. Edges whose target key exists on the right side
are resolved; self-loops (own = reference) are counted and excluded. A resolved
edge a → b is reciprocated when some right-side row with key b references a.
Reciprocity counts distinct edges, and its state is `empty` when every resolved
edge is a self-loop.

**Row samples.** Each finding's examples and exceptions belong to one side
(`selector["side"]`), and two-sided findings add `other_side`:

| Finding | Sides | Examples | Exceptions |
| --- | --- | --- | --- |
| Coverage | source side | rows whose key is found | rows with a complete key not found |
| Relation | both | rows of matched keys repeated on that side | rows of matched keys that occur once on that side |
| Agreement | both | rows of agreeing keys | rows of disagreeing keys |
| Reciprocity | left | rows whose edge is reciprocated | rows whose resolved edge is not |
