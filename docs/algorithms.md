# Discovery algorithms and budgets

## Availability

Features are encoded once per operation. Repeated boolean availability signatures
are ranked by descending row count with lexical signature ties. `max_signatures=50`
limits stored signatures, with omitted row mass reported. Identical masks form
families even for always-missing or always-present columns.

Pairs are enumerated in input-column combination order, bounded by `max_pairs=200`.
For A and B, presence Jaccard is both-present / either-present. If neither is ever
present, it is undefined, never perfect similarity. Agreement additionally includes
co-absence and is reported separately. A implies B has conditional presence
both-present / A-present, an exception rate, and B's baseline presence. No antecedent
support means no implication finding. Similarity and implication thresholds default
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

A redundant step leaves the prefix count unchanged. Lower scores rank first.
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
three-character prefixes and lengths. `max_patterns=10` bounds displayed counts.
Indexed-name families are explicitly name evidence, augmented by identical presence
when observed. Numeric summaries use finite values, observed minimum spacing, and
an allclose grid check. Offset and ratio checks require at least two finite paired
rows; ratios exclude zero denominators. Tolerances are rtol 1e-5 and atol 1e-8.
They are simple measured relationships, not fitted latent models. Context constancy
reports how many populated context groups have a single populated target value.

### Population-compatible grain views

Discovery builds one view per distinct, nonempty candidate complete-case mask.
Each view includes candidates whose supported rows contain that mask. The anchor
candidate guarantees their common complete-case population is exactly that mask;
the foundation still recomputes and checks every combined relationship on it.
Views are ordered by descending population, then candidate enumeration order.
`exact_grain` is the first view; `grain_views` retains every view, candidate IDs,
source positions and population accounting. No relation is composed across views.
All candidates remain in `candidates`, with view membership; unsupported candidates
also appear in `graph_selection.excluded` with `no_evaluated_support`. With
`dropna=False`, all candidates share the scoped population.

Presentation ranks supported repeated groupings before unique identifiers,
constants, and candidates without evaluated support. Within each class, more exact
determined targets and repeated rows rank first, with shorter keys and lexical
order breaking ties. A one-group candidate is a constant, including a singleton
population. Ranking changes presentation only; the complete evaluated set remains
available in enumeration order. Conditional findings carry typed context predicates
in `structure.context`; readable statements name the feature, value and scalar type.
Topology retains these predicates while suppressing measurements and selectors.
