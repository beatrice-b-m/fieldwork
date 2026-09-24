# Design: relating two tables (`fw.relate`)

Status: implemented on branch `worktree-cross-table-relate`, for
[issue #23](https://github.com/beatrice-b-m/fieldwork/issues/23).
`docs/contracts.md#relating-two-tables` and
`docs/algorithms.md#relations-across-tables` now describe the behavior; retire
this page when the branch merges. The user guide in fieldwork-docs still needs a
relate page.

## Goal

Answer the structural questions that span two tables (or a table and itself)
without joining them by hand. Every answer is available as a count-free state,
so it fits the topology contract:

| Question | Output | Unit |
| --- | --- | --- |
| Key coverage: are A's keys found in B, and B's in A? | `all` / `some` / `none` / `empty` per direction | distinct complete keys |
| Cardinality on matched keys | `1:1` / `1:n` / `n:1` / `n:m` | matched keys |
| Agreement of a compared attribute on matched keys | `all` / `some` / `none` / `empty`, plus `ambiguity` `some` / `none` | matched keys |
| Self-reference reciprocity | `all` / `some` / `none` / `empty`, plus `self_references` `some` / `none` | distinct resolved references |

Out of scope for this change: attribute placement across a join (grain over a
fanned-out population), automatic discovery of join keys, adding `relate` to
`explore` or lead ranking, and value transforms such as "same calendar day"
(derive the column before calling).

## Public interface

```python
fw.relate(
    left: pd.DataFrame,
    right: pd.DataFrame | None = None,        # None: self-reference within left
    *,
    on: Mapping[str, str],                    # left column -> right column; several = composite key
    compare: Mapping[str, str] | None = None, # left column -> right column, tested on matched keys
    match: Literal["typed", "text"] = "typed",
    dropna: bool = True,
    scopes: tuple[Scope | None, Scope | None] | None = None,
    missing: tuple[MissingMap | None, MissingMap | None] | None = None,
    table_ids: tuple[str, str] | None = None, # default ("left", "right"), or ("table", "table") for a self-reference
    limits: RelateLimits | None = None,       # {"example_limit": 5}
    **runtime: Unpack[Runtime],
) -> Result                                   # kind "relation"
```

Each side keeps its own scope, sentinels and table ID. The two tables are
never joined into one frame; every measurement is computed from per-side key
codes.

A self-reference (`right=None`) reads both sides from `left`. The left columns
of `on` are the reference columns, and the right columns are the key each row
is referenced by. For example, `on={"linked_acc": "acc"}` asks whether a row's
`linked_acc` names another row's `acc`. The sides can still have different
scopes: references from 2024 rows can be resolved against targets anywhere.

### Row access on two sources

`Result.inspect`, `Result.select` and `Result.recompute` still take the left
frame as `df`. The first two gain two keyword arguments that only relation
results accept:

- `side="left" | "right"` picks the table whose rows are returned. The returned
  Scope belongs to that table.
- `right=` is the right frame. It is required when `side="right"` and for any
  complete selection (`select`, `inspect(all_matches=True)`), because matching
  needs both key sets. A self-reference never needs it.

Both frames are verified against their saved fingerprints. `recompute(left,
right=right)` and `Recipe("relate", {...}).run(left, right=right)` pass the right
frame as an ordinary override. A Recipe saves neither frame nor scope.

## Matching values across tables

Within one table, `1`, `1.0` and `"1"` are different values. Across tables,
integer keys often arrive as floats because a missing value in pandas turns an
integer column into a float column, and CSV exports turn identifiers into text.
Two named modes cover this:

- `typed` (default): booleans, numbers, strings and temporal kinds never match
  each other, and numbers match numerically (`1` matches `1.0`). This is the
  rule sentinels already follow.
- `text`: integer-valued numbers are compared by their decimal text, so `123`
  matches `"123"`. `"0123"` still does not match `123`, and other values keep
  typed identity.

Compared attributes use the same mode as the keys. When a key or compared pair
has value kinds on each side that share nothing (for example only strings on the
left and only numbers on the right), the result carries a
`VALUE_KIND_MISMATCH` warning. It names both columns and their kinds, never
values, so a type mismatch can't pass silently as "no key is found".

## Semantics

Rows with any missing key column (native missing or a declared sentinel) are
excluded from matching and counted per side. Keys are distinct complete key
tuples. A key is *matched* when it occurs on both sides.

- **Coverage** (per direction): of the distinct keys on the source side, how
  many are found on the other side. `empty` when the source side has no complete
  key. Examples are source-side rows whose key is found. Exceptions are
  source-side rows whose complete key is not found.
- **Relation**: over matched keys, a side is "many" when any matched key has
  more than one row on that side. The finding records `joined_rows`, the row
  count an inner join would produce. For each side, examples are rows of matched
  keys repeated on that side, and exceptions are rows of matched keys that occur
  once on that side. There is no relation finding when no key matches.
- **Agreement** (per compared pair): for each matched key, collect the compared
  values on each side. With `dropna=True`, missing values are ignored. With
  `dropna=False`, missing is a value. Each matched key has one state:
  - `unavailable`: one side has no value.
  - `ambiguous`: either side has more than one value.
  - `agree` or `disagree`: both sides have exactly one value.

  The state (`all`, `some`, `none` or `empty`) is computed over agreeing plus
  disagreeing keys. Ambiguity inside one table is kept separate from
  disagreement between tables. For each side, examples are rows of agreeing
  keys and exceptions are rows of disagreeing keys.
- **Reciprocity** (self-reference only): each referencing row with a complete
  reference and a complete own key contributes an edge `own → reference`.
  Resolved edges whose target exists are tested for a reverse edge among the
  right-side rows. Self-loops (`own == reference`) are counted separately and
  excluded. The unit is distinct edges. Examples are referencing rows whose edge
  is reciprocated; exceptions are those whose edge is not.

## Result payload (kind `relation`)

- `source`, `scope`, `missing_convention` describe the left side, so generic
  code keeps working. `sides.left` and `sides.right` each hold `source`,
  `scope`, `missing_convention`, `key` and `compare` columns, plus row and key
  accounting: evaluated rows, rows with an incomplete key, distinct, matched and
  unmatched keys, and matched and unmatched rows.
- `self_reference`, `parameters` (`on`, `compare`, `match`, `dropna`, `limits`),
  `key_coverage` (`left_to_right`, `right_to_left`), `relation`, `agreement`
  (one record per compared pair), `reciprocity` (self-reference only, else
  null), `warnings` and `findings`.
- Findings have the patterns `key_coverage`, `key_relation`, `value_agreement`
  and `reference_reciprocity`. Their features are table-qualified per side.
  `examples`/`exceptions` belong to the finding's primary side
  (`selector["side"]`). Findings with rows on both sides also carry
  `other_side` with that side's bounded samples. Structure keeps the
  qualitative state (`direction`, `coverage`, `relation`, `agreement`,
  `ambiguity`, `reciprocity`, `self_references`), which topology exports keep.
  Measurements carry the counts.

## Presentation

`relation` is a findings kind: the generic finding cards in text, SVG and HTML
render it. The projection adds each side's table ID, key columns and scope
(name and parent only in topology), plus the warnings. Topology statements use
only table IDs, column names and states.

## Plan (all steps done)

1. Design doc (this page).
2. `relate` implementation in `src/fieldwork/relate.py`, exported from
   `fieldwork`, with `RelateLimits`, `MatchMode` and `Side` types.
3. `Result.inspect`/`select`/`recompute` side-aware row access. `relate` added
   to Recipe operations, and Recipes reject saved `scopes`.
4. Presentation: projection of sides and warnings, and relation labels in text,
   SVG and HTML.
5. Tests: known answers for each measurement, a pandas-merge oracle for coverage,
   relation, joined rows and agreement, dtype-mismatch and `text` matching,
   sentinels and scopes per side, self-reference reciprocity, selection and
   inspection on both sides, source-mismatch rejection, round trips and recipes,
   the rendering-contract kinds, the typing consumer and the public API smoke
   test.
6. Durable docs: contracts, algorithms, architecture module map and extension
   note, developer map. Then retire this page.
