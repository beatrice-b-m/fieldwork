# One single-table investigation

The [worked notebook](../examples/investigation.ipynb) follows a concrete question:
which exams in a synthetic export have both image references on every export slot,
and does a corrected delivery resolve an incomplete exam? It includes saved outputs,
interpretation, and the decisions that connect each step. The
[Python companion](../examples/investigation.py) runs the same investigation.

The example's specification expects paired images for MR slots and dose measurements
for CT slots. This is an export-specific rule, not a general modality rule. Eligibility
is selected from the MR context independently of image availability, so incomplete
records remain in the denominator. A `-999` sentinel is treated as a missing second
image reference throughout.

| Question | Initial delivery | Interpretation |
| --- | --- | --- |
| How many MR slots have both references? | 7/8 (87.5%) | Row completeness; larger exams contribute more weight |
| How many MR exams have any second-image reference? | 4/4 (100%) | At least one populated slot can conceal incomplete exams |
| How many MR exams have complete pairs on every slot? | 3/4 (75%) | With the first image present everywhere, `entities` + `all` for the second image answers the acceptance question |

Inspection identifies South / S01 / slot 2, while an entity selection retrieves
both slots of the affected exam. The site-to-exam census organizes eight eligible
rows into four exams. Reapplying a saved recipe to a simulated corrected delivery,
with a newly selected MR scope, yields 4/4 complete exams: a 25-percentage-point
increase. Reference availability does not establish file existence or image quality.

The notebook also contrasts overall availability (7/12 rows) with MR availability
(7/8 rows). That change reflects a population restriction in one delivery, not an
improvement over time. In either kind of comparison, interpret fractions alongside
their populations, denominators, units, and missing-value conventions.

## Explore for leads

Start with `overview = fw.explore(df)`. Its compact summary opens with ranked
leads (`f0` first, each with the reason it was ranked), then candidate grains with
equivalent columns merged, census recommendations, availability families, and
availability signatures described by the columns they are missing.
`overview.relationships("image_1")` lists typed connections to other features and
the finding IDs behind them. Indexed names, shared availability and exact mappings
remain distinct evidence. Follow one relationship to `overview.inspect(df, id)`.

## Discover and inspect a signature

```python
availability = fw.missingness(
    df, features=["image_1", "image_2"], by=["site"],
    entity="exam_id", missing={"image_2": [-999]}, example_limit=3,
)
signature = next(s for s in availability["signatures"] if "image_2" in s["absent"])
examples = availability.inspect(df, signature["finding_id"])
scope = availability.select(df, signature["finding_id"], name="missing second image")
all_rows = availability.inspect(df, signature["finding_id"], all_matches=True)
```

Examples are bounded source rows. The selected scope contains every matching row,
including duplicate index labels. It is bound to the ordered source values; changes
to that source require a new analysis. `scope.refine(df, positions, name=...)`
restricts it further. Context findings retain their predicates, such as
`site = North`, in text, HTML and topology; a string that could be read as a
number is quoted (`site = '1'` versus `site = 1`). Entity pattern findings can
select all rows belonging to entities with some, all, one, any or no populated rows.

## Refine, compare and navigate

```python
local = fw.missingness(df, scope=scope, missing={"image_2": [-999]})
baseline = fw.missingness(df, missing={"image_2": [-999]})
change = fw.compare(baseline, local)
paths = fw.suggest_paths(
    df, scope=scope, missing={"image_2": [-999]},
    features=["site", "exam_id", "image_2"], start_with=["site"],
)
if paths.best is not None:
    print(paths["paths"][0]["explanation"])
    tree = paths.best.census(df)
```

Path reasons name actual nesting, observed branching, redundant steps, and
availability or target separation. Alternatives use different feature sets;
steering order remains authoritative. The census handoff preserves scope and
sentinels. Its population accounting separates restrictions and missing exclusions
against the original source. `path.dimensions` alone is an ordinary tuple and
carries no context.

Candidate grains distinguish repeated groupings, unique identifiers, constants
and candidates without evaluated support. In overview presentations, single-column
candidates that determine each other (the same partition) are listed once with
their `equivalent` columns; the saved candidate records are unchanged. `dependencies["grain_views"]` retains
separate compatible populations for sparse keys. Inspect a view's `population`
and render its `grain` with the standard foundation renderer. Relationships are
combined only within a compatible view; excluded candidates retain their evidence.

## Choose the counting unit

```python
row_evidence = fw.missingness(df, entity="exam_id", unit="rows")
entity_evidence = fw.missingness(
    df, entity="exam_id", unit="entities", entity_presence="any",
)
```

Rows each contribute one observation by default. With entity units, distinct
populated keys each contribute one observation, regardless of row count. `any`
means a feature is present on at least one entity row; `all` requires every row.
Missing entity keys are excluded and counted. These choices apply to availability,
implications, similarity, families, signatures and context availability. A matching
entity's selection includes all its scoped source rows.

For 100 matching rows from one entity and one exception from another, conditional
presence is 100/101 with row units and 1/2 with entity units. A context analysis
aggregates each entity within that context. Comparisons require compatible units
and aggregation. `explore(df, discovery={"entity": "exam_id", "unit": "entities"})`
uses entity availability alongside row-based path and dependency evidence.

## Save and reapply

```python
import json

saved = json.loads(json.dumps(availability.to_dict(), allow_nan=False))
restored = fw.Result.from_dict(saved)
html = fw.render_html(restored)  # Works without a source dataframe.
restored.select(df, signature["finding_id"])

recipe = fw.Recipe(
    "missingness",
    {"entity": "exam_id", "unit": "entities", "missing": {"image_2": [-999]}},
)
recipe.save("availability-recipe.json")
next_evidence = fw.Recipe.load("availability-recipe.json").run(next_delivery)
```

Results preserve source-bound evidence; recipes reapply settings to new deliveries.
Pass a source-bound scope as a run override rather than storing it in a recipe.
This also works for a configured automatic overview:

```python
overview_recipe = fw.Recipe(
    "explore",
    {"discovery": {"max_candidates": 10, "entity": "exam_id", "unit": "entities"}},
)
selected_overview = overview_recipe.run(df, scope=scope)
```

The run uses the selected population in every section and retains the recipe's
search and entity settings. Common run overrides (`scope`, `missing`, `table_id`,
`features`) take precedence over their configured `discovery` values.
Saved path results restore `best.census` as well. Topology exports retain feature
labels, relation types and context predicates while suppressing measurements,
source positions and population identifiers. Related-table discovery remains deferred.
