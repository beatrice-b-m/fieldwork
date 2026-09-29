# Exploring saved outputs

This page describes the output interfaces in Fieldwork 0.2.1. The separate
documentation site records its release and source commit in `docs-source.json`.

## HTML evidence reports

`fw.render_html(result)` returns a standalone document. Save it with
`Path("report.html").write_text(fw.render_html(result), encoding="utf-8")` and open
it in a browser. No network, original dataframe, or Python session is needed to
browse the included evidence. Reports remain snapshots, not analysis archives.

Discovery reports separate population, search coverage, visual summary, and
inspectable records. Search and expand/collapse controls operate independently in
each list. Findings also support pattern and exception-row filters. Live counts
refer to records included in that list, not rows in the original dataframe.
Evidence links clear conflicting filters, open the target, and focus its summary.

`max_findings` bounds findings, candidate grains, and completed dependency tests
independently. Search cannot recover records excluded by this display limit;
render again with a larger limit. The visual summary has its own cap of 12 items
per list and at most five candidate grains. A search with no matches differs from
an analysis with no saved findings and from a report with `max_findings=0`.

Coverage describes tested or retained work. An overview shows each analytical
section's coverage independently, also exposed as `section_coverage` in the full
`visualization_data` projection. Reaching a work budget does not establish absence
of a pattern. An unrequested section is labeled separately. Columns skipped because
their cells have unsupported types are listed with their value type.

Overview findings appear in lead order: `f0` is the most promising lead, and each
card shows the reason it was ranked. The text overview opens with the same leads.
Text output displays Unicode by default; pass `unicode_mode="safe"` for ASCII.

Representative examples show saved/total source-position counts. Positions are
zero-based offsets in the original ordered source, not index labels, and examples
are the first matches in source order. They are not random samples. Each finding
provides a Python inspection expression; use the identical ordered dataframe with
`result.inspect(df, finding_id, all_matches=True)` for all matching rows or add
`exceptions=True` for exceptions. Comparisons instead direct readers to the
original before/after results, because they have no single recoverable population.
A comparison's fraction delta is after minus before: 0.25 is 25 percentage points.

## Grain maps and matrices

The map supports feature selection, key-neighborhood focus, attribute collapse,
and varying-feature overlays. Selection is announced and provides a **View
selected evidence** link. Enter/Space selects a focused feature; Escape clears
selection. Reset view restores controls, matrix filters, and census branches.

The HTML matrix puts **features in rows and candidate keys in columns**. Feature
search and the candidate-key filter help navigate wide data. The matrix scrolls
inside a bounded panel with sticky column headings and feature labels. Long labels
are visually shortened but retain full accessible names and titles; placement
evidence includes the full names. Every cell can select its feature's evidence.
The static SVG matrix uses the same orientation, but keeps complete wrapped labels.

States remain distinct and carry an icon or line style and a text label in addition
to color, so the matrix survives a black-and-white print:

| State | Meaning |
| --- | --- |
| constant (blue, check) | Each evaluated group has one observed value |
| varying (orange, half circle) | At least one evaluated group has conflicting values |
| undefined (grey, dashed, "?") | No evaluated support for consistency |
| untested (white, dashed) | No saved test for this candidate/feature cell |
| `*` suffix | The target's evaluated population differs from the graph population |

Constancy on unique keys does not provide repeated support. Placement evidence
retains singleton/repeated group counts and the original evaluated denominators.

## Smaller viewports and accessibility

SVG figures offer Actual size, Fit width, and 50%, 75%, 150%, and 200% scales.
Actual size keeps labels readable; wide figures scroll within their own panels.
Fit width is useful for orientation, but can make large graphs' text small.
The native HTML grain matrix uses browser text/layout scaling, so SVG scaling
and map-only controls are disabled while viewing it.

Controls wrap with the viewport. Key/value evidence stacks vertically below
600 CSS pixels. Census and other multi-column evidence tables retain readable
column widths and scroll locally. This avoids shrinking all the report text to
fit an arbitrarily wide dataset. Figure panels can receive keyboard focus.

Reports use native buttons, labels, disclosures, table headings, visible focus,
a skip link, and live filter/selection status. With JavaScript disabled, saved
records remain readable through native disclosures; enhanced controls are hidden.
This is browser-tested keyboard support, not a claim of a full assistive-technology
audit.

## Colour and type

Reports follow the viewer's light or dark colour scheme. Figures and the HTML grain
matrix stay on white in both, so they read the same when shared or printed. Reports
use the system's sans-serif and monospace fonts, never web fonts, so they render
without a network; column and key names are set in monospace.

Colour in figures encodes an analysis result and nothing else:

| Colour | Meaning |
| --- | --- |
| Blue | Determined: constant cells, and n:1 pairs (the row determines the column) |
| Orange | Varies: varying cells, and 1:n pairs (the column determines the row) |
| Dark fill, white text | 1:1 pairs, determined both ways |
| Dashed outline | Absent or out of scope: undefined, untested, n:m, missing share, unplaced columns, omitted branches |
| Yellow → green → teal → petrol | Amounts in four ranges (association, joint counts); teal alone for share bars |

Interface controls and focus rings are dark grey, so blue never means an action.
Legends sit at fixed places below grids and after bars, since generated figures
cannot predict where a callout would land.

## Shading ranges

The association view of `pairs` and the joint-count heatmap shade each cell by one
of four ranges, and every cell also prints its value. The cut-offs adapt to each
figure by default and the legend prints the ones used:

| Rule | Cut-offs | Default for |
| --- | --- | --- |
| `"equal"` | Four equal-width ranges between the smallest and largest value | association |
| `"log"` | Four ranges evenly spaced on a log scale up to the largest value (from 1 for counts), rounded | counts |
| `"quantile"` | About a quarter of the cells in each range | – |
| three numbers | Fixed, increasing cut-offs, the same in every report | – |

Pass rules through `shading` on `render_svg`, `render_html` or `visualization_data`,
typed as `fieldwork.typing.Shading`:

```python
fw.render_html(result, shading={"association": "quantile"})
fw.render_svg(result, view="association", shading={"association": (0.1, 0.3, 0.5)})
fw.render_svg(counts, shading={"counts": (10, 50, 150)})
```

Fixed cut-offs keep several reports on one scale. Across the contexts of a pairs
result the cut-offs are computed once, so every context uses the same ranges.
Full-detail pairs and joint-count projections record the cut-offs and rule under
`shading`; topology projections carry no quantities and no cut-offs. A value equal
to a cut-off falls in the higher range.

`detail="topology"` filters data before building the page. Measurements, coverage,
source positions, and inspection commands are absent, including in hidden markup.
Structural feature/value labels remain and can identify records. Filters do not
change the underlying analytical result or saved serialization.
