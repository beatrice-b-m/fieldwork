# Fieldwork (unreleased)

Changes merged since 0.3.1. At release, complete these notes and rename the file
`v<version>.md`.

## Added

- `render_svg`, `render_html` and `visualization_data` accept `shading`
  (`fieldwork.typing.Shading`) to choose how association and joint-count cells
  are split into four ranges: `"equal"`, `"log"`, `"quantile"` or three fixed
  cut-offs. Full-detail pairs and joint-count projections record the cut-offs
  used under `shading` ([shading ranges](../output-ux.md#shading-ranges)).

## Changed

- Figures and reports are restyled. Figures draw on white with dark outlines,
  system fonts (identifiers in monospace), orthogonal grain-map connectors and
  fixed-place legends. Colour now encodes results only: blue for determined
  (constant, n:1), orange for varying (varying, 1:n), a dark fill for 1:1,
  dashed outlines for absent or out-of-scope cells, and a four-range scale for
  amounts that adapts to each figure. HTML reports follow the viewer's light or
  dark scheme, keep figures on white, and remain offline
  ([colour and type](../output-ux.md#colour-and-type)).

- Column availability findings in `missingness` state whether a column is
  populated in some or no units (`X: populated in some rows`, `X: populated in
  no rows`) and record it as `structure["presence"]` (`"some"` or `"none"`),
  as context availability findings do. Availability families record the same
  state (`Same availability (populated in no rows): a, b`). Topology exports
  therefore distinguish always-missing columns from partially populated ones.
  Complete columns are still not findings. Code that matched the previous
  statements, `X: populated values` and `Same availability: ...`, must be
  updated; finding IDs are unchanged (#22).
- Overview leads no longer rank an approximate dependency as a top "near-rule
  with exceptions" merely because one repeated determinant group exists. Its
  repeated rows must cover at least a quarter of evaluated rows and agree with
  the modal rule in at least 90% of them; other near-rules score 0.3 with the
  reason "near-rule with sparse repeated support". Lead order and `lead`
  annotations can therefore change; section findings and their IDs do not
  ([overview lead ranking](../algorithms.md#overview-lead-ranking)) (#27).
