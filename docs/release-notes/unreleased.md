# Unreleased

Changes since 0.3.0.

## Fixes

- `render_svg` and `render_html` of a schema proposal (`infer_schema`) raised
  `KeyError: 'findings'` in 0.3.0, although 0.3.0 introduced SVG and HTML cards
  for proposals. Both now render the proposal cards, and the full HTML report
  states the analyzed population like other reports. The rendering contracts
  now cover schema proposals in every medium.
- `render_plaintext(..., max_nodes=n)` on a census again keeps the first n
  nodes breadth first, as in 0.2.x and like the census's own node budget, so
  every top-level group is listed before any subdivision. 0.3.0 kept them in
  reading order, which could hide whole top-level groups behind the first
  group's subdivisions; this change was not intended and was missing from the
  0.3.0 notes.
