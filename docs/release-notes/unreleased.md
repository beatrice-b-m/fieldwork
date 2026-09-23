# Unreleased

Changes since 0.3.0.

## Fixes

- `render_svg` and `render_html` of a schema proposal (`infer_schema`) raised
  `KeyError: 'findings'` in 0.3.0, although 0.3.0 introduced SVG and HTML cards
  for proposals. Both now render the proposal cards, and the full HTML report
  states the analyzed population like other reports. The rendering contracts
  now cover schema proposals in every medium.
