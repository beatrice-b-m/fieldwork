# Fieldwork (unreleased)

Changes merged since 0.3.1. At release, complete these notes and rename the file
`v<version>.md`.

## Changed

- Column availability findings in `missingness` state whether a column is
  populated in some or no units (`X: populated in some rows`, `X: populated in
  no rows`) and record it as `structure["presence"]` (`"some"` or `"none"`),
  as context availability findings do. Availability families record the same
  state (`Same availability (populated in no rows): a, b`). Topology exports
  therefore distinguish always-missing columns from partially populated ones.
  Complete columns are still not findings. Code that matched the previous
  statements, `X: populated values` and `Same availability: ...`, must be
  updated; finding IDs are unchanged (#22).
