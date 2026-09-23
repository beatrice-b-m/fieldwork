# Unreleased

Changes since 0.2.0 that will be described in the next release notes.

## Fixes

- `value_patterns` string summaries (`formats`, `lengths`, `prefixes`) are lists
  of `[value, count]` pairs in memory, as they already were once saved.
  Previously they were tuples until a JSON round trip, so a live result rendered
  its HTML differently from its own saved export and `to_dict()` did not equal
  the saved payload.

