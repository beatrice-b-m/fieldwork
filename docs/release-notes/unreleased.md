# Unreleased

Changes since 0.2.1. Fieldwork is an alpha: this release simplifies the
architecture and deliberately breaks saved-result formats and parts of the
public API without migration shims.

## Removed

- Saved dependency results from 0.1.x (discovery schema 1.0 without repeated
  support fields) are no longer adapted when rendered. Candidate ranking and
  dependency explanations require the fields current analyses always write.
- `compare` no longer supplies a default row-based analysis unit for missingness
  results saved without `analysis_unit`.
