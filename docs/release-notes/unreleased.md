# Unreleased

Changes since 0.1.2. Fieldwork is an alpha: saved-result formats may change
between releases without migration.

## Fixes

- Discovery analyses accept groupby-style MultiIndexes with float or datetime
  levels. Previously every discovery analysis raised `TypeError`.
- Text output displays Unicode by default, including Fieldwork's own `·` and `→`
  separators. Width uses `wcwidth` when the optional `unicode` extra is installed,
  otherwise East Asian width rules. `unicode_mode="safe"` still produces ASCII.
  Control and bidirectional override characters remain escaped.
- Timedelta values sort numerically and display as durations
  (`0 days 00:01:40`) instead of raw nanoseconds.
- Columns with unsupported cell types (such as lists, dicts or `Decimal`) no
  longer abort an automatic analysis. When `features` is omitted they are
  skipped and listed in the new `skipped_features` field, and in the text, SVG
  and HTML outputs. Explicitly requested columns still raise `TypeError`, now
  naming the column.
- A progress count that exceeds its phase estimate is clamped. Previously it
  raised `ValueError` and aborted the analysis.

## Overview usefulness

- `missingness` no longer emits vacuous findings: no per-feature availability
  finding for always-present columns, no family of always-present columns, and
  no presence implication whose target is always present or whose two features
  have identical availability. The `availability` table still lists every
  feature. On a complete 43-column table this removes about 440 findings and
  lets the feature network separate into meaningful components.
- Similar-presence findings are omitted when either feature is always present.
