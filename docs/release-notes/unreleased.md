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
- Overview findings are ranked as investigation leads and numbered in that order
  (`f0` is the top lead). Each carries `lead.score` and `lead.reason`; see
  [lead ranking](../algorithms.md#overview-lead-ranking). **Overview finding IDs
  change**; section results keep their own IDs.
- Trivially true dependencies (constant targets, unique determinants) are no
  longer edges in the overview feature network.
- The text overview opens with up to eight ranked **Leads** (finding ID,
  statement and reason). Candidate grains take one line each, and single-column
  candidates with the same partition are merged and listed as `equivalent`
  (also in the SVG, HTML and `visualization_data` overview). Availability
  signatures are described by the columns they are missing ("Missing: notes",
  "All populated") instead of long lists of present columns that truncated to
  identical lines. HTML finding cards show each lead reason.

## Speed and saved evidence

- Source fingerprinting uses vectorized pandas hashing and is roughly 100× faster
  (0.04 s instead of 4.8 s for 500,000 rows × 14 columns). This was most of the
  runtime of every call that verifies a source, such as `missingness`, `inspect`,
  `select` and scoped `census`. **Breaking:** fingerprints saved by 0.1.x no longer
  match, so older saved results and scopes still load and render but cannot
  inspect or select against a source. Column dtype is now part of source identity.
