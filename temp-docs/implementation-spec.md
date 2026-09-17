# Implementation specification — delivered behavior

Status: implemented single-table milestone, 2026-09-17.

The original proposal is preserved in Git history. Its delivered contracts now
live in [the developer map](../docs/index.md), [architecture](../docs/architecture.md),
[evidence contracts](../docs/contracts.md), and [algorithm definitions](../docs/algorithms.md).
User journeys are documented in the associated `fieldwork-docs` repository.

## Delivered

- A: extracted standalone pandas/NumPy foundation, tests, benchmark and grain example;
  typed evidence, population accounting, exact graphs and full/topology exports.
- B: native/sentinel availability, signatures, exact/similar families, implications,
  mutual exclusion, context and entity summaries, source-bound exception inspection.
- C: bounded single/composite candidate discovery, conditional and approximate
  dependencies, deterministic multi-objective path recommendations, implicit explore.
- D incremental depth: populated string/numeric/indexed-family patterns, context
  constancy, reusable scopes, JSON recipes and availability comparisons.
- uv packaging for Python 3.11–3.14, executable notebook, generated visual assets,
  versioned PyPI README, CI and release regeneration/publication workflow.

Decisions: MIT with extraction attribution; source positions plus canonical dataset
fingerprints; modal repair accuracy; explicit bounded beam search. The user authorized
initial documentation of the tested unreleased implementation before stable tracking.

## Deliberately later extensions from the proposal

Automatic related-table discovery and join profiling, branch-adaptive census orders,
cached sessions and optional discovery engines remain outside the first single-table
milestone. New feature references include table identity for future extension.

The initial release has not been published. Release preparation and two-repository
synchronization are described in [the release guide](../docs/releases.md).
