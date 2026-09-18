# Dependency support: release documentation handoff

Status: prepared, awaiting a release containing the implementation. This document
is limited-lifespan coordination material; retire it after the documentation PR
lands. No release, stable-site edits, deployment, or PR publication is authorized
by the implementation phase.

## Proposed user narrative

An exact dependency can have little repeated evidence. For X=[1,1,2,2] and
Y=['a',None,'b',None], dropping missing values leaves two singleton determinant
groups. X→Y is exact on those two rows, but the target was observed on only two of
four determinant-eligible rows, and repeat-only consistency cannot be assessed.

Use `result.to_frame('dependencies')` to inspect every completed test, even if it
falls below `min_accuracy`. `target_coverage` measures observed target availability;
`repeat_coverage` measures the share of evaluated rows in repeated groups;
`repeat_modal_accuracy` measures consistency among those repeated rows. Undefined
fractions are None/JSON null. When `dropna=False`, missing values participate as a
category and observed coverage can still be below one.

Candidate `determines` still lists global exact targets. The new
`determines_with_repeated_support` list requires a repeated group in each target's
own evaluated population. Full summaries show both counts and completed/possible
global tests. Ranking within the existing roles uses repeated-supported exact
targets. Search order and budgets still affect the evidence: untested does not mean
false, and repeated support does not establish a meaningful entity or reliability.

Use the executable `examples/dependency_support.py` and generated
`docs/assets/dependency-support.{html,json,png,svg}`. The existing investigation
notebook has refreshed overview output; link to the released copy only after the
release is verified. Durable definitions and qualification are in
`docs/algorithms.md`, `docs/contracts.md`, and `docs/evaluation/dependency-support.md`.

## Synchronization checklist for the future documentation PR

1. Verify the then-current stable tag, package version, source revision and release
   assets contain this implementation. Read the current `fieldwork-docs/AGENTS.md`.
   The planning reference was channel `stable`, release `v0.1.0`, source
   `657555f44d29960a3fe5d11eb9276634cd73f5a8`; do not assume it is still current or
   publish unreleased behavior under that identity.
2. Update all release/source/channel metadata together in a release-matched PR.
   Preserve the documentation repository's release synchronization requirements.
3. Update `reference/discover-dependencies.md`, `reference/measurement-details.md`,
   `reference/results.md`, `guides/dependencies.md`, and overview/grain explanations.
   Explain Q/O/E/R, legacy unavailable fields, global test counters, and the
   intentional full-presentation ordering change. Keep graph populations distinct.
4. Include the executable sparse-target example and the singleton-conflict example
   (98 singletons plus one conflicting pair: overall 0.99, repeat-only 0.5).
   Check JSON null, missing-category behavior, saved rendering and below-threshold
   table inspection. Update executable example checks and generated assets from
   the verified release, and visually inspect them.
5. Run `npm run format` and `npm run validate` in `fieldwork-docs`; review source
   metadata and links together with the narrative. Publish changes through its PR
   workflow. This repository's implementation completion does not perform that PR.
