# fieldwork-docs synchronization for the next release

The next release (after 0.2.1) changes the public API and saved formats; see
`docs/release-notes/unreleased.md`. When it is published, synchronize
`fieldwork-docs` and then retire this file.

- Every item in `docs/release-notes/unreleased.md`.
- Walkthroughs and snippets that use `explore(df, dims)` (now `profile`),
  `discovery=` or `section_options=` (now explicit arguments and `options=`),
  flat budget keywords such as `max_candidates=` or `example_limit=` (now
  `limits={...}`), `include_pairs=` or pair options on `profile` (now
  `pairs=`), `to_dict(resolve_references=True)`, `ExplorerResult`,
  `InvestigationResult` or `PathResult` (now `Result`), `overview["findings"]`
  (now `overview.findings`), or `ProgressEvent.estimated_remaining_seconds`.
- Displayed outputs that show tagged `{"type": ..., "value": ...}` values,
  `level_dictionary`, `scope_metadata`, the `Fieldwork feature explorer v0.3`
  text header, or a phase ETA.
- The rewritten `docs/contracts.md` and the relaxed `docs/inline-api.md`.
- `public/generated/`, regenerated from `docs/assets/` (including
  `README.pypi.md`).
- Links to the retired `docs/releases/*.md`, `docs/performance-results/*.json`
  and `docs/evaluation/*.json` (still available at tag `v0.2.1`).
