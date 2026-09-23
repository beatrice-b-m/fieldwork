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
- `grain(schema=...)`, which had no effect, is removed.
- `engine_metadata=` is removed from `levels`, `census`, `grain`, `explore` and
  `Path.census`, together with the `engine` payload field it added.
- Results no longer carry a `stability` attribute or export a `stability` field.
- Pair records no longer carry the constant `claim` and
  `higher_order_constraints_ruled_out` fields.

## Changed

- Runtime controls are declared once, as `fieldwork.typing.Runtime`
  (`progress`, `cancel`, `timeout`), and appear in signatures as
  `**runtime: Unpack[Runtime]`. Calls are unchanged; unknown keywords still
  raise `TypeError`, and `inspect.signature` still lists the three controls.

## Values and column labels

- Exported values are plain JSON instead of tagged `{"type", "value"}` records:
  `null` for missing, numbers as numbers (large integers exactly), infinities as
  `"inf"`/`"-inf"`, and dates, datetimes and timedeltas as text that pandas
  parses back. Aware datetimes are exported as UTC instants. Timedeltas use
  pandas' own text (`-1 days +23:59:55`), fixing labels whose leading sign
  pandas could not parse. Every saved result from 0.2.x is a different format.
- Analyses count `pandas.factorize` codes instead of per-value Python
  identities. On a 500,000-row frame with three unique float columns, `levels`
  on a float column falls from 2.4 s to 0.14 s.
- Identity rules are unchanged: native missing spellings are one value;
  booleans, numbers and strings never match; integers and floats stay distinct
  in object columns; sentinels still match numbers numerically.
- Columns are identified by `str(label)` in every analysis. Discovery analyses
  now accept non-string labels (such as the integers of a headerless CSV) and
  report them by their string form; foundation analyses report integer and tuple
  labels as strings instead of tagged records. `fieldwork.typing.ColumnLabel` is
  removed; column arguments are typed `str`.
- Census nodes carry their own `column` and `value`; `level_dictionary`, node
  `feature_id`/`level_id`, and retained-set `level_codes` are replaced by
  inline values (`retained_sets[].values`, `path`). Level records drop
  `level_id`, and warnings drop `feature_id`.
- `to_dict(resolve_references=True)` is removed: exports are self-describing.
- Text, SVG and HTML displays quote a string only when it could be read as a
  number, boolean or missing value (`site=North`, `code='1'`). Context
  statements no longer append the value type.
- Saved sentinels are stored and reapplied as exported values; a timestamp
  sentinel saved as ISO text still matches the timestamps it named.

## One result model

- Every analysis returns `fieldwork.Result`, exported as schema `2.0`.
  `ExplorerResult`, `InvestigationResult` and `PathResult` are removed; saved
  schema 0.3 and 1.0 results no longer load. `inspect`, `select`, `recompute`,
  `to_frame` and `relationships` are available on every result; `best` and
  `path(index)` work on paths results and on overviews (through their paths
  section). The new `section(name)` returns one part of an overview as a
  `Result`, replacing `Result.from_dict(overview["sections"][name])`.
- `KeySpec` now lives with `grain`; import it as `fieldwork.KeySpec` as before.
