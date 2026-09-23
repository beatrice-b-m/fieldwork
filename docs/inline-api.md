# Inline API documentation and editor support

The supported Python surface is `fieldwork.__all__`, public members of those
objects, the `Path` returned by `PathResult`, and the types exported from
`fieldwork.typing`. Underscore modules and other unexported implementation helpers
are private. Source docstrings are the reference shown by editors and `help()`;
web documentation supplements them.

## Writing public documentation

Use NumPy-style sections: a concise purpose statement, `Parameters`, `Returns`,
relevant `Raises`, `Notes` for behavioral contracts, and small `Examples` where
useful. Document exported wrappers, properties, returned objects, and dataclass
fields. Constructors use `Parameters`; fields use `Attributes` or adjacent field
docstrings. Explain every supported argument, defaults, valid choices, units,
zero/None behavior, and interactions. For forwarding implementations, enumerate
supported keywords in `Other Parameters` and keep them aligned with overloads. Avoid duplicating type detail unnecessarily
in prose, but keep the text useful without visiting another page.

Specify the population behind counts and fractions, missing conventions, source
identity requirements, and mutation/serialization behavior. Distinguish bounded
search, truncated displays, missing exclusions, and explicit cohort selection.
A budget boundary is not a negative finding. Entity findings can select source
rows where the feature itself is absent. Exports retain all measurements;
topology projections are not anonymization. Keep these descriptions
consistent with [evidence contracts](contracts.md) and
[runtime controls](performance.md).

Examples should use small deterministic dataframes with explicit imports and
useful assertions. They execute under pytest, including examples on classes and
methods. Updating a default or option must also update the corresponding overload,
configuration type, examples, and prose.

## Static interfaces

Public dataframe operations declare keyword-only `progress`, `cancel`, and
`timeout` in their source signatures. The private runtime decorator consumes these
controls and preserves the callable's parameter and return types. It no longer
synthesizes signatures through runtime `__signature__` mutation.

`census` and `Path.census` spell out supported options. The latter preserves the
recommendation's source context; `scope`, `missing`, `table_id`, and replacement
`dimensions` are not arguments. Unsupported context keywords now raise ordinary
`TypeError` at the callable boundary, rather than the previous `ValueError` from
validation inside the method. `grain` exposes only public options; encodings and
cache objects are passed through its private implementation.

`explore` has two statically declared overloads. Omitted/None dimensions select
automatic discovery and an `InvestigationResult`; explicit dimensions select
foundation composition and an `ExplorerResult`. The runtime implementation keeps
its forwarding dictionary so that existing duplicate-setting and override rules
remain intact. Both overloads spell out keyword arguments because not every editor
expands `Unpack[TypedDict]` into completion suggestions. Editors may display both
overloads before enough context is available to select one.

Reusable dictionaries can be annotated with `DiscoveryOptions`, `SectionOptions`,
`OverviewOptions`, `FoundationOptions`, or individual operation option types from
`fieldwork.typing`. These are ordinary dictionaries, not runtime validation models.
Optional keys inherit the operation defaults. Option field docstrings are present
in source for editor hovers. Flexible serialized evidence remains a versioned
mapping; its field meanings are documented on result classes and producer methods.

## Validation

Run the checks documented in [development](development.md). Specifically:

- `tests/test_inline_docs.py` checks direct public documentation, correspondence
  between method/function parameters and docstrings, type annotation presence,
  static runtime controls, option-field descriptions, and executable examples.
- `tests/test_editor_api.py` exercises Jedi hover, signature help, keyword
  completion, and navigation through the actual imported package. It checks that
  private and unsupported census arguments do not appear in completion.
- `uv run pyright --warnings` checks strict consumer examples in `tests/typing`,
  including precise result types and expected rejection of invalid calls. Unused
  diagnostic suppressions fail, so accidentally accepting an invalid option is
  detected. This targets the public consumption contract, not strict typing of
  every numerical implementation detail. Pyright checks the type engine also used
  by Pylance; automated tests do not drive the VS Code UI.
- CI reruns documentation/editor checks against an isolated installed wheel and
  executes the investigation example. The wheel includes source docstrings and
  `py.typed`; no separate stub files can override or drift from those sources.

Analytical behavior remains covered by the foundation/discovery suites and the
revision parity corpus. Docstring coverage does not establish scientific accuracy:
review claims, denominators, exceptions, and defaults against implementation and
contract tests as part of every public change.
