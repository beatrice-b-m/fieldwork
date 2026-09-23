# Inline API documentation and editor support

The supported Python surface is `fieldwork.__all__`, public members of those
objects, the `Path` returned by `Result.best`/`Result.path`, and the types exported from
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

Public dataframe operations declare the runtime controls once, as
`**runtime: Unpack[Runtime]` (`fieldwork.typing.Runtime`: `progress`, `cancel`,
`timeout`), so type checkers and editors complete them. The private runtime
decorator consumes the controls, rejects any other unexpected keyword with
`TypeError`, preserves parameter and return types, and publishes an expanded
`__signature__` so `help()` and IPython list the three controls as keyword-only
parameters. Docstrings refer to `Runtime` instead of repeating the controls.

`census` and `Path.census` spell out supported options. The latter preserves the
recommendation's source context; `scope`, `missing`, `table_id`, and replacement
`dimensions` are not arguments. Unsupported context keywords now raise ordinary
`TypeError` at the callable boundary, rather than the previous `ValueError` from
validation inside the method. `grain` exposes only public options; encodings and
cache objects are passed through its private implementation.

`explore(df)` (the overview) and `profile(df, dimensions)` (explicit composition)
are separate functions with explicit keyword arguments; both return a `Result`.

Reusable dictionaries can be annotated with `SectionOptions` (the overview's
per-section `options`) or the individual operation option types from
`fieldwork.typing`. These are ordinary dictionaries, not runtime validation models.
Optional keys inherit the operation defaults. Option field docstrings are present
in source for editor hovers. Flexible serialized evidence remains a versioned
mapping; its field meanings are documented on result classes and producer methods.

## Validation

Run the checks documented in [development](development.md). Specifically:

- `pytest` runs every docstring example in `src/fieldwork` (`--doctest-modules`
  is configured in `pyproject.toml`), including examples on classes and methods.
- `tests/test_public_api.py` is a smoke test: every export, its public members
  and the returned `Path` have a docstring, annotated parameters and a return
  annotation, and no public signature exposes an underscore-prefixed argument.
- `uv run pyright --warnings` checks strict consumer examples in `tests/typing`,
  including precise result types, overload selection, option dictionaries and
  expected rejection of invalid calls. Unused diagnostic suppressions fail, so
  accidentally accepting an invalid option is detected. Pyright checks the type
  engine also used by Pylance; automated tests do not drive an editor UI.
- CI reruns the smoke test and the docstring examples against an isolated
  installed wheel and executes the investigation example. The wheel includes
  source docstrings and `py.typed`; no separate stub files can drift from them.

Docstring section layout, parameter-list parity and editor completion are not
tested; review them when changing a public interface. Analytical behavior is
covered by the foundation/discovery suites, whose oracles recompute results
independently. Docstring coverage does not establish scientific accuracy: review
claims, denominators, exceptions, and defaults against implementation and
contract tests as part of every public change.
