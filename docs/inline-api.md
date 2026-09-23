# Inline API documentation and editor support

The supported Python surface is `fieldwork.__all__`, public members of those
objects, the `Path` returned by `Result.best`/`Result.path`, and the types exported from
`fieldwork.typing`. Underscore modules and other unexported implementation helpers
are private. Source docstrings are the reference shown by editors and `help()`;
web documentation supplements them.

## Writing public documentation

Docstrings are concise NumPy style: a purpose statement, `Parameters` with
defaults and valid choices, `Returns`, and one small example where it helps.
Add `Raises` or `Notes` only for what a caller must handle or rely on. Exported
wrappers, properties, returned objects and dataclass fields are documented;
constructors use `Parameters`, fields adjacent docstrings.

Population and denominator semantics, missing conventions, source identity,
selection, budgets and serialization are documented once, in
[evidence contracts](contracts.md), [algorithms](algorithms.md) and
[runtime controls](performance.md). Docstrings name the population a count uses
when it is not obvious and refer to those pages rather than restating them.

Examples use small deterministic dataframes with explicit imports and execute
under pytest, including examples on classes and methods. Changing a default or
option also updates its configuration type, examples and docs.

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
per-section `options`), the individual operation option types, or the `limits`
types (`PathLimits`, `MissingnessLimits`, `DependencyLimits`, `PatternLimits`,
`PairLimits`) from `fieldwork.typing`. These are ordinary dictionaries, not
runtime validation models. Optional keys inherit the operation defaults. Option field docstrings are present
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

Docstring section layout and parameter-list parity are not tested; review them
when changing a public interface. Analytical behavior is
covered by the foundation/discovery suites, whose oracles recompute results
independently. Docstring coverage does not establish scientific accuracy: review
claims, denominators, exceptions, and defaults against implementation and
contract tests as part of every public change.
