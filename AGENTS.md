# Repository guidelines

- Track repository changes in appropriately scoped commits. Each commit should contain one coherent change and avoid unrelated modifications.
- Keep documentation synchronized with the functional capabilities, interfaces, configuration, and usage of the codebase.
- Store durable codebase documentation in `docs/` and limited-lifespan development documentation in `temp-docs/`. Update or retire temporary documents as their purpose is fulfilled.

## Inline API documentation

- Give every export, its public members and returned objects such as `Path` a
  concise NumPy-style docstring: purpose, parameters with defaults, returns,
  and one example where it helps. Add `Raises` or `Notes` only when a caller
  needs them.
- Document population and denominator semantics, missing values, source
  identity and selection once in `docs/`; docstrings refer to those pages
  instead of restating them.
- Keep public signatures precise and statically discoverable: annotated
  parameters and returns, TypedDicts for option mappings, no private
  arguments.
- Update docstrings, types and `docs/` with every public API change, and run
  the checks in `docs/development.md` (doctests, the public-API smoke test and
  pyright on `tests/typing`).
