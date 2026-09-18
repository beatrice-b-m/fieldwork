# Repository guidelines

- Track repository changes in appropriately scoped commits. Each commit should contain one coherent change and avoid unrelated modifications.
- Keep documentation synchronized with the functional capabilities, interfaces, configuration, and usage of the codebase.
- Store durable codebase documentation in `docs/` and limited-lifespan development documentation in `temp-docs/`. Update or retire temporary documents as their purpose is fulfilled.

## Inline API documentation

- Update docstrings and type annotations alongside every public API change.
  This includes exports, public members, and returned objects such as `Path`.
- Use NumPy-style docstrings covering purpose, parameters and defaults,
  returns, relevant exceptions, and examples where useful. Document the
  exported entry point, not only its internal implementation.
- Explain applicable behavioral contracts: populations and denominators,
  missing values, source identity, selection, search versus display limits,
  mutation, and serialization. Keep essential usage guidance in docstrings
  and consistent with docs/.
- Provide precise public types and statically discoverable signatures,
  including decorator-added controls and forwarded options. Keep private
  arguments out of public interfaces.
- Verify changed examples and run applicable documentation and typing checks.
  For interface changes, check editor signatures, completions, and result
  navigation. Bring touched public interfaces into compliance.
