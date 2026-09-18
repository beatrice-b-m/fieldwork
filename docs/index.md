# Fieldwork developer map

Start here when changing or reviewing the package. User-facing workflows and API
reference belong in [fieldwork-docs](https://github.com/beatrice-b-m/fieldwork-docs).

1. [Architecture](architecture.md) — boundaries, data flow, where to make changes.
2. [Evidence contracts](contracts.md) — populations, types, saved inspection, exports.
3. [Discovery algorithms](algorithms.md) — measurements, search limits, ranking.
4. [Development and validation](development.md) — uv, parity tests, examples, profiling.
5. [Release and documentation](releases.md) — generated assets, two-repository sync, publication.

See the [complete single-table journey](investigation.md) for scope handoffs,
selectable signatures, entity units and connected feature evidence.

For a small executable tour, read `examples/investigation.py`. For exact interface
signatures, read `src/fieldwork/__init__.py` and the exported functions' modules.
Tests under `tests/foundation/` preserve the extracted explorer contracts;
`tests/discovery/` exercises new investigation behavior.

- [Large dataframes, progress, cancellation, and work budgets](performance.md)
- [Performance implementation measurements](performance-results.md)
- [Dependency support qualification (v0.1.1)](evaluation/dependency-support.md)
- [Exploring saved outputs](output-ux.md)
- [Output UX review and browser qualification](evaluation/output-ux.md)
- [v0.1.1 release record](releases/v0.1.1.md)
- [v0.1.2 release record](releases/v0.1.2.md)
