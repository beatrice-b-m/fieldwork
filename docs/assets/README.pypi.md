# Fieldwork

Explore unfamiliar data through patterns, source-row evidence, and reproducible
investigations. Fieldwork is a Python toolkit for researchers working with pandas
dataframes: find availability families, examine dependencies and candidate grains,
then choose a useful census path through the data.

**Development version — not yet published.** Python 3.11–3.14; pandas and NumPy are
the only runtime dependencies.

![Availability in the worked example](https://raw.githubusercontent.com/beatrice-b-m/fieldwork/v0.1.0/docs/assets/availability.png)

```bash
uv add "fieldwork @ git+https://github.com/beatrice-b-m/fieldwork"
```

```python
import pandas as pd
import fieldwork as fw

df = pd.DataFrame({
    "site": ["North", "North", "South", "South"],
    "exam": [1, 1, 2, 2],
    "image": [10, 11, 20, 21],
    "report": ["ok", "ok", "ok", None],
})
overview = fw.explore(df)
availability = fw.missingness(df, entity="exam", min_implication=0.75)
paths = fw.suggest_paths(df, features=["site", "exam"])
tree = fw.census(df, paths.best.dimensions)
print(tree)
```

Use `result.to_frame()` for discovery tables and `result.inspect(df, finding_id,
exceptions=True)` for the saved example rows. Duplicate indexes are supported;
inspection checks the ordered source dataset. Save results with `to_dict()`, export
with `render_svg()` or `render_html()`, and reapply a `Recipe` to later deliveries.

![Observed census from the worked example](https://raw.githubusercontent.com/beatrice-b-m/fieldwork/v0.1.0/docs/assets/census.png)

The single-table workflow includes independent levels, contextual pair summaries,
joint counts and absence, exact grain graphs, bounded approximate dependency
discovery, five census-path objectives, availability signatures and entity summaries,
string and numeric patterns, scoped investigations, and delivery comparisons.
Topology-only exports retain structure while suppressing quantitative evidence.
Automatic related-table discovery is a later extension.

- [User documentation source](https://github.com/beatrice-b-m/fieldwork-docs)
- [Developer documentation](https://github.com/beatrice-b-m/fieldwork/blob/v0.1.0/docs/index.md): architecture, contracts, algorithms, releases
- [Executable investigation](https://github.com/beatrice-b-m/fieldwork/blob/v0.1.0/examples/investigation.py) and [notebook](https://github.com/beatrice-b-m/fieldwork/blob/v0.1.0/examples/investigation.ipynb)
- [MIT license](https://github.com/beatrice-b-m/fieldwork/blob/v0.1.0/LICENSE) and [extraction provenance](https://github.com/beatrice-b-m/fieldwork/blob/v0.1.0/NOTICE)

For development: `uv sync --locked`, `uv run pytest`, and `uv build`. Documentation
images are generated from executable examples using the public renderers. Run
`uv run python scripts/generate_assets.py` after behavior or styling changes;
CI checks concurrence and release builds regenerate the assets.
