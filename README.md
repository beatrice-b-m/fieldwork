# Fieldwork

Fieldwork is a Python toolkit for exploring the structure of unfamiliar data. It
supports research and data curation by helping researchers identify patterns in
features, values, grain, relationships, and missingness, then inspect the evidence
and pursue explanations.

The project centers on understanding what records represent, which features belong
together, how values and availability vary across entities and contexts, and which
paths through a dataset make its structure easier to understand. Its scope spans
individual tables and relationships among tables.

Fieldwork's design emphasizes composable dataframe operations, readable summaries,
structured results, and useful visualizations. Findings should connect directly to
their supporting observations and exceptions, enabling iterative investigation and
reproducible analysis. The intended Python package and import name is `fieldwork`.

The toolkit trusts researchers to interpret findings in the context of their work.
Outputs should describe patterns, populations, support, and exceptions directly;
methodological explanations belong in the documentation.

Durable codebase documentation lives in [`docs/`](docs/). Limited-lifespan
implementation specifications and development notes live in [`temp-docs/`](temp-docs/).
