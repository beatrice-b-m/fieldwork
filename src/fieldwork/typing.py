"""Public type vocabulary for editor completion and reusable configurations.

These TypedDict classes describe ordinary dictionaries; they do not validate or
fill defaults at runtime. All keys are optional. Function defaults apply when a
key is omitted. Import these types from ``fieldwork.typing`` when annotating a
configuration assembled separately from its call. Result payloads are
versioned JSON-compatible mappings; see ``Result`` for their contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

from .progress import CancellationToken, Progress

if TYPE_CHECKING:
    from ._explore.result import KeySpec
    from .evidence import Scope

__all__ = [
    "CensusOptions",
    "DependencyOptions",
    "DiscoveryOptions",
    "ExplicitDiscoveryOptions",
    "FoundationOptions",
    "MissingnessOptions",
    "OverviewOptions",
    "PathOptions",
    "PatternOptions",
    "Runtime",
    "SchemaRole",
    "Section",
    "SectionOptions",
]

SchemaRole: TypeAlias = Literal["id", "categorical", "continuous", "unknown"]
"""Advisory column role; supplying a role never converts source values."""
Section: TypeAlias = Literal["missingness", "dependencies", "paths", "value_patterns"]
"""An independently selectable automatic overview section."""


class Runtime(TypedDict, total=False):
    """Runtime controls accepted by every analysis and source-bound method.

    They are never saved in results or recipes. Nested analyses share the
    outermost call's controls.
    """

    progress: Progress
    """None or False (default) is silent; True uses ProgressDisplay; a callable
    receives each ProgressEvent synchronously. Callback errors propagate unchanged."""
    cancel: CancellationToken | None
    """Token checked between work items; cancelling raises AnalysisCancelled and
    returns no partial result. Default None."""
    timeout: float | None
    """Finite nonnegative seconds from call start; expiry raises AnalysisCancelled at
    the next checkpoint, after the current pandas/NumPy work item. Default None."""


class CensusOptions(TypedDict, total=False):
    """Output and cohort options accepted by ``census`` and ``Path.census``.

    Omitted keys use the census defaults. Preselection explicitly changes the
    evaluated cohort; postselection and display limits preserve its population.
    Source context and runtime controls are deliberately separate.
    """

    top_n: int | None
    """Positive number of leading levels; None (default) keeps all eligible levels."""
    top_n_mode: Literal["pre", "post"]
    """Default 'post' limits output; 'pre' selects rows before counting."""
    top_n_per_parent: bool
    """Default False selects globally; True selects within each parent in pre or post mode."""
    min_retained_fraction: float
    """Warn below this retained fraction after preselection; default 0.01, range [0, 1]."""
    max_depth: int | None
    """Positive prefix depth, or None (default) for all dimensions."""
    max_levels: int | None
    """Maximum displayed children per node; default 100, zero hides all, None is unbounded."""
    max_nodes: int | None
    """Maximum displayed nodes; default 10000, zero hides all, None is unbounded."""
    min_count: int
    """Minimum displayed node count; default 1, must be nonnegative."""
    dropna: bool
    """Default False includes missing levels; True excludes rows missing active dimensions."""
    schema: dict[str, SchemaRole] | None
    """Optional advisory roles by column; default None."""


class OverviewOptions(TypedDict, total=False):
    """Common context overrides for automatic ``explore``.

    Explicit keys replace matching ``discovery`` entries. They apply across
    requested sections; section-specific feature choices can further override
    ``features``. A scope reduces analytical rows, but source identity still
    covers the entire original frame.
    """

    scope: Scope | None
    """Source-bound selection, or None (default) for all rows."""
    missing: Mapping[str, Iterable[Any]] | None
    """Additional missing sentinels per string column; default None, native missing always counts."""
    table_id: str
    """Nonempty source label; default 'table'."""
    features: Iterable[str] | None
    """Selected unique string column names, or None (default) for all columns."""


class ExplicitDiscoveryOptions(TypedDict, total=False):
    """Common context allowed in ``discovery`` with explicit dimensions.

    Unlike automatic exploration, duplicate keys supplied directly to ``explore``
    raise ValueError. Foundation labels may be strings, integers, or tuples.
    """

    scope: Scope | None
    """Source-bound selection, or None for all rows."""
    missing: Mapping[str, Iterable[Any]] | None
    """Additional missing sentinels by typed column label; default None."""
    table_id: str
    """Nonempty source label; default 'table'."""
    features: Iterable[str] | None
    """Columns for independent levels; default None uses dimensions."""


class PathOptions(TypedDict, total=False):
    """Search options for ``suggest_paths`` or an overview's paths section.

    Budgets bound search work and evidence, never sample input rows. See
    ``suggest_paths`` for objective and constraint semantics.
    """

    features: Iterable[str] | None
    """Candidate string column names; default None selects all."""
    objective: Literal["structure", "availability", "compact", "target", "context"]
    """Ranking objective; default 'structure'."""
    start_with: Iterable[str] | None
    """Required ordered initial dimensions; default None."""
    before: Iterable[tuple[str, str]] | None
    """Acyclic precedence constraints; both columns must appear; default None."""
    exclude: Iterable[str] | None
    """Columns excluded from paths; default None."""
    target: str | None
    """Feature to explain; required for objective='target', otherwise optional."""
    max_dimensions: int
    """Positive maximum path length; default 4."""
    max_candidates: int
    """Positive extension budget; default 200."""
    max_features: int
    """Positive candidate feature budget; default 20; required features must fit."""
    max_pairs: int
    """Nonnegative nesting-pair budget; default 200."""
    beam_width: int
    """Positive number of alternative feature sets retained per depth; default 12."""
    n_paths: int
    """Positive maximum number of recommendations; default 3."""
    display_budget: int
    """Positive preview node limit and ranking cost reference; default 40."""


class MissingnessOptions(TypedDict, total=False):
    """Availability options for the automatic overview's missingness section.

    Source context and runtime controls belong on ``explore``, not here.
    Defaults match ``missingness``.
    """

    features: Iterable[str] | None
    """Selected string columns, default None for all."""
    by: Iterable[str] | None
    """Joint context columns; missing context values form categories; default None."""
    entity: str | Iterable[str] | None
    """Single or composite entity key; incomplete keys are excluded from entity counts."""
    unit: Literal["rows", "entities"]
    """Default 'rows'; 'entities' weights each populated key equally and requires entity."""
    entity_presence: Literal["any", "all"]
    """Default 'any'; 'all' requires every row of an entity to be populated."""
    min_implication: float
    """Minimum conditional presence fraction in [0, 1]; default 0.9."""
    min_similarity: float
    """Minimum Jaccard presence fraction in [0, 1]; default 0.8."""
    max_pairs: int
    """Nonnegative pair budget; default 200."""
    max_signatures: int
    """Nonnegative saved signature limit; default 50."""
    max_contexts: int
    """Nonnegative context-group budget; default 32."""
    example_limit: int
    """Nonnegative representative source-row limit per finding side; default 5."""


class DependencyOptions(TypedDict, total=False):
    """Dependency options for the automatic overview's dependencies section.

    Overview defaults are maximum key size 1 and 20 candidates; standalone
    ``discover_dependencies`` defaults to size 2 and 100 candidates.
    """

    features: Iterable[str] | None
    """Candidate and target string columns; default None for all."""
    max_key_size: int
    """Positive maximum determinant size; overview default 1."""
    max_candidates: int
    """Nonnegative determinant budget; overview default 20."""
    min_accuracy: float
    """Minimum modal repair accuracy in [0, 1] for reporting; default 0.95."""
    by: Iterable[str] | None
    """Additional joint context analyses; default None."""
    max_contexts: int
    """Nonnegative context budget in addition to global analysis; default 32."""
    dropna: bool
    """Default True excludes rows missing determinant or target per test."""
    example_limit: int
    """Nonnegative saved source-row limit per finding side; default 5."""
    include_grain: bool
    """Build foundation grain views; default True."""
    max_grain_views: int | None
    """Nonnegative graph-view budget; default None permits all supported views."""
    max_dependency_tests: int | None
    """Nonnegative candidate/target/context test budget; default None permits all."""


class PatternOptions(TypedDict, total=False):
    """Populated-value options for the automatic overview's value_patterns section.

    The overview defaults to 20 pair tests; standalone ``value_patterns`` uses
    100. Source context and runtime controls belong on ``explore``.
    """

    features: Iterable[str] | None
    """Selected string columns; default None for all."""
    by: Iterable[str] | None
    """Joint context columns for observed value mappings; default None."""
    max_pairs: int
    """Nonnegative pair-test budget; overview default 20."""
    max_patterns: int
    """Nonnegative saved patterns per summary; default 10."""
    example_limit: int
    """Nonnegative representative source-row limit per finding side; default 5."""


class DiscoveryOptions(PathOptions, total=False):
    """Automatic overview configuration: paths plus shared context and availability.

    Search keys affect paths only. ``by`` also configures dependencies and value
    patterns. Entity settings affect missingness only. Use ``section_options``
    for independent analytical budgets. No runtime controls may be stored here.
    """

    scope: Scope | None
    """Source-bound selection; default None for all rows."""
    missing: Mapping[str, Iterable[Any]] | None
    """Additional missing sentinels by string column; default None."""
    table_id: str
    """Nonempty source label; default 'table'."""
    by: Iterable[str] | None
    """Context columns for availability, dependencies, and value patterns; default None."""
    entity: str | Iterable[str] | None
    """Availability entity keys; default None."""
    unit: Literal["rows", "entities"]
    """Availability counting unit; default 'rows'."""
    entity_presence: Literal["any", "all"]
    """Availability entity aggregation; default 'any'."""


class SectionOptions(TypedDict, total=False):
    """Per-section options overriding automatic overview defaults.

    Only requested sections may have entries. Source context and runtime controls
    cannot be overridden here. Each entry is an ordinary options dictionary.
    """

    missingness: MissingnessOptions
    """Availability options for the requested missingness section."""
    dependencies: DependencyOptions
    """Dependency options for the requested dependencies section."""
    paths: PathOptions
    """Path-search options for the requested paths section."""
    value_patterns: PatternOptions
    """Value-summary options for the requested value_patterns section."""


class FoundationOptions(CensusOptions, ExplicitDiscoveryOptions, total=False):
    """Forwarded options for ``explore`` with explicit dimensions.

    Includes census and source context options. Pair budgets apply independently
    from census display limits. Omitted keys use explicit exploration defaults.
    """

    candidate_keys: Iterable[str | KeySpec] | None
    """Explicit grain candidates; default None skips grain."""
    top_n_applies_to: Literal["census", "both"]
    """Default 'census'; 'both' also restricts pairs/grain and requires pre mode and top_n."""
    include_pairs: bool
    """Include pair relationships; default True."""
    include_absence: bool
    """Include bounded absent combinations; default False; requires include_pairs."""
    reference_domains: Mapping[str, Iterable[Any]] | None
    """Declared pair value domains, otherwise observed domains; default None."""
    pair_contexts: Iterable[Mapping[str, Any]] | None
    """Additional pair analyses restricted by exact context values; default None."""
    max_absence_cells: int | None
    """Nonnegative absent-cell output budget; default 1000, None is unbounded."""
    max_contexts: int | None
    """Total context budget including global; default 32, zero skips all, None is unbounded."""
    max_pairs: int | None
    """Nonnegative pair budget; default 15, None is unbounded."""
