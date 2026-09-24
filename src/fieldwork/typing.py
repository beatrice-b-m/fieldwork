"""Public type vocabulary for editor completion and reusable configurations.

These TypedDict classes describe ordinary dictionaries; they do not validate or
fill defaults at runtime. All keys are optional. Function defaults apply when a
key is omitted. Import these types from ``fieldwork.typing`` when annotating a
configuration assembled separately from its call. Result payloads are
versioned JSON-compatible mappings; see ``Result`` for their contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypeAlias, TypedDict

from .progress import CancellationToken, Progress

__all__ = [
    "CensusOptions",
    "DependencyLimits",
    "DependencyOptions",
    "MatchMode",
    "MissingnessLimits",
    "MissingnessOptions",
    "PairLimits",
    "PairOptions",
    "PathLimits",
    "PathOptions",
    "PatternLimits",
    "PatternOptions",
    "RelateLimits",
    "Runtime",
    "SchemaRole",
    "Section",
    "SectionOptions",
    "Side",
]

SchemaRole: TypeAlias = Literal["id", "categorical", "continuous", "unknown"]
"""Advisory column role; supplying a role never converts source values."""
Section: TypeAlias = Literal["missingness", "dependencies", "paths", "value_patterns"]
"""An independently selectable automatic overview section."""
MatchMode: TypeAlias = Literal["typed", "text"]
"""How ``relate`` matches values across tables; see docs/algorithms.md."""
Side: TypeAlias = Literal["left", "right"]
"""A table of a relation result, for its row inspection and selection."""


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
    safe_errors: bool | None
    """True raises any other failure as AnalysisError, whose message names the
    operation, phase and column but never echoes the original message, which can
    quote cell values. Default None (False) raises failures unchanged."""


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


class PathLimits(TypedDict, total=False):
    """Search budgets of ``suggest_paths``; they bound work, never sample rows."""

    max_candidates: int
    """Positive extension budget; default 200."""
    max_features: int
    """Positive candidate feature budget; default 20; required features must fit."""
    max_pairs: int
    """Nonnegative nesting-pair budget; default 200."""
    beam_width: int
    """Positive number of alternative feature sets retained per depth; default 12."""
    display_budget: int
    """Positive preview node limit and ranking cost reference; default 40."""


class MissingnessLimits(TypedDict, total=False):
    """Work and output budgets of ``missingness``."""

    max_pairs: int
    """Nonnegative pair budget; default 200."""
    max_signatures: int
    """Nonnegative saved signature limit; default 50."""
    max_contexts: int
    """Nonnegative context-group budget; default 32."""
    example_limit: int
    """Nonnegative representative source-row limit per finding side; default 5."""


class DependencyLimits(TypedDict, total=False):
    """Work and output budgets of ``discover_dependencies``.

    The overview's dependencies section defaults to 20 candidates.
    """

    max_candidates: int
    """Nonnegative determinant budget, in size then column order; default 100."""
    max_contexts: int
    """Nonnegative context budget in addition to global analysis; default 32."""
    max_dependency_tests: int | None
    """Nonnegative candidate/target/context test budget; default None permits all."""
    max_grain_views: int | None
    """Nonnegative graph-view budget; default None permits all supported views."""
    example_limit: int
    """Nonnegative saved source-row (and exception-group) limit per test; default 5."""


class PatternLimits(TypedDict, total=False):
    """Work and output budgets of ``value_patterns``.

    The overview's value_patterns section defaults to 20 pair tests.
    """

    max_pairs: int
    """Nonnegative pair-test budget, in column order; default 100."""
    max_patterns: int
    """Nonnegative saved formats, lengths and prefixes per string column; default 10."""
    example_limit: int
    """Nonnegative representative source-row limit per finding side; default 5."""


class RelateLimits(TypedDict, total=False):
    """Output budget of ``relate``."""

    example_limit: int
    """Nonnegative representative source-row limit per finding and side; default 5."""


class PairLimits(TypedDict, total=False):
    """Work budgets of ``pairs``; None leaves a budget unbounded."""

    max_pairs: int | None
    """Nonnegative number of dimension pairs; default 15."""
    max_contexts: int | None
    """Nonnegative number of pair contexts; default 32."""
    max_absence_cells: int | None
    """Nonnegative number of absent combinations listed per pair; default 1000."""


class PairOptions(TypedDict, total=False):
    """Options of a profile's pairs section; see ``pairs``."""

    include_absence: bool
    """List unobserved combinations of observed or reference levels; default False."""
    reference_domains: dict[str, Iterable[object]] | None
    """Expected levels per dimension for absence; default None."""
    pair_contexts: Iterable[dict[str, object]] | None
    """Conditions under which each pair is repeated; default None."""
    limits: PairLimits
    """Work budgets; see PairLimits."""


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
    n_paths: int
    """Positive maximum number of recommendations; default 3."""
    limits: PathLimits
    """Search budgets; see PathLimits."""


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
    limits: MissingnessLimits
    """Work and output budgets; see MissingnessLimits."""


class DependencyOptions(TypedDict, total=False):
    """Dependency options for the automatic overview's dependencies section.

    Overview defaults are maximum key size 1 and 20 candidates; standalone
    ``discover_dependencies`` defaults to size 2 and 100 candidates.
    """

    features: Iterable[str] | None
    """Candidate and target string columns; default None for all."""
    max_key_size: int
    """Positive maximum determinant size; overview default 1."""
    min_accuracy: float
    """Minimum modal repair accuracy in [0, 1] for reporting; default 0.95."""
    by: Iterable[str] | None
    """Additional joint context analyses; default None."""
    dropna: bool
    """Default True excludes rows missing determinant or target per test."""
    include_grain: bool
    """Build foundation grain views; default True."""
    limits: DependencyLimits
    """Work and output budgets; see DependencyLimits (overview: 20 candidates)."""


class PatternOptions(TypedDict, total=False):
    """Populated-value options for the automatic overview's value_patterns section.

    The overview defaults to 20 pair tests; standalone ``value_patterns`` uses
    100. Source context and runtime controls belong on ``explore``.
    """

    features: Iterable[str] | None
    """Selected string columns; default None for all."""
    by: Iterable[str] | None
    """Joint context columns for observed value mappings; default None."""
    limits: PatternLimits
    """Work and output budgets; see PatternLimits (overview: 20 pair tests)."""
    min_count: int
    """Minimum rows for a reported format, length or prefix; default 1."""


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
