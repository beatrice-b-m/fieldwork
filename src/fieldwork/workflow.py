"""Composition and portable recipes delegate to public analytical operations."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Unpack, overload

import pandas as pd

from ._explore.orchestration import explore as explicit_explore
from ._explore.result import ExplorerResult, KeySpec
from ._runtime import operation, phase
from .availability import missingness
from .discovery import discover_dependencies
from .evidence import InvestigationResult, Scope, finding, foundation_context, prepare, result
from .families import feature_network
from .leads import rank
from .navigation import suggest_paths
from .patterns import value_patterns
from .progress import CancellationToken, Progress
from .typing import (
    ColumnLabel,
    DiscoveryOptions,
    ExplicitDiscoveryOptions,
    FoundationOptions,
    SchemaRole,
    Section,
    SectionOptions,
)


@overload
def explore(
    df: pd.DataFrame,
    dimensions: None = None,
    *,
    discovery: DiscoveryOptions | None = None,
    sections: Iterable[Section] | None = None,
    section_options: SectionOptions | None = None,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    features: Iterable[str] | None = None,
    progress: Progress = None,
    cancel: CancellationToken | None = None,
    timeout: float | None = None,
) -> InvestigationResult: ...


@overload
def explore(
    df: pd.DataFrame,
    dimensions: Iterable[ColumnLabel],
    *,
    discovery: ExplicitDiscoveryOptions | None = None,
    scope: Scope | None = None,
    missing: Mapping[ColumnLabel, Iterable[Any]] | None = None,
    table_id: str = "table",
    features: Iterable[ColumnLabel] | None = None,
    candidate_keys: Iterable[ColumnLabel | KeySpec] | None = None,
    top_n: int | None = None,
    top_n_mode: Literal["pre", "post"] = "post",
    top_n_per_parent: bool = False,
    top_n_applies_to: Literal["census", "both"] = "census",
    min_retained_fraction: float = 0.01,
    max_depth: int | None = None,
    max_levels: int | None = 100,
    max_nodes: int | None = 10000,
    min_count: int = 1,
    dropna: bool = False,
    schema: dict[ColumnLabel, SchemaRole] | None = None,
    include_pairs: bool = True,
    include_absence: bool = False,
    reference_domains: Mapping[ColumnLabel, Iterable[Any]] | None = None,
    pair_contexts: Iterable[Mapping[ColumnLabel, Any]] | None = None,
    max_absence_cells: int | None = 1000,
    max_contexts: int | None = 32,
    max_pairs: int | None = 15,
    engine_metadata: bool = False,
    progress: Progress = None,
    cancel: CancellationToken | None = None,
    timeout: float | None = None,
) -> ExplorerResult: ...


@operation("overview")
def explore(
    df: pd.DataFrame,
    dimensions: Iterable[ColumnLabel] | None = None,
    *,
    discovery: DiscoveryOptions | ExplicitDiscoveryOptions | None = None,
    sections: Iterable[Section] | None = None,
    section_options: SectionOptions | None = None,
    progress: Progress = None,
    cancel: CancellationToken | None = None,
    timeout: float | None = None,
    **options: Unpack[FoundationOptions],
) -> ExplorerResult:
    """Explore a table automatically or compose explicit foundational analyses.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation. Automatic mode requires unique string
        columns; explicit mode also supports integer and recursively tuple labels.
    dimensions : iterable of column labels or None, optional
        Default None runs automatic discovery and returns InvestigationResult.
        An explicit nonempty ordered collection composes levels, census, optional
        grain, and pairs and returns ExplorerResult.
    discovery : DiscoveryOptions or ExplicitDiscoveryOptions or None, optional
        Default None uses mode defaults. Automatic mode accepts path-search
        settings plus shared features/scope/missing/table_id, by, and entity
        settings. Search budgets affect paths only. Explicit mode accepts only
        features/scope/missing/table_id and rejects duplicates supplied directly.
    sections : iterable of str or None, optional
        Automatic mode only: distinct names from 'missingness', 'dependencies',
        'paths', 'value_patterns'. Default None requests all four; an empty
        collection is invalid. Omitted sections are marked not_requested.
    section_options : SectionOptions or None, optional
        Automatic mode only: per-requested-section analytical overrides; default
        None. Cannot override scope, missing, table_id, or runtime controls.
        Use this to set independent feature choices and search budgets.
    progress : bool or callable, optional
        Default None is silent; True uses the built-in display. A callback receives
        ProgressEvent objects synchronously. False is also silent. Callback errors
        propagate unchanged; do not mutate the frame from a callback.
    cancel : CancellationToken or None, optional
        Cooperative cancellation token; default None. A cancelled token raises
        AnalysisCancelled at the next checkpoint, with no partial result.
    timeout : float or None, optional
        Finite nonnegative seconds from call start; default None disables the
        deadline. Expiration raises AnalysisCancelled cooperatively, after the
        current pandas/NumPy work item returns, rather than at a hard deadline.
    **options : Unpack[OverviewOptions] or Unpack[FoundationOptions]
        Automatic mode accepts only scope=None, missing=None, table_id="table",
        and features=None; explicit keys replace matching discovery settings.
        Explicit mode additionally accepts census options (top_n=None,
        top_n_mode="post", top_n_per_parent=False, min_retained_fraction=0.01,
        max_depth=None, max_levels=100, max_nodes=10000, min_count=1, dropna=False,
        schema=None, engine_metadata=False), candidate_keys=None,
        top_n_applies_to="census", include_pairs=True, include_absence=False,
        reference_domains=None, pair_contexts=None, max_absence_cells=1000,
        max_contexts=32, and max_pairs=15. See census and pairs for meanings.
        top_n_applies_to="both" also restricts pair/grain cohorts and requires
        top_n with pre mode. include_absence requires include_pairs.

    Other Parameters
    ----------------
    scope : Scope or None, optional
        Source-bound population selection; default None uses all rows. The scope
        must match the ordered source. Fingerprinting still scans the full frame.
    missing : mapping or None, optional
        Additional missing sentinels per column; default None. Native missing
        values are always absent. Numeric sentinels match integer/float values
        numerically; booleans remain distinct. The source is not modified.
    table_id : str, optional
        Nonempty source label; default 'table'. Does not replace the fingerprint.
    features : iterable of column labels or None, optional
        Automatic mode: unique string columns for all sections, default None
        selects all, skipping columns with unsupported values (listed in
        skipped_features). Explicit mode: typed labels for independent levels,
        default None uses dimensions. Section options can override automatic
        features.
    candidate_keys : iterable of column labels or KeySpec or None, optional
        Explicit mode only. Default None skips grain. Labels denote single-column
        keys; use KeySpec for composite determinants. Tuple labels denote one
        column, not a composite key. Names and components must be unique.
    top_n : int or None, optional
        Positive number of leading levels; default None keeps all eligible levels.
        With pre mode this selects a cohort; with post mode it only limits output.
    top_n_mode : {'pre', 'post'}, optional
        Default 'post' counts the full eligible population before limiting output.
        'pre' restricts rows to selected levels before counting, records exclusions,
        and can warn about low retention.
    top_n_per_parent : bool, optional
        Default False chooses leading levels globally for each dimension. True
        chooses them separately within each parent prefix.
    top_n_applies_to : {'census', 'both'}, optional
        Explicit mode only. Default 'census' limits census alone. 'both' applies
        its selected cohort to pairs and grain as well, and requires top_n with
        top_n_mode='pre'. Independent levels retain their own population.
    min_retained_fraction : float, optional
        Retention warning threshold in [0, 1]; default 0.01. Does not reject or
        change the selected population.
    max_depth : int or None, optional
        Positive number of active dimensions; default None uses all dimensions.
    max_levels : int or None, optional
        Nonnegative displayed child-level limit per parent; default 100. None is
        unbounded; zero omits all child levels. Omitted mass remains reported.
    max_nodes : int or None, optional
        Nonnegative total non-root node budget; default 10000. None is unbounded;
        zero keeps only the root and omission evidence.
    min_count : int, optional
        Nonnegative minimum displayed count; default 1. Does not filter input rows.
    dropna : bool, optional
        Explicit mode only. Default False includes native/declared missing levels.
        True uses each component's complete cases: independently per levels
        feature, over active census dimensions, per pair/context, and per
        determinant/target. Denominators can therefore differ across sections.
    schema : dict or None, optional
        Advisory roles by column: 'id', 'categorical', 'continuous', or 'unknown'.
        Default None. Roles annotate evidence and warnings; they do not cast values.
    include_pairs : bool, optional
        Explicit mode only. Default True computes bounded pair relationships.
        False marks pairs as not_requested and cannot be used with include_absence.
    include_absence : bool, optional
        Include absent domain combinations when True; default False; requires include_pairs=True. Observed
        mapping and association evidence is computed independently.
    reference_domains : mapping or None, optional
        Optional declared value domains by column; default None uses observed
        domains. Absence means unobserved in the evaluated population, not invalid.
    pair_contexts : iterable of mappings or None, optional
        Additional exact column-to-value context filters; default None. Context
        columns must be disjoint from the evaluated pair. Global evidence remains.
    max_absence_cells : int or None, optional
        Nonnegative absent-cell output budget; default 1000. None is unbounded;
        zero retains absence totals without enumerating cells.
    max_contexts : int or None, optional
        Nonnegative total context budget, including the global population; default
        32. None is unbounded; zero skips all pair/context records; one keeps
        only global pair evidence.
    max_pairs : int or None, optional
        Nonnegative pair budget; default 15. None is unbounded; zero skips pairs.
        Omitted tests are reported, not treated as failed relationships.
    engine_metadata : bool, optional
        Explicit mode only. Default False omits producer metadata. True includes
        the combined result's orchestrator identifier.

    Returns
    -------
    InvestigationResult or ExplorerResult
        Automatic mode returns kind 'overview', with linked findings, feature
        relationships, and independent section results. Overview findings are
        ranked as leads (f0 first), each with lead.score and lead.reason; section
        results keep their own order. Explicit mode returns
        kind 'explore', with levels/census/grain/pairs sections. Unrequested
        sections carry status='not_requested'.

    Raises
    ------
    KeyError
        A requested column is unknown.
    ValueError
        Columns, limits, thresholds, constraints, or source scope are invalid.
    TypeError
        The frame, column labels, or scalar values are unsupported.
    AnalysisCancelled
        Cancellation or the cooperative timeout stops analysis.

    Notes
    -----
    Automatic dependencies default to single-column keys and 20 candidates;
    value patterns default to 20 pairs. Standalone calls have different budgets.
    Entity settings affect availability only; dependency and pattern findings
    still count rows and retain their own analysis units. Shared context never
    changes source identity. Runtime controls belong on this call, not inside
    configuration dictionaries. Section options do not apply in explicit mode.

    See Also
    --------
    fieldwork.typing.DiscoveryOptions, fieldwork.typing.SectionOptions
    fieldwork.typing.FoundationOptions, census, pairs

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"site": ["A", "A", "B"], "visit": [1, 2, 1]})
    >>> fw.explore(df, sections=["missingness"]).kind
    'overview'
    >>> fw.explore(df, ["site", "visit"], include_pairs=False).kind
    'explore'
    """
    if dimensions is not None:
        if sections is not None or section_options is not None:
            raise ValueError("sections and section_options apply to discovery overviews only")
        config = dict(discovery or {})
        incompatible = config.keys() - {"scope", "missing", "table_id", "features"}
        if incompatible:
            raise ValueError(
                f"Explicit dimensions do not accept discovery search options: {sorted(incompatible)}"
            )
        duplicate = config.keys() & options.keys()
        if duplicate:
            raise ValueError(f"Configuration supplied twice: {sorted(duplicate)}")
        options = {**config, **options}
        context = {k: options.pop(k) for k in ("scope", "missing", "table_id") if k in options}
        if context:
            return foundation_context(df, explicit_explore, dimensions, **context, **options)
        return explicit_explore(df, dimensions, **options)
    if options.keys() - {"scope", "missing", "table_id", "features"}:
        raise TypeError("With omitted dimensions, configure search with discovery dictionary")
    # Run-time population overrides retain all other configured search settings.
    config = {**(discovery or {}), **options}
    if {"progress", "cancel", "timeout"} & config.keys():
        raise ValueError("Pass runtime controls directly to explore, not in discovery")
    shared = {k: v for k, v in config.items() if k in {"scope", "missing", "table_id", "features"}}
    availability_options = {
        k: config.pop(k) for k in ("entity", "unit", "entity_presence") if k in config
    }
    contexts = config.pop("by", None)
    names = ("missingness", "dependencies", "paths", "value_patterns")
    requested = list(names if sections is None else sections)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(names):
        raise ValueError(f"sections must contain distinct names from {names}")
    section_options = dict(section_options or {})
    if section_options.keys() - set(requested):
        raise ValueError("section_options must refer to requested sections")
    protected = {"scope", "missing", "table_id", "progress", "cancel", "timeout"}
    functions = {
        "paths": suggest_paths,
        "missingness": missingness,
        "dependencies": discover_dependencies,
        "value_patterns": value_patterns,
    }
    configs = {
        "paths": dict(config),
        "missingness": {"by": contexts, **availability_options, **shared},
        "dependencies": {"by": contexts, "max_key_size": 1, "max_candidates": 20, **shared},
        "value_patterns": {"by": contexts, "max_pairs": 20, **shared},
    }
    # Validate all options before starting potentially expensive work.
    inspect.signature(suggest_paths).bind(df, **config)
    for name in requested:
        overrides = dict(section_options.get(name, {}))
        if protected & overrides.keys() or any(k.startswith("_") for k in overrides):
            raise ValueError("section_options cannot override source context or runtime controls")
        configs[name].update(overrides)
        inspect.signature(functions[name]).bind(df, **configs[name])
    analyses = {}
    for name in ("paths", "missingness", "dependencies", "value_patterns"):
        if name in requested:
            analyses[name] = functions[name](df, **configs[name])
    if "missingness" in analyses:
        base = dict(analyses["missingness"].payload)
    else:
        base = prepare(df, features=[], **{k: v for k, v in shared.items() if k != "features"})[-1]
    base["sections"] = {
        name: analyses[name].to_dict() if name in analyses else {"status": "not_requested"}
        for name in names
    }
    skipped = {}
    for analysis in analyses.values():
        for record in analysis["skipped_features"]:
            skipped.setdefault(record["feature"], record)
    base["skipped_features"] = list(skipped.values())
    paths = analyses.get("paths")
    if paths is not None and paths.best:
        base["sections"]["census"] = paths["paths"][0]["preview"]
    if sections is not None or section_options:
        base["section_selection"] = {
            "requested": [n for n in names if n in requested],
            "omitted": [n for n in names if n not in requested],
        }
    findings = []
    for section in names:
        if section not in analyses:
            continue
        for record in analyses[section]["findings"]:
            findings.append(
                {
                    **record,
                    "selector": {
                        **record["selector"],
                        "analysis_section": section,
                        "scope_ref": f"sections.{section}.scope",
                        "parameters_ref": f"sections.{section}.parameters",
                        "missing_convention_ref": f"sections.{section}.missing_convention",
                    },
                }
            )
    constant = set()
    for c in df.columns:
        try:
            if df[c].nunique(dropna=True) <= 1:
                constant.add(c)
        except TypeError:  # Unhashable cells: such columns are skipped anyway.
            pass
    # Overview IDs follow lead rank, so f0 is the most promising finding.
    base["findings"] = [
        {**record, "id": f"f{i}"} for i, record in enumerate(rank(findings, constant))
    ]
    with phase("assembling overview"):
        base["feature_network"] = feature_network(base)
    return result("overview", base)


@dataclass(frozen=True)
class Recipe:
    """Store reusable, strict-JSON parameters for an allowlisted operation.

    Parameters
    ----------
    operation : str
        One of 'missingness', 'dependencies', 'paths', 'value_patterns', 'explore',
        'census', 'grain', 'levels', or 'joint_counts'. Use 'dependencies' for
        discover_dependencies and 'paths' for suggest_paths.
    parameters : dict[str, Any], optional
        Operation keyword arguments; default is a new empty dictionary. Must be
        strict JSON. Source-bound scope and progress/cancel/timeout controls
        belong in run overrides, not persisted parameters.
    notes : str, optional
        Free-text notes; default empty string.
    version : str, optional
        Recipe format version; default and only supported value is '1.0'.

    Attributes
    ----------
    operation : str
        Allowlisted operation identifier.
    parameters : dict[str, Any]
        Saved options. Nested parameters are mutable despite the frozen record.
    notes : str
        User notes retained in saved JSON.
    version : str
        Recipe format version, independent of evidence/package versions.

    Raises
    ------
    ValueError
        Version/operation is unsupported, runtime controls/scope are persisted,
        or parameters contain nonfinite JSON numbers.
    TypeError
        Parameters contain values not serializable as JSON, such as KeySpec,
        Scope, timestamps, or callbacks.

    Notes
    -----
    Recipes reapply parameters to new deliveries; evidence and scopes retain old
    source identities. Construction validates serializability, not every operation
    argument. Operation-specific validation occurs at run time. JSON decoding
    converts tuples to lists. Runtime overrides do not mutate saved parameters.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> recipe = fw.Recipe('missingness', {'features': ['value']})
    >>> recipe.run(pd.DataFrame({'value': [1, None]})).kind
    'missingness'
    """

    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if self.version != "1.0" or self.operation not in self.operations():
            raise ValueError("Unsupported recipe version or operation")
        if {"progress", "cancel", "timeout"} & self.parameters.keys():
            raise ValueError(
                "Runtime controls belong in Recipe.run overrides, not saved parameters"
            )
        if "scope" in self.parameters:
            raise ValueError(
                "Recipes reapply to deliveries; pass a scope when running, not in the recipe"
            )
        json.dumps(self.to_dict(), allow_nan=False)

    @staticmethod
    def operations() -> dict[str, Callable[..., ExplorerResult]]:
        """Return the supported recipe operation registry.

        Returns
        -------
        dict[str, Callable[..., ExplorerResult]]
            New mapping from persisted operation names to public callables. Editing
            this returned dictionary does not register or replace operations.
        """
        from ._explore import census, grain, joint_counts, levels

        return {
            "missingness": missingness,
            "dependencies": discover_dependencies,
            "paths": suggest_paths,
            "value_patterns": value_patterns,
            "explore": explore,
            "census": census,
            "grain": grain,
            "levels": levels,
            "joint_counts": joint_counts,
        }

    @operation("recipe")
    def run(
        self,
        df: pd.DataFrame,
        *,
        progress: Progress = None,
        cancel: CancellationToken | None = None,
        timeout: float | None = None,
        **overrides: Any,
    ) -> ExplorerResult:
        """Apply saved parameters to a delivery, with explicit overrides.

        Parameters
        ----------
        df : pandas.DataFrame
            Delivery to analyze; may differ from prior recipe runs. Supply any Scope
            override created from this delivery.
        progress : bool or callable, optional
            Default None is silent; True uses the built-in display. A callback receives
            ProgressEvent objects synchronously. False is also silent. Callback errors
            propagate unchanged; do not mutate the frame from a callback.
        cancel : CancellationToken or None, optional
            Cooperative cancellation token; default None. A cancelled token raises
            AnalysisCancelled at the next checkpoint, with no partial result.
        timeout : float or None, optional
            Finite nonnegative seconds from call start; default None disables the
            deadline. Expiration raises AnalysisCancelled cooperatively, after the
            current pandas/NumPy work item returns, rather than at a hard deadline.
        **overrides : Any
            Operation-specific keyword overrides, such as scope or example_limit.
            Explicit keys replace saved parameters without modifying the recipe. For
            automatic explore, scope/missing/table_id/features replace corresponding
            discovery entries while preserving other discovery settings.

        Returns
        -------
        ExplorerResult
            Result of the named operation. Discovery operations return
            InvestigationResult; paths returns PathResult. The concrete type depends
            on the recipe's runtime operation and, for explore, its dimensions.

        Raises
        ------
        KeyError
            A requested column is unknown.
        ValueError
            Columns, limits, thresholds, constraints, or source scope are invalid.
        TypeError
            The frame, column labels, or scalar values are unsupported.
        AnalysisCancelled
            Cancellation or the cooperative timeout stops analysis.

        Notes
        -----
        Runtime controls are never saved into evidence or recipe parameters. Nested
        analyses share this call's cancellation/progress context. Invalid or unsupported
        operation arguments are rejected by the selected operation.
        """
        return self.operations()[self.operation](df, **{**self.parameters, **overrides})

    def to_dict(self) -> dict[str, Any]:
        """Export the recipe configuration as ordinary JSON-compatible fields.

        Returns
        -------
        dict[str, Any]
            Version, operation, parameters, and notes. The top-level mapping is new;
            the parameters dictionary remains shared with the recipe.
        """
        return {
            "version": self.version,
            "operation": self.operation,
            "parameters": self.parameters,
            "notes": self.notes,
        }

    def save(self, path: str | PathLike[str]) -> None:
        """Write the recipe as indented strict JSON.

        Parameters
        ----------
        path : str or os.PathLike[str]
            Destination file. Existing contents are overwritten; parent directories
            are not created.

        Returns
        -------
        None
            Writes the complete recipe followed by a newline.

        Raises
        ------
        OSError
            The destination cannot be written.
        TypeError or ValueError
            Mutated parameters are no longer strict-JSON serializable.
        """
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Recipe:
        """Read and validate a saved recipe JSON file.

        Parameters
        ----------
        path : str or os.PathLike[str]
            Existing recipe JSON file to read.

        Returns
        -------
        Recipe
            Restored recipe with validated version, operation, and strict JSON options.

        Raises
        ------
        OSError
            The file cannot be read.
        ValueError
            JSON is malformed or the recipe version/operation/parameters are invalid.
        TypeError
            Required constructor fields are absent or extra fields are supplied.
        """
        return cls(**json.loads(Path(path).read_text()))


def compare(before: InvestigationResult, after: InvestigationResult) -> InvestigationResult:
    """Compare populated fractions by feature across two availability results.

    Parameters
    ----------
    before : InvestigationResult
        Earlier missingness result. Features align by name, not position.
    after : InvestigationResult
        Later missingness result with compatible counting unit, entity keys, and
        aggregation. Source deliveries and scopes may differ.

    Returns
    -------
    InvestigationResult
        Kind 'comparison', with changes, findings, and both source identities,
        conventions, scopes, and analysis units. populated_fraction_delta is
        after minus before; absent features or empty denominators yield None.

    Raises
    ------
    ValueError
        Either result is not missingness, or counting units/entity aggregation
        differ.

    Notes
    -----
    Delta is a fraction (0.25 means 25 percentage points), not relative percent
    change. A comparison preserves evidence about both populations; it does not
    establish that their selection or missing conventions are equivalent.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> before = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> after = fw.missingness(pd.DataFrame({"x": [1, 2]}))
    >>> fw.compare(before, after)["changes"][0]["populated_fraction_delta"]
    0.5
    """
    if before.kind != "missingness" or after.kind != "missingness":
        raise ValueError("compare accepts two missingness results")

    def analysis_unit(analysis):
        return analysis.payload.get(
            "analysis_unit",
            {
                "counting_unit": "rows",
                "entity_keys": [],
                "denominator": analysis["scope"]["evaluated_rows"],
                "presence_aggregation": "per_row",
            },
        )

    def counting(analysis):
        unit = analysis_unit(analysis)
        if unit.get("counting_unit", "rows") == "rows":
            return ("rows",)
        return ("entities", unit["entity_keys"], unit["presence_aggregation"])

    if counting(before) != counting(after):
        raise ValueError("Comparison requires the same analysis unit, entity keys and aggregation")
    left = {r["feature"]: r for r in before["availability"]}
    right = {r["feature"]: r for r in after["availability"]}
    records = []
    for c in sorted(left.keys() | right.keys()):
        a, b = left.get(c), right.get(c)
        delta = (
            b["populated_fraction"] - a["populated_fraction"]
            if a and b and a["denominator"] and b["denominator"]
            else None
        )
        records.append({"feature": c, "before": a, "after": b, "populated_fraction_delta": delta})
    base = {
        "source": after["source"],
        "scope": after["scope"],
        "before_source": before["source"],
        "after_source": after["source"],
        "before_scope": before["scope"],
        "after_scope": after["scope"],
        "before_analysis_unit": analysis_unit(before),
        "after_analysis_unit": analysis_unit(after),
        "before_convention": before["missing_convention"],
        "after_convention": after["missing_convention"],
        "changes": records,
        "analysis_unit": analysis_unit(after),
        "findings": [],
    }
    for record in records:
        finding(
            base,
            "availability_change",
            f"{record['feature']}: availability before → after",
            [record["feature"]],
            record,
            [],
            example_limit=0,
            unit=counting(after)[0],
        )
    return result("comparison", base)
