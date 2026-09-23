"""Deterministic beam search over census prefixes and supported nesting."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from functools import cache
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._explore import census
from ._explore._kernels import group_ids, modal_groups, pair_groups
from ._explore.result import ExplorerResult
from ._runtime import checkpoint, operation, phase
from .evidence import (
    InvestigationResult,
    Scope,
    analyzable,
    columns,
    finding,
    fingerprint,
    limit,
    prepare,
    saved_context,
)
from .progress import CancellationToken, Progress
from .typing import ColumnLabel, SchemaRole


class PathResult(InvestigationResult):
    """A discovery result containing ranked, source-bound census recommendations.

    Parameters
    ----------
    kind : str
        'paths' for results returned by suggest_paths.
    payload : dict, optional
        Path evidence; normally supplied by suggest_paths or from_dict.
    schema_version : str, optional
        Producers/loaders use discovery '1.0'; the inherited raw constructor
        defaults to foundation '0.3'. Prefer the producer/loader.
    stability : str, optional
        Evidence stability marker; default 'unstable'.

    Attributes
    ----------
    best : Path or None
        First ranked path, or None if none is available.
    payload : dict[str, Any]
        Paths with dimensions, measurements, reasons, and previews, plus aliases,
        nesting and search coverage. Also includes common investigation evidence.

    Notes
    -----
    Inherits the mapping, serialization, inspection, and selection methods of
    InvestigationResult. Path rankings concern observed prefixes, not guarantees
    about the data's true schema. JSON restoration retains context-aware handoff.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> paths = fw.suggest_paths(pd.DataFrame({'x': [1, 2]}))
    >>> paths.best.dimensions
    ('x',)
    """

    @property
    def best(self) -> Path | None:
        """Return the highest-ranked path, when one is available.

        Returns
        -------
        Path or None
            Source-bound first recommendation, or None for an empty path list.
        """
        return self.path(0) if self.payload["paths"] else None

    def path(self, index: int = 0) -> Path:
        """Return a ranked recommendation with its original analysis context.

        Parameters
        ----------
        index : int, optional
            Zero-based path position; default 0. Negative positions follow Python list
            indexing. This indexes paths, not the findings list.

        Returns
        -------
        Path
            Recommended dimensions plus saved source, scope, and missing conventions.

        Raises
        ------
        IndexError
            The requested path does not exist.
        """
        return Path(self.payload["paths"][index]["dimensions"], self.payload)


class Path:
    """An ordered census recommendation retaining its source analysis context.

    Parameters
    ----------
    dimensions : iterable of str
        Recommended ordered columns, normalized to a tuple.
    context : mapping
        Saved path-result payload containing source, scope, and missing
        conventions. Obtain Path from PathResult.best or PathResult.path instead
        of assembling this context manually.

    Attributes
    ----------
    dimensions : tuple[str, ...]
        Ordered recommended columns. This tuple alone does not carry source
        context; call census on this Path to retain it.

    Notes
    -----
    The context references saved evidence; it is not a copy of the source frame.
    The census handoff verifies the original source and preserves its scope and
    sentinels. To change population or delivery, rerun suggest_paths.
    """

    dimensions: tuple[str, ...]
    """Ordered recommended columns; use census() to preserve source context."""

    def __init__(self, dimensions: Iterable[str], context: Mapping[str, Any]) -> None:
        self.dimensions = tuple(dimensions)
        self._context = context

    @operation("path census")
    def census(
        self,
        df: pd.DataFrame,
        *,
        top_n: int | None = None,
        top_n_mode: Literal["pre", "post"] = "post",
        top_n_per_parent: bool = False,
        min_retained_fraction: float = 0.01,
        max_depth: int | None = None,
        max_levels: int | None = 100,
        max_nodes: int | None = 10000,
        min_count: int = 1,
        dropna: bool = False,
        schema: dict[ColumnLabel, SchemaRole] | None = None,
        engine_metadata: bool = False,
        progress: Progress = None,
        cancel: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> ExplorerResult:
        """Evaluate this recommendation with its original scope and missing conventions.

        Parameters
        ----------
        df : pandas.DataFrame
            Original ordered source frame; labels, index, and values must match the
            saved fingerprint. Duplicate index labels are supported.
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
            Default False includes missing values as levels. True excludes rows
            missing any active dimension before census counting.
        schema : dict or None, optional
            Advisory roles by column: 'id', 'categorical', 'continuous', or 'unknown'.
            Default None. Roles annotate evidence and warnings; they do not cast values.
        engine_metadata : bool, optional
            Include analytical producer metadata when True; default False.
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

        Returns
        -------
        ExplorerResult
            Census for the recommended ordered dimensions with original-source scope
            accounting. Display/cohort options use ordinary census defaults.

        Raises
        ------
        ValueError
            The source differs or census options are invalid.
        TypeError
            Unsupported options, including scope, missing, table_id, or dimensions,
            are supplied. Context cannot be overridden through a recommendation.
        AnalysisCancelled
            Cancellation or the cooperative timeout stops analysis.

        Notes
        -----
        The handoff preserves context after JSON restoration. Display limits can be
        changed; explicit census preselection can further restrict the evaluated
        cohort and records the additional exclusion. Rerun discovery to change source
        scope or missing conventions.
        """
        if fingerprint(df) != self._context["source"]["dataset_id"]:
            raise ValueError("Source dataset differs; reapply a path recipe for a new delivery")
        return census(
            df,
            self.dimensions,
            **saved_context(self._context),
            top_n=top_n,
            top_n_mode=top_n_mode,
            top_n_per_parent=top_n_per_parent,
            min_retained_fraction=min_retained_fraction,
            max_depth=max_depth,
            max_levels=max_levels,
            max_nodes=max_nodes,
            min_count=min_count,
            dropna=dropna,
            schema=schema,
            engine_metadata=engine_metadata,
        )


@operation("paths")
def suggest_paths(
    df: pd.DataFrame,
    *,
    objective: Literal["structure", "availability", "compact", "target", "context"] = "structure",
    features: Iterable[str] | None = None,
    start_with: Iterable[str] | None = None,
    before: Iterable[tuple[str, str]] | None = None,
    exclude: Iterable[str] | None = None,
    target: str | None = None,
    max_dimensions: int = 4,
    max_candidates: int = 200,
    max_features: int = 20,
    max_pairs: int = 200,
    beam_width: int = 12,
    n_paths: int = 3,
    display_budget: int = 40,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    progress: Progress = None,
    cancel: CancellationToken | None = None,
    timeout: float | None = None,
) -> PathResult:
    """Recommend ordered census dimensions from observed prefix evidence.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation. Discovery requires unique string
        column names. Duplicate index labels are supported; source selections use
        integer row positions. Unsupported scalar objects raise TypeError.
    objective : {'structure', 'availability', 'compact', 'target', 'context'}, optional
        Default 'structure' favors coarse-to-fine nesting. 'compact' favors small
        prefixes; 'availability' separates presence signatures; 'target' separates
        target values and requires target; 'context' favors nesting after required
        start_with dimensions. Scores are heuristic costs, not probabilities.
    features : iterable of str or None, optional
        Unique column names to analyze, in requested order; default None selects
        all columns, skipping those with unsupported values (such as lists,
        dicts or Decimal) and listing them in skipped_features. Restricts
        analysis, not full-source identity validation.
    start_with : iterable of str or None, optional
        Required ordered initial columns; default None. Must fit both feature and
        dimension budgets. Required for the context objective.
    before : iterable of (str, str) pairs or None, optional
        Acyclic precedence constraints; default None. Both columns must occur in
        the path. Constraints must agree with start_with and exclusions.
    exclude : iterable of str or None, optional
        Columns excluded from paths; default None. Cannot contain required columns.
    target : str or None, optional
        Feature to explain; default None. Required by the target objective. Omitted
        from candidate paths unless also explicitly required by steering.
    max_dimensions : int, optional
        Positive maximum path length; default 4. Must accommodate steering columns.
    max_candidates : int, optional
        Positive maximum path extensions evaluated; default 200. Exhaustion is
        reported and can leave paths shorter than the requested depth.
    max_features : int, optional
        Positive candidate feature budget; default 20. Required columns survive
        truncation; other columns follow requested order.
    max_pairs : int, optional
        Nonnegative pair budget for observed nesting and aliases; default 200.
        Zero skips nesting/alias tests.
    beam_width : int, optional
        Positive alternatives retained per depth; default 12. Keeps the best order
        per feature set so alternatives need not be permutations of one set.
    n_paths : int, optional
        Positive maximum returned alternatives; default 3. Alias substitutions
        are collapsed; fewer paths may be available.
    display_budget : int, optional
        Positive preview node limit and prefix-cost reference; default 40. Affects
        ranking as well as preview size, but never samples source rows.
    scope : Scope or None, optional
        Source-bound population selection; default None uses all rows. The scope
        must match the ordered source. Fingerprinting still scans the full frame.
    missing : mapping or None, optional
        Additional missing sentinels per column; default None. Native missing
        values are always absent. Numeric sentinels match integer/float values
        numerically; booleans remain distinct. The source is not modified.
    table_id : str, optional
        Nonempty source label; default 'table'. Does not replace the fingerprint.
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

    Returns
    -------
    PathResult
        Kind 'paths', with ranked paths, measurements, explanations, aliases,
        nesting, and coverage. best is None when no nonempty path is available.

    Raises
    ------
    KeyError
        A requested column is unknown.
    ValueError
        Columns, limits, thresholds, constraints, or source scope are invalid.
    TypeError
        The frame or column labels are unsupported, or an explicitly requested
        column contains unsupported values.
    AnalysisCancelled
        Cancellation or the cooperative timeout stops analysis.

    Notes
    -----
    Search and display budgets never sample rows. Evidence records evaluated
    populations and omissions separately. Source identity covers ordered column
    labels, index labels, column dtypes and all cell values; changing or
    reordering them invalidates inspection against saved findings.

    Missing values form categories under the saved missing convention. Rankings
    use intermediate prefixes: joint information alone is invariant to order.
    Call paths.best.census(df) to preserve source, scope, and sentinel context;
    copying best.dimensions alone discards that context.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"site": ["A", "A", "B"], "visit": [1, 2, 1]})
    >>> paths = fw.suggest_paths(df, start_with=["site"])
    >>> paths.best is not None
    True
    >>> paths.best.census(df).kind
    'census'
    """
    objectives = {"structure", "availability", "compact", "target", "context"}
    if objective not in objectives:
        raise ValueError(f"objective must be one of {sorted(objectives)}")
    for name, value in [
        ("max_dimensions", max_dimensions),
        ("max_candidates", max_candidates),
        ("max_features", max_features),
        ("beam_width", beam_width),
        ("n_paths", n_paths),
        ("display_budget", display_budget),
    ]:
        limit(name, value, minimum=1)
    limit("max_pairs", max_pairs)
    requested = columns(df, features)
    excluded = set(columns(df, exclude or []))
    starts = columns(df, start_with or [])
    constraints = list(before or [])
    for edge in constraints:
        if len(edge) != 2:
            raise ValueError("before entries must be pairs of columns")
        columns(df, edge)
    required = set(starts) | {c for edge in constraints for c in edge}
    if required & excluded or not required <= set(requested):
        raise ValueError("Steering columns must be included and not excluded")
    if len(starts) > max_dimensions or len(required) > max_dimensions:
        raise ValueError("max_dimensions cannot fit steering constraints")
    if objective == "context" and not starts:
        raise ValueError("context objective requires start_with")
    if objective == "target" and target is None:
        raise ValueError("target objective requires target")
    if target is not None:
        columns(df, [target])
    pool = [c for c in requested if c not in excluded and (c != target or c in required)]
    # Required columns survive the feature budget; otherwise use input order.
    selected = [c for c in pool if c in required] + [c for c in pool if c not in required]
    if len(required) > max_features:
        raise ValueError("max_features cannot fit steering columns")
    selected = selected[:max_features]
    frame, positions, encoded, present, base = prepare(
        df,
        scope=scope,
        missing=missing,
        table_id=table_id,
        features=[*selected, *([target] if target is not None else [])],
        optional=[c for c in selected if c not in required] if features is None else (),
    )
    selected = analyzable(selected, base)
    encoded = {c: np.where(present[c], code, -1) for c, code in encoded.items()}
    cardinality = {c: len(np.unique(encoded[c])) for c in selected}
    active = [c for c in selected if cardinality[c] > 1 or c in required]
    edges, aliases = set(), []
    tested_pairs = 0
    with phase("path nesting", min(math.comb(len(active), 2), max_pairs), "pairs") as tracker:
        for a, b in combinations(active, 2):
            if tested_pairs >= max_pairs:
                break
            tested_pairs += 1
            pairs = pair_groups(encoded[a], encoded[b])
            count = int(pairs.max()) + 1 if len(pairs) else 0
            a_to_b, b_to_a = count == cardinality[a], count == cardinality[b]
            if a_to_b and b_to_a:
                aliases.append([a, b])
            elif a_to_b:
                edges.add((b, a))  # coarse before finer
            elif b_to_a:
                edges.add((a, b))
            tracker.advance(detail=f"{a} / {b}")
    for a, b in constraints:
        if a == b:
            raise ValueError("before constraints must be acyclic")
    # Validate user constraints, separately from soft observed nesting.
    pending, emitted = set(required), set()
    while pending:
        ready = {c for c in pending if all(a in emitted for a, b in constraints if b == c)}
        if not ready:
            raise ValueError("before constraints contain a cycle")
        pending -= ready
        emitted |= ready
    for i, c in enumerate(starts):
        if any(b == c and a not in starts[:i] for a, b in constraints):
            raise ValueError("start_with conflicts with before constraints")
    target_codes = encoded[target] if target is not None else None

    @cache
    def availability_codes():
        packed = np.zeros((len(frame), (len(selected) + 7) // 8), dtype=np.uint8)
        for j, c in enumerate(selected):
            packed[:, j // 8] |= present[c].astype(np.uint8) << (7 - j % 8)
        return (
            np.unique(packed, axis=0, return_inverse=True)[1]
            if selected
            else np.zeros(len(frame), dtype=np.int64)
        )

    def impurity(labels, groups):
        if not len(labels):
            return 0.0
        _, sizes, _, maxima, _ = modal_groups(groups, labels)
        return int((sizes - maxima).sum()) / len(labels)

    prefix_cache = OrderedDict()
    cached_bytes = 0

    def prefix(path):
        nonlocal cached_bytes
        checkpoint()
        if path in prefix_cache:
            prefix_cache.move_to_end(path)
            return prefix_cache[path]
        ids = (
            group_ids([encoded[path[0]]])
            if len(path) == 1
            else pair_groups(prefix(path[:-1]), encoded[path[-1]])
        )
        if ids.nbytes <= 32 * 1024 * 1024:
            prefix_cache[path] = ids
            cached_bytes += ids.nbytes
            while cached_bytes > 32 * 1024 * 1024:
                _, removed = prefix_cache.popitem(last=False)
                cached_bytes -= removed.nbytes
        return ids

    @cache
    def measure(path, explain=False):
        prefixes, previous, redundancy = [], 1, 0
        target_losses, availability_losses = [], []
        for depth, c in enumerate(path, 1):
            keys = prefix(path[:depth])
            count = int(keys.max()) + 1 if len(keys) else 0
            prefixes.append(count)
            redundancy += count == previous
            previous = count
            if target_codes is not None and (explain or objective == "target"):
                target_losses.append(impurity(target_codes, keys))
            if explain or objective == "availability":
                availability_losses.append(impurity(availability_codes(), keys))
        inversions = sum(
            a in path and b in path and path.index(a) > path.index(b) for a, b in edges
        )
        alias_steps = sum(a in path and b in path for a, b in aliases)
        prefix_cost = sum(prefixes) / max(1, display_budget)
        overflow = sum(max(0, n - display_budget) for n in prefixes) / max(1, display_budget)
        # Objective penalties use prefix behavior: early useful splits and nesting.
        score = prefix_cost + overflow + 2 * redundancy + 3 * alias_steps
        if objective in {"structure", "context"}:
            score += 8 * inversions
        if objective == "target":
            score += 12 * sum(target_losses)
        if objective == "availability":
            score += 12 * sum(availability_losses)
        return {
            "score": score,
            "prefix_counts": prefixes,
            "prefix_cost": prefix_cost,
            "overflow": overflow,
            "nesting_inversions": inversions,
            "redundant_steps": redundancy,
            "alias_steps": alias_steps,
            "target_impurity_sum": sum(target_losses),
            "target_impurity_by_depth": target_losses,
            "availability_impurity_sum": sum(availability_losses),
            "availability_impurity_by_depth": availability_losses,
        }

    width = min(max_dimensions, len(active))
    beam = [tuple(starts)]
    evaluated = 0
    with phase("path search", max_candidates, "extensions budget") as tracker:
        for depth in range(len(starts), width):
            expanded = []
            for path in beam:
                for c in active:
                    if c in path or any(b == c and a not in path for a, b in constraints):
                        continue
                    next_path = (*path, c)
                    if len(required - set(next_path)) > width - len(next_path):
                        continue
                    if evaluated >= max_candidates:
                        break
                    evaluated += 1
                    expanded.append((measure(next_path)["score"], next_path))
                    tracker.advance(detail=" → ".join(next_path))
                if evaluated >= max_candidates:
                    break
            if not expanded:
                break
            # Future extensions depend on the selected set, so keep its best order.
            # This reserves beam slots for different feature choices, not permutations.
            best_sets = {}
            for _, path in sorted(expanded):
                best_sets.setdefault(frozenset(path), path)
            beam = list(best_sets.values())[:beam_width]
    beam = [p for p in beam if required <= set(p)]
    alias_representative = {c: c for c in active}
    for a, b in aliases:
        old, new = alias_representative[b], alias_representative[a]
        alias_representative = {
            c: new if representative == old else representative
            for c, representative in alias_representative.items()
        }
    diverse = {}
    for path in sorted(beam, key=lambda p: (measure(p)["score"], p)):
        diverse.setdefault(frozenset(alias_representative[c] for c in path), path)
    base["paths"] = []
    for path in list(diverse.values())[:n_paths]:
        if not path:
            continue
        metrics = measure(path, True)
        reasons, explanation = path_reasons(path, metrics, edges, aliases, target)
        preview = census(
            df,
            path,
            scope=scope,
            missing=missing or {},
            table_id=table_id,
            max_nodes=display_budget,
            max_levels=8,
        ).to_dict()
        base["paths"].append(
            {
                "dimensions": list(path),
                "measurements": metrics,
                "explanation": explanation,
                "reasons": reasons,
                "preview": preview,
            }
        )
        finding(
            base,
            "census_path",
            " → ".join(path),
            path,
            {**metrics, "explanation": explanation, "reasons": reasons},
            positions,
            example_limit=3,
        )
    for a, b in aliases:
        finding(
            base,
            "value_alias",
            f"{a} and {b}: equivalent value partitions",
            [a, b],
            {"evaluated_rows": len(frame), "groups": cardinality[a]},
            positions,
            example_limit=3,
            structure={"relation": "equivalent_value_partitions"},
        )
    base["aliases"] = aliases
    base["nesting"] = [list(e) for e in sorted(edges)]
    base["coverage"] = {
        "features_requested": len(pool),
        "features_evaluated": len(selected),
        "features_omitted": [c for c in pool if c not in selected],
        "pairs_evaluated": tested_pairs,
        "pair_candidates": math.comb(len(active), 2),
        "paths_evaluated": evaluated,
        "distinct_alternatives": len(diverse),
        "alternative_policy": "best_order_per_feature_set_collapsing_alias_substitutions",
        "search_exhausted_budget": evaluated >= max_candidates,
        "requested_depth": width,
        "returned_depth": max((len(p["dimensions"]) for p in base["paths"]), default=0),
    }
    base["parameters"] = {
        "objective": objective,
        "features": requested,
        "start_with": starts,
        "before": constraints,
        "exclude": sorted(excluded),
        "target": target,
        "max_dimensions": max_dimensions,
        "max_candidates": max_candidates,
        "max_features": max_features,
        "max_pairs": max_pairs,
        "beam_width": beam_width,
        "n_paths": n_paths,
        "display_budget": display_budget,
    }
    return PathResult("paths", base, schema_version="1.0")


def path_reasons(path, metrics, edges, aliases, target):
    """Name observed evidence contributing to a path's score."""
    counts = metrics["prefix_counts"]
    nesting = [list(edge) for edge in sorted(edges) if all(c in path for c in edge)]
    redundant = [
        c for c, previous, current in zip(path, [1, *counts], counts) if previous == current
    ]
    alias_pairs = [pair for pair in aliases if all(c in path for c in pair)]
    reasons = [
        {
            "kind": "branching",
            "dimensions": list(path),
            "prefix_groups": counts,
            "overflow_cost": metrics["overflow"],
        },
        {
            "kind": "nesting",
            "coarse_to_fine": nesting,
            "reversed_edges": metrics["nesting_inversions"],
        },
        {"kind": "redundancy", "no_new_groups": redundant, "equivalent_pairs": alias_pairs},
        {
            "kind": "availability_separation",
            "nonmodal_fraction_by_depth": metrics["availability_impurity_by_depth"],
        },
    ]
    text = ["Observed prefix groups: " + " → ".join(f"{c}: {n}" for c, n in zip(path, counts))]
    if nesting:
        text.append(
            "Supported coarse-to-fine nesting: " + ", ".join(f"{a} → {b}" for a, b in nesting)
        )
    if redundant:
        text.append("Adds no groups: " + ", ".join(redundant))
    if alias_pairs:
        text.append(
            "Equivalent value partitions: " + ", ".join(f"{a} / {b}" for a, b in alias_pairs)
        )
    text.append(
        "Availability nonmodal fractions: "
        + ", ".join(f"{v:.3g}" for v in metrics["availability_impurity_by_depth"])
    )
    if target is not None:
        reasons.append(
            {
                "kind": "target_separation",
                "target": target,
                "nonmodal_fraction_by_depth": metrics["target_impurity_by_depth"],
            }
        )
        text.append(
            f"{target} nonmodal fractions: "
            + ", ".join(f"{v:.3g}" for v in metrics["target_impurity_by_depth"])
        )
    return reasons, "; ".join(text)
