"""Deterministic beam search over census prefixes and supported nesting."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, Unpack

import numpy as np
import pandas as pd

from ._explore import census
from ._explore._kernels import group_ids, modal_groups, pair_groups
from ._runtime import checkpoint, operation, phase
from .evidence import (
    Scope,
    analyzable,
    budgets,
    columns,
    finding,
    fingerprint,
    limit,
    prepare,
    saved_context,
)
from .result import Result
from .typing import PathLimits, Runtime, SchemaRole


class Path:
    """An ordered census recommendation that keeps its source analysis context.

    Obtain one from ``Result.best`` or ``Result.path`` rather than constructing it.

    Parameters
    ----------
    dimensions : iterable of str
        Recommended ordered columns, stored as a tuple.
    context : mapping
        The saved paths payload (source, scope and missing conventions); a
        reference to saved evidence, not a copy of the source frame.
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
        schema: dict[str, SchemaRole] | None = None,
        **runtime: Unpack[Runtime],
    ) -> Result:
        """Census of these dimensions under the saved scope and missing conventions.

        Works on a restored (JSON) result too. Scope, missing conventions, table ID
        and dimensions come from the recommendation and cannot be passed; rerun
        suggest_paths to change them.

        Parameters
        ----------
        df : pandas.DataFrame
            The original ordered source; it must match the saved fingerprint.
        top_n, top_n_mode, top_n_per_parent, min_retained_fraction : optional
            Leading levels (default None: all), 'post' (default, limits output) or
            'pre' (selects a cohort), per parent (default False), and the
            retention warning threshold (default 0.01); see census.
        max_depth, max_levels, max_nodes, min_count : optional
            Active dimensions (default all), displayed children per parent (100),
            displayed nodes (10000) and minimum displayed count (1); see census.
        dropna : bool, optional
            Exclude rows missing an active dimension; default False.
        schema : dict or None, optional
            Advisory roles by column; default None.
        **runtime : Unpack[Runtime]
            Optional runtime controls; see fieldwork.typing.Runtime.

        Returns
        -------
        Result
            Kind 'census' for the recommended ordered dimensions.

        Raises
        ------
        ValueError
            ``df`` is not the saved source.
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
        )


_LIMITS = {
    "max_candidates": 200,
    "max_features": 20,
    "max_pairs": 200,
    "beam_width": 12,
    "display_budget": 40,
}


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
    n_paths: int = 3,
    limits: PathLimits | None = None,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Recommend orders of dimensions for a census, from observed prefix structure.

    Scores are heuristic costs of each prefix's groups (see docs/algorithms.md),
    not probabilities.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    objective : {'structure', 'availability', 'compact', 'target', 'context'}, optional
        'structure' (default) favors coarse-to-fine nesting; 'compact' small
        prefixes; 'availability' separating presence patterns; 'target'
        separating values of ``target``; 'context' nesting after ``start_with``.
    features : iterable of str or None, optional
        Candidate columns; default None selects every column, skipping (and
        listing) columns with unsupported values.
    start_with, before, exclude, target : optional
        Steering: required initial columns, acyclic (earlier, later) column
        pairs, excluded columns, and the column 'target' explains.
    max_dimensions, n_paths : int, optional
        Path length (default 4) and number of paths returned (default 3).
    limits : PathLimits or None, optional
        Search budgets: extensions scored (``max_candidates``, default 200),
        candidate columns (``max_features``, 20), column pairs tested for nesting
        (``max_pairs``, 200), alternatives kept per depth (``beam_width``, 12),
        and each preview's node budget, also the prefix-cost reference
        (``display_budget``, 40).
    scope, missing, table_id
        Source context shared by every analysis.
    **runtime : Unpack[Runtime]
        Optional runtime controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'paths': ranked ``paths`` (measurements, reasons, census preview),
        ``aliases``, ``nesting`` pairs and ``coverage``; ``best`` is the top Path.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"site": ["A", "A", "B"], "visit": [1, 2, 1]})
    >>> paths = fw.suggest_paths(df, start_with=["site"])
    >>> paths.best.dimensions
    ('site', 'visit')
    """
    objectives = {"structure", "availability", "compact", "target", "context"}
    if objective not in objectives:
        raise ValueError(f"objective must be one of {sorted(objectives)}")
    limit("max_dimensions", max_dimensions, minimum=1)
    limit("n_paths", n_paths, minimum=1)
    budget = budgets(limits, _LIMITS, positive=_LIMITS.keys() - {"max_pairs"})
    max_features, display_budget = budget["max_features"], budget["display_budget"]
    steering = _steering(df, features, exclude, start_with, before, target, objective)
    if len(steering.starts) > max_dimensions or len(steering.required) > max_dimensions:
        raise ValueError("max_dimensions cannot fit steering constraints")
    if len(steering.required) > max_features:
        raise ValueError("max_features cannot fit steering columns")
    # Required columns survive the feature budget; otherwise use input order.
    pool, required = steering.pool, steering.required
    selected = ([c for c in pool if c in required] + [c for c in pool if c not in required])[
        :max_features
    ]
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
    edges, aliases, tested_pairs = _nesting(encoded, active, cardinality, budget["max_pairs"])
    _check_constraints(steering)
    scorer = _Scorer(
        encoded,
        present,
        selected,
        edges,
        aliases,
        objective,
        display_budget,
        encoded[target] if target is not None else None,
    )
    width = min(max_dimensions, len(active))
    beam, evaluated = _beam_search(
        scorer, active, steering, width, budget["max_candidates"], budget["beam_width"]
    )
    alternatives = _alternatives(scorer, beam, active, aliases)
    context = {"scope": scope, "missing": missing or {}, "table_id": table_id}
    base["paths"] = []
    for path in [p for p in alternatives[:n_paths] if p]:
        _add_path(df, base, path, scorer, target, positions, context, display_budget)
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
        "distinct_alternatives": len(alternatives),
        "alternative_policy": "best_order_per_feature_set_collapsing_alias_substitutions",
        "search_exhausted_budget": evaluated >= budget["max_candidates"],
        "requested_depth": width,
        "returned_depth": max((len(p["dimensions"]) for p in base["paths"]), default=0),
    }
    base["parameters"] = {
        "objective": objective,
        "features": steering.requested,
        "start_with": steering.starts,
        "before": steering.constraints,
        "exclude": sorted(steering.excluded),
        "target": target,
        "max_dimensions": max_dimensions,
        "n_paths": n_paths,
        "limits": budget,
    }
    return Result("paths", base)


# Hand-tuned costs of the path score (heuristics, not fitted or probabilistic).
# The base cost is the prefix group counts relative to the display budget, plus
# the groups beyond it; these weights add penalties on top.
REDUNDANT_STEP = 2  # a dimension that adds no groups to its prefix
ALIAS_STEP = 3  # both columns of an equivalent pair in one path
NESTING_INVERSION = 8  # a finer column before a coarser one (structure, context)
SEPARATION = 12  # summed nonmodal fraction of the target or availability patterns


@dataclass(frozen=True)
class _Steering:
    requested: list[str]
    excluded: set[str]
    starts: list[str]
    constraints: list[tuple[str, str]]
    required: set[str]
    pool: list[str]


def _steering(df, features, exclude, start_with, before, target, objective) -> _Steering:
    """Validate the steering arguments and list the columns a path may use."""
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
    if objective == "context" and not starts:
        raise ValueError("context objective requires start_with")
    if objective == "target" and target is None:
        raise ValueError("target objective requires target")
    if target is not None:
        columns(df, [target])
    pool = [c for c in requested if c not in excluded and (c != target or c in required)]
    return _Steering(requested, excluded, starts, constraints, required, pool)


def _nesting(encoded, active, cardinality, max_pairs):
    """Observed coarse-to-fine nesting and equivalent (alias) column pairs."""
    edges, aliases, tested = set(), [], 0
    with phase("path nesting", min(math.comb(len(active), 2), max_pairs), "pairs") as tracker:
        for a, b in combinations(active, 2):
            if tested >= max_pairs:
                break
            tested += 1
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
    return edges, aliases, tested


def _check_constraints(steering: _Steering) -> None:
    """User precedence must be acyclic and agree with start_with."""
    constraints, starts = steering.constraints, steering.starts
    if any(a == b for a, b in constraints):
        raise ValueError("before constraints must be acyclic")
    pending, emitted = set(steering.required), set()
    while pending:
        ready = {c for c in pending if all(a in emitted for a, b in constraints if b == c)}
        if not ready:
            raise ValueError("before constraints contain a cycle")
        pending -= ready
        emitted |= ready
    for i, c in enumerate(starts):
        if any(b == c and a not in starts[:i] for a, b in constraints):
            raise ValueError("start_with conflicts with before constraints")


class _Scorer:
    """Heuristic costs of candidate paths, with cached prefix groupings."""

    CACHE_BYTES = 32 * 1024 * 1024

    def __init__(self, encoded, present, selected, edges, aliases, objective, budget, target):
        self.encoded, self.present, self.selected = encoded, present, selected
        self.edges, self.aliases, self.objective = edges, aliases, objective
        self.budget, self.target = max(1, budget), target
        self.prefixes: OrderedDict[tuple[str, ...], np.ndarray] = OrderedDict()
        self.cached_bytes = 0
        self.measured: dict[tuple[tuple[str, ...], bool], dict[str, Any]] = {}
        self._availability: np.ndarray | None = None

    def availability(self) -> np.ndarray:
        """Each row's availability pattern over the candidate columns."""
        if self._availability is None:
            rows = len(next(iter(self.present.values()))) if self.present else 0
            packed = np.zeros((rows, (len(self.selected) + 7) // 8), dtype=np.uint8)
            for j, c in enumerate(self.selected):
                packed[:, j // 8] |= self.present[c].astype(np.uint8) << (7 - j % 8)
            self._availability = (
                np.unique(packed, axis=0, return_inverse=True)[1]
                if self.selected
                else np.zeros(rows, dtype=np.int64)
            )
        return self._availability

    def prefix(self, path: tuple[str, ...]) -> np.ndarray:
        checkpoint()
        if path in self.prefixes:
            self.prefixes.move_to_end(path)
            return self.prefixes[path]
        ids = (
            group_ids([self.encoded[path[0]]])
            if len(path) == 1
            else pair_groups(self.prefix(path[:-1]), self.encoded[path[-1]])
        )
        if ids.nbytes <= self.CACHE_BYTES:
            self.prefixes[path] = ids
            self.cached_bytes += ids.nbytes
            while self.cached_bytes > self.CACHE_BYTES:
                _, removed = self.prefixes.popitem(last=False)
                self.cached_bytes -= removed.nbytes
        return ids

    def measure(self, path: tuple[str, ...], explain: bool = False) -> dict[str, Any]:
        key = (path, explain)
        if key not in self.measured:
            self.measured[key] = self._measure(path, explain)
        return self.measured[key]

    def _measure(self, path: tuple[str, ...], explain: bool) -> dict[str, Any]:
        prefixes, previous, redundancy = [], 1, 0
        target_losses, availability_losses = [], []
        for depth in range(1, len(path) + 1):
            keys = self.prefix(path[:depth])
            count = int(keys.max()) + 1 if len(keys) else 0
            prefixes.append(count)
            redundancy += count == previous
            previous = count
            if self.target is not None and (explain or self.objective == "target"):
                target_losses.append(_impurity(self.target, keys))
            if explain or self.objective == "availability":
                availability_losses.append(_impurity(self.availability(), keys))
        inversions = sum(
            a in path and b in path and path.index(a) > path.index(b) for a, b in self.edges
        )
        alias_steps = sum(a in path and b in path for a, b in self.aliases)
        prefix_cost = sum(prefixes) / self.budget
        overflow = sum(max(0, n - self.budget) for n in prefixes) / self.budget
        # Objective penalties use prefix behavior: early useful splits and nesting.
        score = prefix_cost + overflow + REDUNDANT_STEP * redundancy + ALIAS_STEP * alias_steps
        if self.objective in {"structure", "context"}:
            score += NESTING_INVERSION * inversions
        if self.objective == "target":
            score += SEPARATION * sum(target_losses)
        if self.objective == "availability":
            score += SEPARATION * sum(availability_losses)
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


def _impurity(labels: np.ndarray, groups: np.ndarray) -> float:
    """Share of rows not carrying their group's most common label."""
    if not len(labels):
        return 0.0
    _, sizes, _, maxima, _ = modal_groups(groups, labels)
    return int((sizes - maxima).sum()) / len(labels)


def _beam_search(scorer, active, steering, width, max_candidates, beam_width):
    """Extend paths depth by depth, keeping the best order of each column set."""
    constraints, required = steering.constraints, steering.required
    beam, evaluated = [tuple(steering.starts)], 0
    with phase("path search", max_candidates, "extensions budget") as tracker:
        for _ in range(len(steering.starts), width):
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
                    expanded.append((scorer.measure(next_path)["score"], next_path))
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
    return [p for p in beam if required <= set(p)], evaluated


def _alternatives(scorer, beam, active, aliases) -> list[tuple[str, ...]]:
    """Paths in score order, one per column set once equivalent columns are merged."""
    representative = {c: c for c in active}
    for a, b in aliases:
        old, new = representative[b], representative[a]
        representative = {c: new if r == old else r for c, r in representative.items()}
    diverse = {}
    for path in sorted(beam, key=lambda p: (scorer.measure(p)["score"], p)):
        diverse.setdefault(frozenset(representative[c] for c in path), path)
    return list(diverse.values())


def _add_path(df, base, path, scorer, target, positions, context, display_budget) -> None:
    metrics = scorer.measure(path, True)
    reasons, explanation = path_reasons(path, metrics, scorer.edges, scorer.aliases, target)
    preview = census(df, path, max_nodes=display_budget, max_levels=8, **context).to_dict()
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
