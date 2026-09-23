"""Availability signatures, directional presence and entity-balanced evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations, islice
from math import comb
from typing import Any, Literal, Unpack

import numpy as np
import pandas as pd

from ._explore._kernels import first_indices, group_ids
from ._explore.encoding import cell
from ._runtime import checkpoint, operation, phase
from .evidence import (
    EvidenceRows,
    Scope,
    analyzable,
    columns,
    context_statement,
    finding,
    limit,
    prepare,
    result,
)
from .result import Result
from .typing import Runtime


class _Units:
    """Source-ordered rows and optional dense entity IDs, without lists per row."""

    def __init__(self, rows, ids=None):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.ids = ids
        self.sizes = np.bincount(ids) if ids is not None else None

    def __len__(self):
        return len(self.sizes) if self.ids is not None else len(self.rows)

    def counts(self, present):
        return np.bincount(self.ids, weights=present[self.rows], minlength=len(self)).astype(
            np.int64
        )

    def evidence(self, mask, positions, limit):
        matches = mask[self.ids] if self.ids is not None else mask
        return EvidenceRows(positions[self.rows[first_indices(matches, limit)]], int(matches.sum()))


@dataclass
class _Analysis:
    """One availability analysis: prepared rows, units and the payload being built."""

    frame: pd.DataFrame
    positions: np.ndarray
    present: dict[str, np.ndarray]
    codes: dict[str, np.ndarray]
    base: dict[str, Any]
    selected: list[str]
    contexts: list[str]
    entities: list[str]
    unit: str
    entity_presence: str
    example_limit: int
    eligible: np.ndarray = field(init=False)
    entity_ids: np.ndarray = field(init=False)

    def __post_init__(self):
        self.eligible = np.ones(len(self.frame), dtype=bool)
        for c in self.entities:
            self.eligible &= self.present[c]
        self.entity_ids = group_ids(self.codes[c] for c in self.entities)

    def units(self, rows, entity_mode: bool | None = None) -> _Units:
        """Rows, or the entities they belong to (rows with incomplete keys excluded)."""
        rows = np.asarray(rows, dtype=np.int64)
        if self.unit == "entities" if entity_mode is None else entity_mode:
            rows = rows[self.eligible[rows]]
            return _Units(rows, group_ids([self.entity_ids[rows]]))
        return _Units(rows)

    def masks(self, units: _Units) -> dict[str, np.ndarray]:
        """Presence per unit and feature; entities aggregate their rows."""
        masks = {}
        with phase("availability masks", len(self.selected), "columns") as tracker:
            for c in self.selected:
                if units.ids is None:
                    whole = len(units.rows) == len(self.frame)
                    masks[c] = self.present[c] if whole else self.present[c][units.rows]
                else:
                    counts = units.counts(self.present[c])
                    any_row = self.entity_presence == "any"
                    masks[c] = counts > 0 if any_row else counts == units.sizes
                tracker.advance(detail=c)
        return masks

    def emit(self, units, kind, statement, features, metrics, mask, *, exceptions=None, **extra):
        """Record a finding whose examples and exceptions are the units' source rows."""
        unit = extra.pop("unit", self.unit)
        return finding(
            self.base,
            kind,
            statement,
            features,
            metrics,
            units.evidence(mask, self.positions, self.example_limit),
            exceptions=units.evidence(exceptions, self.positions, self.example_limit)
            if exceptions is not None
            else (),
            example_limit=self.example_limit,
            unit=unit,
            **extra,
        )


@operation("missingness")
def missingness(
    df: pd.DataFrame,
    *,
    features: Iterable[str] | None = None,
    by: Iterable[str] | None = None,
    entity: str | Iterable[str] | None = None,
    unit: Literal["rows", "entities"] = "rows",
    entity_presence: Literal["any", "all"] = "any",
    missing: Mapping[str, Iterable[Any]] | None = None,
    scope: Scope | None = None,
    table_id: str = "table",
    min_implication: float = 0.9,
    min_similarity: float = 0.8,
    max_pairs: int = 200,
    max_signatures: int = 50,
    max_contexts: int = 32,
    example_limit: int = 5,
    **runtime: Unpack[Runtime],
) -> Result:
    """Measure where values are present: per column, jointly, and within contexts.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    features : iterable of str or None, optional
        Columns to analyze; default None selects every column, skipping (and
        listing in ``skipped_features``) columns with unsupported values.
    by : iterable of str or None, optional
        Context columns; each joint context value gets its own availability.
    entity : str, iterable of str, or None, optional
        Entity key columns; adds per-entity summaries. Rows with an incomplete key
        are excluded from entity counts and counted.
    unit : {'rows', 'entities'}, optional
        Count rows (default) or entities (requires ``entity``).
    entity_presence : {'any', 'all'}, optional
        With unit='entities': populated when any row (default) or every row is.
    missing, scope, table_id
        Source context shared by every analysis.
    min_implication, min_similarity : float, optional
        Report "A populated implies B populated" at or above min_implication
        (default 0.9, denominator: units with A), and similar presence at or above
        min_similarity Jaccard (default 0.8, denominator: units with A or B).
    max_pairs, max_signatures, max_contexts : int, optional
        Budgets: column pairs tested (default 200), availability patterns saved
        (default 50), and contexts analyzed (default 32). Omissions are counted.
    example_limit : int, optional
        Saved example and exception source rows per finding; default 5. Selection
        always recovers the complete population.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'missingness': ``availability`` (every column), ``signatures``,
        ``families``, ``contexts``, ``entities``, ``analysis_unit``, ``coverage``
        and findings. Vacuous evidence (always-present targets, identical
        columns) is not reported as findings.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"id": [1, 1, 2], "value": [10, None, 20]})
    >>> result = fw.missingness(df, features=["value"], entity="id",
    ...                         unit="entities", entity_presence="all")
    >>> result["availability"][0]["populated_fraction"]
    0.5
    """
    for name, value in [
        ("max_pairs", max_pairs),
        ("max_signatures", max_signatures),
        ("max_contexts", max_contexts),
        ("example_limit", example_limit),
    ]:
        limit(name, value)
    if not 0 <= min_implication <= 1 or not 0 <= min_similarity <= 1:
        raise ValueError("Thresholds must be between zero and one")
    if unit not in {"rows", "entities"} or entity_presence not in {"any", "all"}:
        raise ValueError("unit must be rows or entities; entity_presence must be any or all")
    selected = columns(df, features)
    contexts = columns(df, by or [])
    entities = columns(df, [entity] if isinstance(entity, str) else entity or [])
    if unit == "entities" and not entities:
        raise ValueError("unit='entities' requires entity keys")
    if unit == "rows" and entity_presence != "any":
        raise ValueError("entity_presence applies only to unit='entities'")
    frame, positions, codes, present, base = prepare(
        df,
        scope=scope,
        missing=missing,
        table_id=table_id,
        features=[*contexts, *entities],
        presence_features=selected,
        optional=selected if features is None else (),
    )
    analysis = _Analysis(
        frame,
        positions,
        present,
        {c: np.where(present[c], values, -1) for c, values in codes.items()},
        base,
        analyzable(selected, base),
        contexts,
        entities,
        unit,
        entity_presence,
        example_limit,
    )
    units = analysis.units(np.arange(len(frame)))
    masks = analysis.masks(units)
    base["parameters"] = {
        "features": analysis.selected,
        "by": contexts,
        "entity": entities,
        "unit": unit,
        "entity_presence": entity_presence,
        "min_implication": min_implication,
        "min_similarity": min_similarity,
        "max_pairs": max_pairs,
        "max_signatures": max_signatures,
        "max_contexts": max_contexts,
        "example_limit": example_limit,
    }
    base["analysis_unit"] = _analysis_unit(analysis, units)
    base["availability"] = _availability(analysis, units, masks)
    signature_ids, shown, total = _signatures(analysis, units, masks, max_signatures)
    base["families"] = _families(analysis, units, masks)
    evaluated = _pairs(analysis, units, masks, max_pairs, min_similarity, min_implication)
    context_count = _contexts(analysis, max_contexts)
    base["entities"] = _entities(analysis) if entities else []
    omitted = ~np.isin(signature_ids, shown)
    base["coverage"] = {
        "pair_candidates": comb(len(analysis.selected), 2),
        "pairs_evaluated": evaluated,
        "signatures_total": total,
        "signatures_shown": len(shown),
        "signature_omitted_units": int(omitted.sum()),
        "signature_omitted_rows": len(units.evidence(omitted, positions, 0)),
        "contexts_total": context_count,
        "contexts_shown": min(context_count, max_contexts),
        "entity_missing_key_rows": int((~analysis.eligible).sum()) if entities else 0,
    }
    return result("missingness", base)


def _analysis_unit(analysis: _Analysis, units: _Units) -> dict[str, Any]:
    entities_counted = analysis.unit == "entities"
    return {
        "counting_unit": analysis.unit,
        "denominator": len(units),
        "entity_keys": analysis.entities,
        "presence_aggregation": analysis.entity_presence if entities_counted else "per_row",
        "source_rows": len(units.rows),
        "missing_key_excluded_rows": int((~analysis.eligible).sum()) if entities_counted else 0,
        "context_aggregation": "within_each_context",
    }


def _availability(analysis, units, masks) -> list[dict[str, Any]]:
    """Every column's populated count; partially populated columns are findings."""
    n, records = len(units), []
    for c in analysis.selected:
        mask = masks[c]
        metrics = {
            "populated": int(mask.sum()),
            "missing": int((~mask).sum()),
            "denominator": n,
            "populated_fraction": float(mask.mean()) if n else None,
        }
        records.append({"feature": c, **metrics})
        if not mask.all():  # A complete column's finding would say nothing.
            analysis.emit(
                units,
                "availability",
                f"{c}: populated values",
                [c],
                metrics,
                mask,
                exceptions=~mask,
                selector={"operation": "presence", "feature": c},
            )
    return records


def _signatures(analysis, units, masks, max_signatures) -> tuple[np.ndarray, np.ndarray, int]:
    """The most common combinations of populated columns, as findings.

    Returns each unit's signature, the signatures shown, and how many exist.
    """
    selected, n = analysis.selected, len(units)
    with phase("packing availability", len(selected), "columns") as tracker:
        packed = np.zeros((n, (len(selected) + 7) // 8), dtype=np.uint8)
        for j, c in enumerate(selected):
            packed[:, j // 8] |= masks[c].astype(np.uint8) << (7 - j % 8)
            tracker.advance(detail=c)
    with phase("availability signatures"):
        if selected:
            signatures, ids, counts = np.unique(
                packed, axis=0, return_inverse=True, return_counts=True
            )
        else:
            signatures = np.empty((int(n > 0), 0), dtype=np.uint8)
            ids = np.zeros(n, dtype=np.int64)
            counts = np.array([n], dtype=np.int64) if n else np.empty(0, dtype=np.int64)
        ordered = np.lexsort((np.arange(len(counts)), -counts))
    analysis.base["signatures"] = []
    for group in ordered[:max_signatures]:
        signature = np.unpackbits(signatures[group], count=len(selected)).astype(bool)
        present = [c for c, value in zip(selected, signature) if value]
        absent = [c for c, value in zip(selected, signature) if not value]
        found = analysis.emit(
            units,
            "availability_signature",
            f"Present: {', '.join(present) or 'none'}; absent: {', '.join(absent) or 'none'}",
            selected,
            {"count": int(counts[group]), "denominator": n},
            ids == group,
            structure={"present": present, "absent": absent},
            selector={"operation": "availability_signature", "present": present, "absent": absent},
        )
        analysis.base["signatures"].append(
            {
                "present": present,
                "absent": absent,
                "count": int(counts[group]),
                "denominator": n,
                "finding_id": found["id"],
            }
        )
    return ids, ordered[:max_signatures], len(signatures)


def _families(analysis, units, masks) -> list[dict[str, Any]]:
    """Groups of partially populated columns with identical availability."""
    groups = defaultdict(list)
    for c in analysis.selected:
        groups[np.packbits(masks[c]).tobytes()].append(c)
    families = []
    for group in groups.values():
        mask = masks[group[0]]
        if len(group) < 2 or mask.all():
            continue
        found = analysis.emit(
            units,
            "availability_family",
            "Same availability: " + ", ".join(group),
            group,
            {"denominator": len(units), "populated": int(mask.sum())},
            mask,
        )
        families.append(
            {
                "features": group,
                "evidence": ["identical_availability"],
                "populated": int(mask.sum()),
                "finding_id": found["id"],
            }
        )
    return families


def _pairs(analysis, units, masks, max_pairs, min_similarity, min_implication) -> int:
    """Similar, mutually exclusive and implied presence, in column-pair order."""
    selected, n, evaluated = analysis.selected, len(units), 0
    with phase("availability pairs", min(comb(len(selected), 2), max_pairs), "pairs") as tracker:
        for a, b in islice(combinations(selected, 2), max_pairs):
            evaluated += 1
            x, y = masks[a], masks[b]
            both, union = int((x & y).sum()), int((x | y).sum())
            metrics = {
                "both_present": both,
                "either_present": union,
                "both_absent": int((~x & ~y).sum()),
                "denominator": n,
                "presence_jaccard": both / union if union else None,
                "agreement": float((x == y).mean()) if n else None,
            }
            _pair_findings(analysis, units, (a, b), (x, y), metrics, min_similarity)
            for names, pair in [((a, b), (x, y)), ((b, a), (y, x))]:
                # Skip vacuous rules: an always-present target, or identical columns
                # (their family already reports them).
                if not (pair[1].all() or np.array_equal(x, y)):
                    _implication(analysis, units, names, pair, metrics, min_implication)
            tracker.advance(detail=f"{a} / {b}")
    return evaluated


def _pair_findings(analysis, units, names, masks, metrics, min_similarity) -> None:
    (a, b), (x, y) = names, masks
    similarity = metrics["presence_jaccard"]
    if (
        similarity is not None
        and similarity >= min_similarity
        and not np.array_equal(x, y)
        and not (x.all() or y.all())
    ):
        analysis.emit(
            units,
            "similar_availability",
            f"{a} and {b} have similar presence",
            [a, b],
            metrics,
            x & y,
            exceptions=x != y,
        )
    if metrics["both_present"] == 0 and x.any() and y.any():
        statement = f"{a} and {b} are mutually exclusive"
        analysis.emit(units, "mutually_exclusive", statement, [a, b], metrics, x | y)


def _implication(analysis, units, names, masks, metrics, min_implication) -> None:
    (source, target), (first, second) = names, masks
    denominator = int(first.sum())
    rate = metrics["both_present"] / denominator if denominator else None
    if rate is None or rate < min_implication:
        return
    analysis.emit(
        units,
        "presence_implication",
        f"{source} populated implies {target} populated",
        [source, target],
        {
            **metrics,
            "antecedent_populated": denominator,
            "conditional_presence": rate,
            "exception_rate": 1 - rate,
            "baseline_presence": float(second.mean()) if len(units) else None,
        },
        first & second,
        exceptions=first & ~second,
        selector={"operation": "presence_implication", "source": source, "target": target},
    )


def _contexts(analysis, max_contexts) -> int:
    """Availability within each joint context value; returns how many contexts exist."""
    contexts, frame = analysis.contexts, analysis.frame
    context_ids = group_ids(analysis.codes[c] for c in contexts)
    count = int(context_ids.max()) + 1 if len(context_ids) else 0
    analysis.base["contexts"] = []
    for context_id in range(min(count, max_contexts)):
        checkpoint()
        rows = np.flatnonzero(context_ids == context_id)
        values = {c: cell(frame, c, rows[0], analysis.present[c][rows[0]]) for c in contexts}
        units = analysis.units(rows)
        masks = analysis.masks(units)
        record = {"values": values, "rows": len(rows), "availability": [], "finding_ids": []}
        for c in analysis.selected:
            metrics = {"feature": c, "populated": int(masks[c].sum()), "denominator": len(units)}
            record["availability"].append(metrics)
            found = analysis.emit(
                units,
                "context_availability",
                f"{c}: availability within {context_statement(values)}",
                list(dict.fromkeys([*contexts, c])),
                metrics,
                masks[c],
                exceptions=~masks[c],
                structure={"context": values},
                selector={"operation": "context_availability", "context": values, "feature": c},
            )
            record["finding_ids"].append(found["id"])
        summary = analysis.emit(
            units,
            "context_summary",
            "Availability within " + context_statement(values),
            list(dict.fromkeys([*contexts, *analysis.selected])),
            {
                "source_rows": len(rows),
                "denominator": len(units),
                "availability": record["availability"],
            },
            np.ones(len(units), dtype=bool),
            structure={"context": values},
            selector={"operation": "context_summary", "context": values},
        )
        record["finding_id"] = summary["id"]
        analysis.base["contexts"].append(record)
    return count


def _entities(analysis) -> list[dict[str, Any]]:
    """How many entities have a feature on any, all, one, some or none of their rows."""
    units = analysis.units(np.arange(len(analysis.frame)), True)
    keys, summaries = analysis.entities, []
    for c in analysis.selected:
        checkpoint()
        counts = units.counts(analysis.present[c])
        matches = {
            "any": counts > 0,
            "all": counts == units.sizes,
            "one": counts == 1,
            "some": (counts > 0) & (counts < units.sizes),
            "none": counts == 0,
        }
        summary: dict[str, Any] = {"feature": c, "denominator": len(counts)}
        features = list(dict.fromkeys([*keys, c]))
        for pattern, matching in matches.items():
            summary[pattern] = int(matching.sum())
            analysis.emit(
                units,
                "entity_availability",
                f"{c}: {pattern} populated rows per entity ({', '.join(keys)})",
                features,
                {"entities": summary[pattern], "denominator": len(counts)},
                matching,
                unit="entities",
                structure={"entity_keys": keys, "presence_pattern": pattern},
                selector={
                    "operation": "entity_availability",
                    "feature": c,
                    "presence_pattern": pattern,
                },
            )
        analysis.emit(
            units,
            "entity_summary",
            f"{c}: presence across entities ({', '.join(keys)})",
            features,
            summary,
            np.ones(len(units), dtype=bool),
            unit="entities",
            structure={"entity_keys": keys},
            selector={"operation": "entity_summary", "feature": c},
        )
        summaries.append(summary)
    return summaries
