"""Availability signatures, directional presence and entity-balanced evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
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
    InvestigationResult,
    Scope,
    analyzable,
    columns,
    context_statement,
    finding,
    limit,
    prepare,
    result,
)
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
) -> InvestigationResult:
    """Measure availability, co-presence, and patterns of absence.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation. Discovery requires unique string
        column names. Duplicate index labels are supported; source selections use
        integer row positions. Unsupported scalar objects raise TypeError.
    features : iterable of str or None, optional
        Unique column names to analyze, in requested order; default None selects
        all columns, skipping those with unsupported values (such as lists,
        dicts or Decimal) and listing them in skipped_features. Restricts
        analysis, not full-source identity validation.
    by : iterable of str or None, optional
        Joint context columns; default None. Missing context values form categories.
        An entity spanning contexts contributes once within each relevant context.
    entity : str, iterable of str, or None, optional
        Single or composite entity key; default None. Entity analyses exclude and
        count rows with incomplete keys. Supplying entity alone keeps unit="rows"
        and adds entity summaries.
    unit : {'rows', 'entities'}, optional
        Default 'rows' weights each row equally. 'entities' requires entity keys
        and weights each distinct populated key equally.
    entity_presence : {'any', 'all'}, optional
        Default 'any' counts an entity as populated if any of its rows is populated.
        'all' requires all its rows and is valid only with unit='entities'.
    missing : mapping or None, optional
        Additional missing sentinels per column; default None. Native missing
        values are always absent. Numeric sentinels match integer/float values
        numerically; booleans remain distinct. The source is not modified.
    scope : Scope or None, optional
        Source-bound population selection; default None uses all rows. The scope
        must match the ordered source. Fingerprinting still scans the full frame.
    table_id : str, optional
        Nonempty source label; default 'table'. Does not replace the fingerprint.
    min_implication : float, optional
        Minimum directional presence fraction in [0, 1]; default 0.9. Its
        denominator is units with the antecedent feature populated.
    min_similarity : float, optional
        Minimum Jaccard presence similarity in [0, 1]; default 0.8. Its denominator
        is units with either feature populated.
    max_pairs : int, optional
        Nonnegative pair-search budget in requested column order; default 200.
        Zero skips pair tests while retaining single-feature summaries.
    max_signatures : int, optional
        Nonnegative number of saved availability signatures; default 50. Omitted
        units and source rows are counted; zero saves no signatures.
    max_contexts : int, optional
        Nonnegative number of joint context groups to analyze; default 32.
        Zero skips context summaries while retaining global evidence.
    example_limit : int, optional
        Nonnegative maximum saved example/exception source rows per finding side;
        default 5. Zero retains totals without row examples. This display limit
        does not restrict the population recovered by select or all_matches.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    InvestigationResult
        Kind 'missingness', with availability, signatures, families, contexts,
        entity summaries, findings, analysis_unit, and search coverage. Fractions
        name their denominators; empty denominators have undefined fractions.

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
    labels, index labels, and all cell values (not dtype metadata); changing or
    reordering them invalidates inspection against saved findings.

    Examples and exceptions count source rows even when metrics count entities.
    Selecting an entity finding returns all rows of matching entities inside the
    scope/context, including rows with absent features. Entity summaries' 'all'
    and 'one' overlap for singletons; 'some' means at least one but fewer than all.

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
    selected = analyzable(selected, base)
    codes = {c: np.where(present[c], values, -1) for c, values in codes.items()}
    eligible = np.ones(len(frame), dtype=bool)
    for c in entities:
        eligible &= present[c]
    entity_ids = group_ids(codes[c] for c in entities)

    def units_for(rows, entity_mode=None):
        rows = np.asarray(rows, dtype=np.int64)
        use_entities = unit == "entities" if entity_mode is None else entity_mode
        if use_entities:
            rows = rows[eligible[rows]]
            return _Units(rows, group_ids([entity_ids[rows]]))
        return _Units(rows)

    def masks_for(units):
        masks = {}
        with phase("availability masks", len(selected), "columns") as tracker:
            for c in selected:
                if units.ids is None:
                    masks[c] = (
                        present[c] if len(units.rows) == len(frame) else present[c][units.rows]
                    )
                else:
                    counts = units.counts(present[c])
                    masks[c] = counts > 0 if entity_presence == "any" else counts == units.sizes
                tracker.advance(detail=c)
        return masks

    units = units_for(np.arange(len(frame)))
    entity_units = units_for(np.arange(len(frame)), True) if entities else None
    masks = masks_for(units)
    n = len(units)

    def emit(
        kind,
        statement,
        features,
        metrics,
        mask,
        *,
        exception_mask=None,
        structure=None,
        selector=None,
    ):
        return finding(
            base,
            kind,
            statement,
            features,
            metrics,
            units.evidence(mask, positions, example_limit),
            exceptions=units.evidence(exception_mask, positions, example_limit)
            if exception_mask is not None
            else (),
            example_limit=example_limit,
            unit=unit,
            structure=structure,
            selector=selector,
        )

    base["parameters"] = {
        "features": selected,
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
    base["analysis_unit"] = {
        "counting_unit": unit,
        "denominator": n,
        "entity_keys": entities,
        "presence_aggregation": entity_presence if unit == "entities" else "per_row",
        "source_rows": len(units.rows),
        "missing_key_excluded_rows": int((~eligible).sum()) if unit == "entities" else 0,
        "context_aggregation": "within_each_context",
    }
    base["availability"] = []
    for c in selected:
        mask = masks[c]
        metrics = {
            "populated": int(mask.sum()),
            "missing": int((~mask).sum()),
            "denominator": n,
            "populated_fraction": float(mask.mean()) if n else None,
        }
        base["availability"].append({"feature": c, **metrics})
        if mask.all():
            # Complete columns stay in the table; a finding would say nothing.
            continue
        emit(
            "availability",
            f"{c}: populated values",
            [c],
            metrics,
            mask,
            exception_mask=~mask,
            selector={"operation": "presence", "feature": c},
        )
    with phase("packing availability", len(selected), "columns") as tracker:
        packed = np.zeros((n, (len(selected) + 7) // 8), dtype=np.uint8)
        for j, c in enumerate(selected):
            packed[:, j // 8] |= masks[c].astype(np.uint8) << (7 - j % 8)
            tracker.advance(detail=c)
    with phase("availability signatures"):
        if selected:
            signatures, signature_ids, counts = np.unique(
                packed, axis=0, return_inverse=True, return_counts=True
            )
        else:
            signatures = np.empty((int(n > 0), 0), dtype=np.uint8)
            signature_ids = np.zeros(n, dtype=np.int64)
            counts = np.array([n], dtype=np.int64) if n else np.empty(0, dtype=np.int64)
        ordered = np.lexsort((np.arange(len(counts)), -counts))
    base["signatures"] = []
    for group in ordered[:max_signatures]:
        signature = np.unpackbits(signatures[group], count=len(selected)).astype(bool)
        populated = [c for c, value in zip(selected, signature) if value]
        absent = [c for c, value in zip(selected, signature) if not value]
        record = {
            "present": populated,
            "absent": absent,
            "count": int(counts[group]),
            "denominator": n,
        }
        f = emit(
            "availability_signature",
            "Present: "
            + (", ".join(populated) or "none")
            + "; absent: "
            + (", ".join(absent) or "none"),
            selected,
            {"count": int(counts[group]), "denominator": n},
            signature_ids == group,
            structure={"present": populated, "absent": absent},
            selector={
                "operation": "availability_signature",
                "present": populated,
                "absent": absent,
            },
        )
        base["signatures"].append({**record, "finding_id": f["id"]})
    groups = defaultdict(list)
    for c in selected:
        groups[np.packbits(masks[c]).tobytes()].append(c)
    base["families"] = []
    for group in groups.values():
        if len(group) > 1 and not masks[group[0]].all():
            f = emit(
                "availability_family",
                "Same availability: " + ", ".join(group),
                group,
                {"denominator": n, "populated": int(masks[group[0]].sum())},
                masks[group[0]],
            )
            base["families"].append(
                {
                    "features": group,
                    "evidence": ["identical_availability"],
                    "populated": int(masks[group[0]].sum()),
                    "finding_id": f["id"],
                }
            )
    evaluated = 0
    with phase("availability pairs", min(comb(len(selected), 2), max_pairs), "pairs") as tracker:
        for a, b in islice(combinations(selected, 2), max_pairs):
            evaluated += 1
            x, y = masks[a], masks[b]
            both, union = int((x & y).sum()), int((x | y).sum())
            similarity = both / union if union else None
            metrics = {
                "both_present": both,
                "either_present": union,
                "both_absent": int((~x & ~y).sum()),
                "denominator": n,
                "presence_jaccard": similarity,
                "agreement": float((x == y).mean()) if n else None,
            }
            if (
                similarity is not None
                and similarity >= min_similarity
                and not np.array_equal(x, y)
                and not (x.all() or y.all())
            ):
                emit(
                    "similar_availability",
                    f"{a} and {b} have similar presence",
                    [a, b],
                    metrics,
                    x & y,
                    exception_mask=x != y,
                )
            if both == 0 and x.any() and y.any():
                emit(
                    "mutually_exclusive",
                    f"{a} and {b} are mutually exclusive",
                    [a, b],
                    metrics,
                    x | y,
                )
            for source, target, first, second in [(a, b, x, y), (b, a, y, x)]:
                if second.all() or np.array_equal(x, y):
                    # Vacuous: the target is always present, or a family covers it.
                    continue
                denominator = int(first.sum())
                rate = both / denominator if denominator else None
                if rate is not None and rate >= min_implication:
                    emit(
                        "presence_implication",
                        f"{source} populated implies {target} populated",
                        [source, target],
                        {
                            **metrics,
                            "antecedent_populated": denominator,
                            "conditional_presence": rate,
                            "exception_rate": 1 - rate,
                            "baseline_presence": float(second.mean()) if n else None,
                        },
                        first & second,
                        exception_mask=first & ~second,
                        selector={
                            "operation": "presence_implication",
                            "source": source,
                            "target": target,
                        },
                    )
            tracker.advance(detail=f"{a} / {b}")
    context_ids = group_ids(codes[c] for c in contexts)
    context_count = int(context_ids.max()) + 1 if len(context_ids) else 0
    base["contexts"] = []
    for context_id in range(min(context_count, max_contexts)):
        checkpoint()
        rows = np.flatnonzero(context_ids == context_id)
        values = {c: cell(frame, c, rows[0], present[c][rows[0]]) for c in contexts}
        context_units = units_for(rows)
        context_masks = masks_for(context_units)
        record = {"values": values, "rows": len(rows), "availability": [], "finding_ids": []}
        for c in selected:
            mask = context_masks[c]
            metrics = {
                "feature": c,
                "populated": int(mask.sum()),
                "denominator": len(context_units),
            }
            record["availability"].append(metrics)
            f = finding(
                base,
                "context_availability",
                f"{c}: availability within {context_statement(values)}",
                list(dict.fromkeys([*contexts, c])),
                metrics,
                context_units.evidence(mask, positions, example_limit),
                exceptions=context_units.evidence(~mask, positions, example_limit),
                example_limit=example_limit,
                unit=unit,
                structure={"context": values},
                selector={"operation": "context_availability", "context": values, "feature": c},
            )
            record["finding_ids"].append(f["id"])
        summary = finding(
            base,
            "context_summary",
            "Availability within " + context_statement(values),
            list(dict.fromkeys([*contexts, *selected])),
            {
                "source_rows": len(rows),
                "denominator": len(context_units),
                "availability": record["availability"],
            },
            context_units.evidence(
                np.ones(len(context_units), dtype=bool), positions, example_limit
            ),
            example_limit=example_limit,
            unit=unit,
            structure={"context": values},
            selector={"operation": "context_summary", "context": values},
        )
        record["finding_id"] = summary["id"]
        base["contexts"].append(record)
    base["entities"] = []
    if entities:
        for c in selected:
            checkpoint()
            counts = entity_units.counts(present[c])
            predicates = {
                "any": counts > 0,
                "all": counts == entity_units.sizes,
                "one": counts == 1,
                "some": (counts > 0) & (counts < entity_units.sizes),
                "none": counts == 0,
            }
            summary = {"feature": c, "denominator": len(counts)}
            for pattern, matching in predicates.items():
                summary[pattern] = int(matching.sum())
                finding(
                    base,
                    "entity_availability",
                    f"{c}: {pattern} populated rows per entity ({', '.join(entities)})",
                    list(dict.fromkeys([*entities, c])),
                    {"entities": int(matching.sum()), "denominator": len(counts)},
                    entity_units.evidence(matching, positions, example_limit),
                    example_limit=example_limit,
                    unit="entities",
                    structure={"entity_keys": entities, "presence_pattern": pattern},
                    selector={
                        "operation": "entity_availability",
                        "feature": c,
                        "presence_pattern": pattern,
                    },
                )
            finding(
                base,
                "entity_summary",
                f"{c}: presence across entities ({', '.join(entities)})",
                list(dict.fromkeys([*entities, c])),
                summary,
                entity_units.evidence(
                    np.ones(len(entity_units), dtype=bool), positions, example_limit
                ),
                example_limit=example_limit,
                unit="entities",
                structure={"entity_keys": entities},
                selector={"operation": "entity_summary", "feature": c},
            )
            base["entities"].append(summary)
    omitted = ~np.isin(signature_ids, ordered[:max_signatures])
    base["coverage"] = {
        "pair_candidates": comb(len(selected), 2),
        "pairs_evaluated": evaluated,
        "signatures_total": len(signatures),
        "signatures_shown": min(len(signatures), max_signatures),
        "signature_omitted_units": int(omitted.sum()),
        "signature_omitted_rows": len(units.evidence(omitted, positions, 0)),
        "contexts_total": context_count,
        "contexts_shown": min(context_count, max_contexts),
        "entity_missing_key_rows": int((~eligible).sum()) if entities else 0,
    }
    return result("missingness", base)
