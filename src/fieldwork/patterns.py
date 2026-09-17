"""Bounded populated-value summaries and evidence-labelled column families."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations, islice

import numpy as np
import pandas as pd

from ._explore._kernels import group_ids, modal_groups
from ._runtime import checkpoint, operation, phase
from .evidence import bounded_rows, columns, finding, limit, prepare, result


@operation("value patterns")
def value_patterns(
    df,
    *,
    features=None,
    by=None,
    missing=None,
    scope=None,
    table_id="table",
    max_pairs=100,
    max_patterns=10,
    example_limit=5,
):
    for name, value in [
        ("max_pairs", max_pairs),
        ("max_patterns", max_patterns),
        ("example_limit", example_limit),
    ]:
        limit(name, value)
    selected = columns(df, features)
    contexts = columns(df, by or [])
    frame, positions, codes, present, base = prepare(
        df,
        missing=missing,
        scope=scope,
        table_id=table_id,
        features=[*selected, *contexts] if contexts else [],
        presence_features=selected,
    )
    families = defaultdict(list)
    base["summaries"] = []
    with phase("value summaries", len(selected), "columns") as progress:
        for c in selected:
            values = frame[c].iloc[np.flatnonzero(present[c])]
            record = {"feature": c, "populated": len(values), "missing": len(frame) - len(values)}
            examples = bounded_rows(positions, present[c], example_limit)
            if (
                len(values)
                and pd.api.types.infer_dtype(values.to_numpy(copy=False), skipna=True) == "string"
            ):
                ids, uniques = pd.factorize(values, sort=False)
                counts = np.bincount(ids)
                formats, lengths, prefixes = Counter(), Counter(), Counter()
                for i, (v, count) in enumerate(zip(uniques, counts)):
                    if i % 8192 == 0:
                        checkpoint()
                    formats[re.sub(r"[A-Za-z]+", "A", re.sub(r"\d+", "9", v))] += int(count)
                    lengths[len(v)] += int(count)
                    prefixes[v[:3]] += int(count)
                record.update(
                    formats=formats.most_common(max_patterns),
                    lengths=lengths.most_common(max_patterns),
                    prefixes=prefixes.most_common(max_patterns),
                    format_count=len(formats),
                    omitted_format_rows=sum(n for _, n in formats.most_common()[max_patterns:]),
                )
                finding(
                    base,
                    "string_patterns",
                    f"{c}: string formats, lengths and prefixes",
                    [c],
                    record,
                    examples,
                    example_limit=example_limit,
                )
            if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(
                values.dtype
            ):
                numeric = values.to_numpy(dtype=float)
                finite = np.sort(np.unique(numeric[np.isfinite(numeric)]))
                differences = np.diff(finite)
                step = float(differences.min()) if len(differences) else None
                quantized = (
                    bool(
                        np.allclose(
                            (finite - finite[0]) / step, np.round((finite - finite[0]) / step)
                        )
                    )
                    if step
                    else None
                )
                record.update(
                    minimum=float(finite[0]) if len(finite) else None,
                    maximum=float(finite[-1]) if len(finite) else None,
                    nonfinite=int((~np.isfinite(numeric)).sum()),
                    observed_step=step,
                    on_observed_step_grid=quantized,
                )
                finding(
                    base,
                    "numeric_range",
                    f"{c}: numeric range and observed spacing",
                    [c],
                    record,
                    examples,
                    example_limit=example_limit,
                )
            base["summaries"].append(record)
            match = re.match(r"^(.*?)[_\-]?\d+$", c)
            if match:
                families[match.group(1)].append(c)
            progress.advance(detail=c)
    base["families"] = []
    for prefix, group in families.items():
        if len(group) > 1:
            evidence = ["indexed_name"]
            if all(np.array_equal(present[group[0]], present[c]) for c in group[1:]):
                evidence.append("identical_availability")
            base["families"].append({"features": group, "prefix": prefix, "evidence": evidence})
            finding(
                base,
                "indexed_family",
                f"Indexed family: {', '.join(group)}",
                group,
                {"evidence": evidence},
                positions,
                example_limit=example_limit,
            )
    tested = 0
    for a, b in islice(combinations(selected, 2), max_pairs):
        checkpoint()
        tested += 1
        eligible = present[a] & present[b]
        if not eligible.any():
            continue
        x, y = frame[a].iloc[np.flatnonzero(eligible)], frame[b].iloc[np.flatnonzero(eligible)]
        if not (pd.api.types.is_numeric_dtype(x.dtype) and pd.api.types.is_numeric_dtype(y.dtype)):
            continue
        x, y = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) < 2:
            continue
        for name, values, mask in [
            ("offset", y - x, np.ones(len(x), dtype=bool)),
            ("ratio", np.divide(y, x, out=np.zeros_like(y), where=x != 0), x != 0),
        ]:
            values = values[mask]
            if len(values) >= 2 and np.all(np.isfinite(values)) and np.allclose(values, values[0]):
                finding(
                    base,
                    f"numeric_{name}",
                    f"{b} has a constant {name} relative to {a}",
                    [a, b],
                    {
                        "value": float(values[0]),
                        "evaluated_rows": len(values),
                        "excluded_rows": len(frame) - len(values),
                        "rtol": 1e-5,
                        "atol": 1e-8,
                    },
                    positions[eligible][finite][mask],
                    example_limit=example_limit,
                )
    if contexts:
        valid_rows = np.flatnonzero(np.logical_and.reduce([present[c] for c in contexts]))
        context_ids = group_ids(codes[c][valid_rows] for c in contexts)
        with phase("context constancy", len(selected), "columns") as progress:
            for c in selected:
                populated = present[c][valid_rows]
                valid_contexts = pd.unique(context_ids[populated])
                _, sizes, _, _, distinct = modal_groups(
                    context_ids[populated], codes[c][valid_rows[populated]]
                )
                constant = np.isin(context_ids, valid_contexts[distinct == 1])
                nonconstant = np.isin(context_ids, valid_contexts[distinct > 1])
                nconstant = int(np.count_nonzero(distinct == 1))
                finding(
                    base,
                    "context_constancy",
                    f"{c}: constancy within {', '.join(contexts)}",
                    [*contexts, c],
                    {
                        "evaluated_groups": len(sizes),
                        "constant_groups": nconstant,
                        "constant_group_fraction": nconstant / len(sizes) if len(sizes) else None,
                    },
                    bounded_rows(positions[valid_rows], constant, example_limit),
                    exceptions=bounded_rows(positions[valid_rows], nonconstant, example_limit),
                    example_limit=example_limit,
                )
                progress.advance(detail=c)
    base["coverage"] = {
        "pair_candidates": len(selected) * (len(selected) - 1) // 2,
        "pairs_evaluated": tested,
    }
    base["parameters"] = {
        "features": selected,
        "by": contexts,
        "max_pairs": max_pairs,
        "max_patterns": max_patterns,
        "example_limit": example_limit,
    }
    return result("value_patterns", base)
