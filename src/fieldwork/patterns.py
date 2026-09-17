"""Bounded populated-value summaries and evidence-labelled column families."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations, islice

import numpy as np
import pandas as pd

from .evidence import columns, finding, limit, prepare, result


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
        df, missing=missing, scope=scope, table_id=table_id
    )
    families = defaultdict(list)
    base["summaries"] = []
    for c in selected:
        values = frame[c].iloc[np.flatnonzero(present[c])]
        record = {"feature": c, "populated": len(values), "missing": len(frame) - len(values)}
        examples = positions[present[c]]
        if len(values) and all(isinstance(v, str) for v in values):
            formats = Counter(re.sub(r"[A-Za-z]+", "A", re.sub(r"\d+", "9", v)) for v in values)
            lengths = Counter(len(v) for v in values)
            prefixes = Counter(v[:3] for v in values)
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
                    np.allclose((finite - finite[0]) / step, np.round((finite - finite[0]) / step))
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
        # Evaluate exactly the declared context, without enumerating its subsets.
        grouped = defaultdict(list)
        for i in range(len(frame)):
            if all(present[c][i] for c in contexts):
                grouped[tuple(int(codes[c][i]) for c in contexts)].append(i)
        for c in selected:
            valid = [rows for rows in grouped.values() if any(present[c][i] for i in rows)]
            constant = [
                rows
                for rows in valid
                if len({int(codes[c][i]) for i in rows if present[c][i]}) == 1
            ]
            finding(
                base,
                "context_constancy",
                f"{c}: constancy within {', '.join(contexts)}",
                [*contexts, c],
                {
                    "evaluated_groups": len(valid),
                    "constant_groups": len(constant),
                    "constant_group_fraction": len(constant) / len(valid) if valid else None,
                },
                positions[sorted(i for rows in constant for i in rows)],
                exceptions=positions[
                    sorted(i for rows in valid if rows not in constant for i in rows)
                ],
                example_limit=example_limit,
            )
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
