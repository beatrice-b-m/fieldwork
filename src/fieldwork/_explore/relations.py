"""Sparse pair relationships, association, and bounded absence summaries."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from ._kernels import exact_pair_ids
from .census import _scope, _source
from .encoding import (
    MISSING,
    ScalarIdentity,
    encode_series,
    normalize_scalar,
    resolve_columns,
    validate_limit,
)
from .result import ExplorerResult


def _relation(pair_counts: Counter[tuple[ScalarIdentity, ScalarIdentity]]) -> str | None:
    if not pair_counts:
        return None
    a_to_b: dict[ScalarIdentity, set[ScalarIdentity]] = defaultdict(set)
    b_to_a: dict[ScalarIdentity, set[ScalarIdentity]] = defaultdict(set)
    for (a, b), count in pair_counts.items():
        if count:
            a_to_b[a].add(b)
            b_to_a[b].add(a)
    a_many = any(len(values) > 1 for values in a_to_b.values())
    b_many = any(len(values) > 1 for values in b_to_a.values())
    if not a_many and not b_many:
        return "1:1"
    if a_many and not b_many:
        return "1:n"
    if not a_many and b_many:
        return "n:1"
    return "n:m"


def _cramers_v(
    pairs: Counter[tuple[ScalarIdentity, ScalarIdentity]],
    a_support: Counter[ScalarIdentity],
    b_support: Counter[ScalarIdentity],
) -> tuple[float | None, str | None]:
    total = sum(pairs.values())
    positive_a = [value for value, count in a_support.items() if count]
    positive_b = [value for value, count in b_support.items() if count]
    if total == 0:
        return None, "empty_population"
    denominator = min(len(positive_a) - 1, len(positive_b) - 1)
    if denominator <= 0:
        return None, "constant_dimension"
    chi_term = sum(count * count / (a_support[a] * b_support[b]) for (a, b), count in pairs.items())
    chi2 = max(0.0, total * chi_term - total)
    return math.sqrt(chi2 / (total * denominator)), None


def _declared_domain(
    reference_domains: Mapping[Any, Iterable[Any]] | None,
    column: Any,
    observed: set[ScalarIdentity],
) -> tuple[list[ScalarIdentity], str]:
    if reference_domains is None or column not in reference_domains:
        return sorted(observed, key=ScalarIdentity.sort_key), "empirical_observed"
    normalized = {normalize_scalar(value) for value in reference_domains[column]}
    missing = observed - normalized
    if missing:
        raise ValueError(
            f"Declared reference domain for {column!r} omits observed levels: "
            f"{[value.to_dict() for value in sorted(missing, key=ScalarIdentity.sort_key)]}"
        )
    return sorted(normalized, key=ScalarIdentity.sort_key), "caller_declared"


def pairs(
    df: pd.DataFrame,
    dimensions: Iterable[Any],
    *,
    dropna: bool = False,
    include_absence: bool = False,
    reference_domains: Mapping[Any, Iterable[Any]] | None = None,
    pair_contexts: Iterable[Mapping[Any, Any]] | None = None,
    max_absence_cells: int | None = 1000,
    max_contexts: int | None = 32,
    max_pairs: int | None = 15,
    scope_metadata: dict[str, Any] | None = None,
) -> ExplorerResult:
    selected = resolve_columns(df, dimensions, argument="dimensions")
    validate_limit("max_absence_cells", max_absence_cells)
    validate_limit("max_contexts", max_contexts)
    validate_limit("max_pairs", max_pairs)
    requested_pairs = list(combinations(range(len(selected)), 2))
    processed_pairs = requested_pairs[:max_pairs] if max_pairs is not None else requested_pairs
    requested_context_values = list(pair_contexts or [])
    requested_context_values.sort(
        key=lambda context: tuple(
            sorted(
                (
                    normalize_scalar(column, label=True).sort_key(),
                    normalize_scalar(value).sort_key(),
                )
                for column, value in context.items()
            )
        )
    )
    contexts: list[Mapping[Any, Any]] = [{}]
    contexts.extend(requested_context_values)
    if max_contexts is not None:
        contexts = contexts[:max_contexts]
    context_columns = tuple(dict.fromkeys(column for context in contexts for column in context))
    if context_columns:
        context_columns = resolve_columns(df, context_columns, argument="pair_contexts")
    encoded = {
        column: encode_series(df[column]) for column in dict.fromkeys((*selected, *context_columns))
    }
    records: list[dict[str, Any]] = []
    examples_remaining = max_absence_cells
    for a_index, b_index in processed_pairs:
        a_column, b_column = selected[a_index], selected[b_index]
        pair_tokens = {
            normalize_scalar(a_column, label=True),
            normalize_scalar(b_column, label=True),
        }
        if any(normalize_scalar(column, label=True) in pair_tokens for column in context_columns):
            raise ValueError("pair context columns must be disjoint from the analyzed pair")
        a_values, a_codes = encoded[a_column]
        b_values, b_codes = encoded[b_column]
        base_mask = np.ones(len(df), dtype=bool)
        if dropna:
            for column in (a_column, b_column):
                values, codes = encoded[column]
                if MISSING in values:
                    base_mask &= codes != values.index(MISSING)
        base_rows = np.flatnonzero(base_mask)
        global_a_raw = np.bincount(a_codes[base_rows], minlength=len(a_values))
        global_b_raw = np.bincount(b_codes[base_rows], minlength=len(b_values))
        global_a = Counter(
            {a_values[index]: int(count) for index, count in enumerate(global_a_raw) if count}
        )
        global_b = Counter(
            {b_values[index]: int(count) for index, count in enumerate(global_b_raw) if count}
        )
        domain_a, source_a = _declared_domain(reference_domains, a_column, set(global_a))
        domain_b, source_b = _declared_domain(reference_domains, b_column, set(global_b))
        for context_index, context in enumerate(contexts):
            local_mask = base_mask.copy()
            if dropna:
                for column in context:
                    values, codes = encoded[column]
                    if MISSING in values:
                        local_mask &= codes != values.index(MISSING)
            eligible_rows = int(local_mask.sum())
            for column, raw_value in context.items():
                values, codes = encoded[column]
                wanted = normalize_scalar(raw_value)
                try:
                    wanted_code = values.index(wanted)
                except ValueError:
                    local_mask[:] = False
                    break
                local_mask &= codes == wanted_code
            local_rows = np.flatnonzero(local_mask)
            local_a_codes = a_codes[local_rows]
            local_b_codes = b_codes[local_rows]
            pair_ids, code_pairs = exact_pair_ids(local_a_codes, local_b_codes)
            pair_sizes = np.bincount(pair_ids, minlength=len(code_pairs))
            pair_counts = Counter(
                {
                    (a_values[a_code], b_values[b_code]): int(count)
                    for (a_code, b_code), count in zip(code_pairs, pair_sizes)
                    if count
                }
            )
            a_raw = np.bincount(local_a_codes, minlength=len(a_values))
            b_raw = np.bincount(local_b_codes, minlength=len(b_values))
            a_support = Counter(
                {a_values[index]: int(count) for index, count in enumerate(a_raw) if count}
            )
            b_support = Counter(
                {b_values[index]: int(count) for index, count in enumerate(b_raw) if count}
            )
            association, association_reason = _cramers_v(pair_counts, a_support, b_support)
            record: dict[str, Any] = {
                "pair": [f"f{a_index}", f"f{b_index}"],
                "columns": [
                    normalize_scalar(a_column, label=True).to_dict(),
                    normalize_scalar(b_column, label=True).to_dict(),
                ],
                "context": [
                    {
                        "column": normalize_scalar(column, label=True).to_dict(),
                        "value": normalize_scalar(value).to_dict(),
                    }
                    for column, value in sorted(
                        context.items(),
                        key=lambda item: normalize_scalar(item[0], label=True).sort_key(),
                    )
                ],
                "scope": _scope(
                    f"pair:{a_index}:{b_index}:context:{context_index}",
                    len(df),
                    len(df) - eligible_rows,
                    eligible_rows - len(local_rows),
                    bool(context) or bool((scope_metadata or {}).get("conditional")),
                    parent_scope=(scope_metadata or {}).get("scope"),
                ),
                "evaluated_rows": len(local_rows),
                "relation": _relation(pair_counts),
                "relation_reason": None if pair_counts else "empty_population",
                "cramers_v": association,
                "cramers_v_reason": association_reason,
                "claim": "support_description_only",
                "higher_order_constraints_ruled_out": False,
                "marginals": {
                    "a_supported_levels": len(a_support),
                    "b_supported_levels": len(b_support),
                    "a": [
                        {"value": value.to_dict(), "count": count}
                        for value, count in sorted(
                            a_support.items(), key=lambda item: item[0].sort_key()
                        )
                    ],
                    "b": [
                        {"value": value.to_dict(), "count": count}
                        for value, count in sorted(
                            b_support.items(), key=lambda item: item[0].sort_key()
                        )
                    ],
                },
                "observed_cells": len(pair_counts),
                "domains": {
                    "a_size": len(domain_a),
                    "b_size": len(domain_b),
                    "a_source": source_a,
                    "b_source": source_b,
                },
            }
            if include_absence:
                total_cells = len(domain_a) * len(domain_b)
                total_absent = total_cells - len(pair_counts)
                zero_a = sum(1 for value in domain_a if global_a[value] == 0)
                zero_b = sum(1 for value in domain_b if global_b[value] == 0)
                zero_support = zero_a * len(domain_b) + zero_b * len(domain_a) - zero_a * zero_b
                positive_a = len(domain_a) - zero_a
                positive_b = len(domain_b) - zero_b
                local_zero_a = sum(
                    1 for value in domain_a if global_a[value] > 0 and a_support[value] == 0
                )
                local_zero_b = sum(
                    1 for value in domain_b if global_b[value] > 0 and b_support[value] == 0
                )
                parent_absent = (
                    local_zero_a * positive_b
                    + local_zero_b * positive_a
                    - local_zero_a * local_zero_b
                )
                within = total_absent - zero_support - parent_absent
                cap = examples_remaining
                example_cap = total_absent if cap is None else max(0, cap)
                examples: list[dict[str, Any]] = []
                scan_limit = max(1000, example_cap * 20)
                scanned = 0
                for a in domain_a:
                    for b in domain_b:
                        if len(examples) >= example_cap or scanned >= scan_limit:
                            break
                        scanned += 1
                        if (a, b) not in pair_counts:
                            examples.append({"a": a.to_dict(), "b": b.to_dict()})
                    if len(examples) >= example_cap or scanned >= scan_limit:
                        break
                if examples_remaining is not None:
                    examples_remaining -= len(examples)
                record["absence"] = {
                    "status": "computed",
                    "total_cells": total_cells,
                    "observed_cells": len(pair_counts),
                    "absent_cells": total_absent,
                    "classes": {
                        "unobserved_zero_support": zero_support,
                        "level_absent_under_parent": parent_absent,
                        "unobserved_within_supported_margins": within,
                    },
                    "examples": examples,
                    "examples_omitted": total_absent - len(examples),
                }
            records.append(record)
    return ExplorerResult(
        "pairs",
        {
            "status": "computed" if records else "empty",
            "source": _source(df),
            "scopes": [],
            "pairs": records,
            "features": [normalize_scalar(c, label=True).to_dict() for c in selected],
            "contexts": [
                [
                    {
                        "column": normalize_scalar(c, label=True).to_dict(),
                        "value": normalize_scalar(v).to_dict(),
                    }
                    for c, v in sorted(
                        context.items(),
                        key=lambda item: normalize_scalar(item[0], label=True).sort_key(),
                    )
                ]
                for context in contexts
            ],
            "absence_status": "computed" if include_absence else "not_requested",
            "requested_pairs": len(requested_pairs),
            "processed_pairs": len(processed_pairs),
            "omitted_pairs": len(requested_pairs) - len(processed_pairs),
            "requested_contexts": 1 + len(requested_context_values),
            "processed_contexts": len(contexts),
            "omitted_contexts": 1 + len(requested_context_values) - len(contexts),
            "scope_metadata": scope_metadata,
            "warnings": [],
        },
    )


def joint_counts(
    df: pd.DataFrame,
    dimensions: Iterable[Any],
    *,
    context: Mapping[Any, Any] | None = None,
    dropna: bool = False,
    max_cells: int = 2500,
) -> ExplorerResult:
    """Compute observed joint counts for one selected pair, on demand.

    ``max_cells`` bounds the supported-domain Cartesian product (including blank
    heatmap cells). Exceeding it raises rather than silently dropping cell mass.
    Context columns must be disjoint from the selected pair.
    """
    selected = resolve_columns(df, dimensions, argument="dimensions")
    if len(selected) != 2:
        raise ValueError("joint_counts requires exactly two dimensions")
    if max_cells is None:
        raise ValueError("max_cells must be a positive integer")
    validate_limit("max_cells", max_cells, zero=False)
    context = context or {}
    context_columns = resolve_columns(df, context, argument="context") if context else ()
    if set(context_columns) & set(selected):
        raise ValueError("context columns must be disjoint from the analyzed pair")
    encoded = {c: encode_series(df[c]) for c in (*selected, *context_columns)}
    mask = np.ones(len(df), dtype=bool)
    if dropna:
        for values, codes in encoded.values():
            if MISSING in values:
                mask &= codes != values.index(MISSING)
    eligible = int(mask.sum())
    for column, value in context.items():
        values, codes = encoded[column]
        token = normalize_scalar(value)
        if token not in values:
            mask[:] = False
        else:
            mask &= codes == values.index(token)
    a_values, a_codes = encoded[selected[0]]
    b_values, b_codes = encoded[selected[1]]
    a_supported = sorted(set(a_codes[mask].tolist()), key=lambda c: a_values[c].sort_key())
    b_supported = sorted(set(b_codes[mask].tolist()), key=lambda c: b_values[c].sort_key())
    if len(a_supported) * len(b_supported) > max_cells:
        raise ValueError(
            "Selected pair exceeds max_cells; narrow the context or increase the budget"
        )
    a_indexes = {code: i for i, code in enumerate(a_supported)}
    b_indexes = {code: i for i, code in enumerate(b_supported)}
    pair_ids, code_pairs = exact_pair_ids(a_codes[mask], b_codes[mask])
    sizes = np.bincount(pair_ids, minlength=len(code_pairs))
    cells = [
        {"a": a_indexes[a], "b": b_indexes[b], "count": int(size)}
        for (a, b), size in zip(code_pairs, sizes)
    ]
    cells.sort(key=lambda c: (c["a"], c["b"]))
    evaluated = int(mask.sum())
    return ExplorerResult(
        "joint_counts",
        {
            "status": "computed" if evaluated else "empty",
            "source": _source(df),
            "columns": [normalize_scalar(c, label=True).to_dict() for c in selected],
            "context": [
                {
                    "column": normalize_scalar(c, label=True).to_dict(),
                    "value": normalize_scalar(v).to_dict(),
                }
                for c, v in context.items()
            ],
            "scopes": [
                _scope("joint", len(df), len(df) - eligible, eligible - evaluated, bool(context))
            ],
            "a": [a_values[c].to_dict() for c in a_supported],
            "b": [b_values[c].to_dict() for c in b_supported],
            "cells": cells,
        },
    )
