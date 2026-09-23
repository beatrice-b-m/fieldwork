"""Sparse pair relationships, association, and bounded absence summaries."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any, Unpack

import numpy as np
import pandas as pd

from .._runtime import checkpoint, operation
from ..typing import Runtime
from ._kernels import exact_pair_ids
from .census import _scope, _source
from .encoding import (
    code_of,
    encode_column,
    json_value,
    labelled,
    missing_code,
    python_value,
    resolve_columns,
    validate_limit,
    value_key,
)

Key = tuple[int, Any]


def _key(value: Any) -> Key:
    """Identity of a dictionary value; the missing value sorts last."""
    return (9, 0) if value is None else value_key(value)


def _json(key: Key) -> Any:
    return None if key[0] == 9 else json_value(key[1])


def _context(df: pd.DataFrame, context: Mapping[Any, Any]) -> dict[str, Any]:
    """Context predicates keyed by column name, with canonical values."""
    columns = resolve_columns(df, context, argument="pair_contexts") if context else ()
    return {c: python_value(v) for c, v in zip(columns, context.values())}


def _predicates(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"column": c, "value": json_value(context[c])} for c in sorted(context)]


from ..result import Result


def _relation(pair_counts: Counter[tuple[Key, Key]]) -> str | None:
    if not pair_counts:
        return None
    a_to_b: dict[Key, set[Key]] = defaultdict(set)
    b_to_a: dict[Key, set[Key]] = defaultdict(set)
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
    pairs: Counter[tuple[Key, Key]],
    a_support: Counter[Key],
    b_support: Counter[Key],
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
    reference_domains: Mapping[str, Iterable[Any]],
    column: str,
    observed: set[Key],
) -> tuple[list[Key], str]:
    if column not in reference_domains:
        return sorted(observed), "empirical_observed"
    declared = {_key(python_value(value)) for value in reference_domains[column]}
    missing = observed - declared
    if missing:
        raise ValueError(
            f"Declared reference domain for {column!r} omits observed levels: "
            f"{[_json(key) for key in sorted(missing)]}"
        )
    return sorted(declared), "caller_declared"


@operation("pairs")
def pairs(
    df: pd.DataFrame,
    dimensions: Iterable[str],
    *,
    dropna: bool = False,
    include_absence: bool = False,
    reference_domains: Mapping[Any, Iterable[Any]] | None = None,
    pair_contexts: Iterable[Mapping[Any, Any]] | None = None,
    max_absence_cells: int | None = 1000,
    max_contexts: int | None = 32,
    max_pairs: int | None = 15,
    scope_metadata: dict[str, Any] | None = None,
    **runtime: Unpack[Runtime],
) -> Result:
    """Measure sparse pair mappings, association, and optional absence.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation. Column labels must be unique strings,
        non-boolean integers, or recursively tuple-valued labels. Native missing
        scalars share one identity; integer and float values remain distinct.
        Unsupported column labels or scalar objects raise TypeError.
    dimensions : iterable of column labels
        Nonempty unique columns. Pairs are generated in requested column order.
    dropna : bool, optional
        Default False includes missing categories. True excludes missing values
        from each tested pair and its context columns, so populations may differ.
    include_absence : bool, optional
        Include absent domain combinations when True; default False. Observed
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
    scope_metadata : mapping or None, optional
        Optional descriptive lineage supplied by composition; default None. This
        does not select rows. Use a Scope with census/explore for row selection.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'pairs', with per-pair/context scopes, mapping and association
        measurements, optional absence summaries, and omission coverage.

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
    Association is uncorrected Cramer's V and does not imply causality.
    Undefined statistics and their reasons remain explicit for degenerate tables.
    Absence uses declared domains when supplied, otherwise observed domains;
    limits bound output/search without sampling rows.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.pairs(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), ["a", "b"])
    >>> result.kind
    'pairs'
    """
    df = labelled(df)
    selected = resolve_columns(df, dimensions, argument="dimensions")
    validate_limit("max_absence_cells", max_absence_cells)
    validate_limit("max_contexts", max_contexts)
    validate_limit("max_pairs", max_pairs)
    requested_pairs = list(combinations(range(len(selected)), 2))
    processed_pairs = requested_pairs[:max_pairs] if max_pairs is not None else requested_pairs
    domains = {
        column if isinstance(column, str) else str(column): values
        for column, values in (reference_domains or {}).items()
    }
    requested_context_values = [_context(df, context) for context in pair_contexts or []]
    requested_context_values.sort(
        key=lambda context: tuple(sorted((c, _key(v)) for c, v in context.items()))
    )
    contexts: list[dict[str, Any]] = [{}]
    contexts.extend(requested_context_values)
    if max_contexts is not None:
        contexts = contexts[:max_contexts]
    context_columns = tuple(dict.fromkeys(column for context in contexts for column in context))
    encoded = {
        column: encode_column(df, column) for column in dict.fromkeys((*selected, *context_columns))
    }
    records: list[dict[str, Any]] = []
    examples_remaining = max_absence_cells
    for a_index, b_index in processed_pairs:
        checkpoint()
        a_column, b_column = selected[a_index], selected[b_index]
        if {a_column, b_column} & set(context_columns):
            raise ValueError("pair context columns must be disjoint from the analyzed pair")
        a_values, a_codes = encoded[a_column]
        b_values, b_codes = encoded[b_column]
        base_mask = np.ones(len(df), dtype=bool)
        if dropna:
            for column in (a_column, b_column):
                values, codes = encoded[column]
                absent = missing_code(values)
                if absent is not None:
                    base_mask &= codes != absent
        base_rows = np.flatnonzero(base_mask)
        global_a_raw = np.bincount(a_codes[base_rows], minlength=len(a_values))
        global_b_raw = np.bincount(b_codes[base_rows], minlength=len(b_values))
        global_a = Counter(
            {_key(a_values[index]): int(count) for index, count in enumerate(global_a_raw) if count}
        )
        global_b = Counter(
            {_key(b_values[index]): int(count) for index, count in enumerate(global_b_raw) if count}
        )
        domain_a, source_a = _declared_domain(domains, a_column, set(global_a))
        domain_b, source_b = _declared_domain(domains, b_column, set(global_b))
        for context_index, context in enumerate(contexts):
            local_mask = base_mask.copy()
            if dropna:
                for column in context:
                    values, codes = encoded[column]
                    absent = missing_code(values)
                    if absent is not None:
                        local_mask &= codes != absent
            eligible_rows = int(local_mask.sum())
            for column, value in context.items():
                values, codes = encoded[column]
                wanted_code = code_of(values, value)
                if wanted_code is None:
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
                    (_key(a_values[a_code]), _key(b_values[b_code])): int(count)
                    for (a_code, b_code), count in zip(code_pairs, pair_sizes)
                    if count
                }
            )
            a_raw = np.bincount(local_a_codes, minlength=len(a_values))
            b_raw = np.bincount(local_b_codes, minlength=len(b_values))
            a_support = Counter(
                {_key(a_values[index]): int(count) for index, count in enumerate(a_raw) if count}
            )
            b_support = Counter(
                {_key(b_values[index]): int(count) for index, count in enumerate(b_raw) if count}
            )
            association, association_reason = _cramers_v(pair_counts, a_support, b_support)
            record: dict[str, Any] = {
                "pair": [f"f{a_index}", f"f{b_index}"],
                "columns": [a_column, b_column],
                "context": _predicates(context),
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
                "marginals": {
                    "a_supported_levels": len(a_support),
                    "b_supported_levels": len(b_support),
                    "a": [
                        {"value": _json(key), "count": count}
                        for key, count in sorted(a_support.items())
                    ],
                    "b": [
                        {"value": _json(key), "count": count}
                        for key, count in sorted(b_support.items())
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
                            examples.append({"a": _json(a), "b": _json(b)})
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
    return Result(
        "pairs",
        {
            "status": "computed" if records else "empty",
            "source": _source(df),
            "scopes": [],
            "pairs": records,
            "features": list(selected),
            "contexts": [_predicates(context) for context in contexts],
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


@operation("joint counts")
def joint_counts(
    df: pd.DataFrame,
    dimensions: Iterable[str],
    *,
    context: Mapping[Any, Any] | None = None,
    dropna: bool = False,
    max_cells: int = 2500,
    **runtime: Unpack[Runtime],
) -> Result:
    """Count observed cells for one selected pair and optional context.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation. Column labels must be unique strings,
        non-boolean integers, or recursively tuple-valued labels. Native missing
        scalars share one identity; integer and float values remain distinct.
        Unsupported column labels or scalar objects raise TypeError.
    dimensions : iterable of column labels
        Exactly two distinct column labels, in axis order.
    context : mapping or None, optional
        Exact context values by column; default None. Context columns must be
        disjoint from the selected pair. An unmatched value yields an empty result.
    dropna : bool, optional
        Default False includes missing categories. True excludes rows with
        missing values in the selected pair or context columns.
    max_cells : int, optional
        Positive supported-domain Cartesian cell budget; default 2500. Includes
        blank heatmap cells, not only nonzero cells. None is not supported.
        Exceeding the budget raises ValueError instead of dropping cell mass.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'joint_counts', with axis dictionaries a and b, observed cells,
        context, and a population scope. Cells index the axis dictionaries.

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
    Only observed cells are stored. The budget applies to the cross-product of
    axis values supported in the evaluated context. Use a narrower context or
    increase max_cells when that product exceeds the limit.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> counts = fw.joint_counts(pd.DataFrame({"a": [1, 1], "b": ["x", "x"]}), ["a", "b"])
    >>> counts["cells"][0]["count"]
    2
    """
    df = labelled(df)
    selected = resolve_columns(df, dimensions, argument="dimensions")
    if len(selected) != 2:
        raise ValueError("joint_counts requires exactly two dimensions")
    if max_cells is None:
        raise ValueError("max_cells must be a positive integer")
    validate_limit("max_cells", max_cells, zero=False)
    context = _context(df, context or {})
    context_columns = tuple(context)
    if set(context_columns) & set(selected):
        raise ValueError("context columns must be disjoint from the analyzed pair")
    encoded = {c: encode_column(df, c) for c in (*selected, *context_columns)}
    mask = np.ones(len(df), dtype=bool)
    if dropna:
        for values, codes in encoded.values():
            absent = missing_code(values)
            if absent is not None:
                mask &= codes != absent
    eligible = int(mask.sum())
    for column, value in context.items():
        values, codes = encoded[column]
        code = code_of(values, value)
        if code is None:
            mask[:] = False
        else:
            mask &= codes == code
    a_values, a_codes = encoded[selected[0]]
    b_values, b_codes = encoded[selected[1]]
    a_supported = np.unique(a_codes[mask]).tolist()  # code order is value order
    b_supported = np.unique(b_codes[mask]).tolist()
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
    return Result(
        "joint_counts",
        {
            "status": "computed" if evaluated else "empty",
            "source": _source(df),
            "columns": list(selected),
            "context": [{"column": c, "value": json_value(v)} for c, v in context.items()],
            "scopes": [
                _scope("joint", len(df), len(df) - eligible, eligible - evaluated, bool(context))
            ],
            "a": [json_value(a_values[c]) for c in a_supported],
            "b": [json_value(b_values[c]) for c in b_supported],
            "cells": cells,
        },
    )
