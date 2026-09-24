"""Pair mappings, association, bounded absence summaries, and joint counts."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any, Unpack

import numpy as np
import pandas as pd

from .._runtime import checkpoint, operation
from ..result import Result
from ..typing import PairLimits, Runtime
from ._kernels import exact_pair_ids
from .census import _complete
from .encoding import code_of, json_value, python_value, validate_limit, value_key

if TYPE_CHECKING:
    from ..evidence import Scope

Key = tuple[int, Any]


def _key(value: Any) -> Key:
    """Identity of a dictionary value; the missing value sorts last."""
    return (9, 0) if value is None else value_key(value)


def _json(key: Key) -> Any:
    return None if key[0] == 9 else json_value(key[1])


def _context(df: pd.DataFrame, context: Mapping[Any, Any]) -> dict[str, Any]:
    """Context predicates keyed by column name, with canonical values."""
    from ..evidence import columns

    return {c: python_value(v) for c, v in zip(columns(df, context), context.values())}


def _predicates(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"column": c, "value": json_value(context[c])} for c in sorted(context)]


def _as_mapping(context: Mapping[str, Any]) -> dict[str, Any]:
    return {c: json_value(v) for c, v in context.items()}


def _context_rows(
    encoded: Mapping[str, tuple[list[Any], np.ndarray]],
    context: Mapping[str, Any],
    mask: np.ndarray,
    dropna: bool,
) -> tuple[np.ndarray, int]:
    """Rows matching every context value, and how many rows were eligible before."""
    if dropna:
        mask = mask & _complete([encoded[c] for c in context], len(mask))
    eligible = int(mask.sum())
    for column, value in context.items():
        values, codes = encoded[column]
        code = code_of(values, value)
        mask = mask & (codes == code) if code is not None else np.zeros_like(mask)
    return np.flatnonzero(mask), eligible


def _relation(pair_counts: Counter[tuple[Key, Key]]) -> str | None:
    if not pair_counts:
        return None
    a_to_b: dict[Key, set[Key]] = defaultdict(set)
    b_to_a: dict[Key, set[Key]] = defaultdict(set)
    for a, b in pair_counts:
        a_to_b[a].add(b)
        b_to_a[b].add(a)
    a_many = any(len(values) > 1 for values in a_to_b.values())
    b_many = any(len(values) > 1 for values in b_to_a.values())
    return {(False, False): "1:1", (True, False): "1:n", (False, True): "n:1"}.get(
        (a_many, b_many), "n:m"
    )


def _cramers_v(
    pairs: Counter[tuple[Key, Key]], a_support: Counter[Key], b_support: Counter[Key]
) -> tuple[float | None, str | None]:
    total = sum(pairs.values())
    if total == 0:
        return None, "empty_population"
    denominator = min(len(a_support) - 1, len(b_support) - 1)
    if denominator <= 0:
        return None, "constant_dimension"
    chi_term = sum(count * count / (a_support[a] * b_support[b]) for (a, b), count in pairs.items())
    chi2 = max(0.0, total * chi_term - total)
    return math.sqrt(chi2 / (total * denominator)), None


def _declared_domain(
    reference_domains: Mapping[str, Iterable[Any]], column: str, observed: set[Key]
) -> tuple[list[Key], str]:
    if column not in reference_domains:
        return sorted(observed), "empirical_observed"
    declared = {_key(python_value(value)) for value in reference_domains[column]}
    missing = observed - declared
    if missing:
        # Name the column and count, never the levels: messages must not echo source values.
        raise ValueError(
            f"Declared reference domain for {column!r} omits observed levels "
            f"({len(missing)}); compare it with levels(df, [{column!r}])"
        )
    return sorted(declared), "caller_declared"


def _support(values: list[Any], codes: np.ndarray) -> Counter[Key]:
    raw = np.bincount(codes, minlength=len(values))
    return Counter({_key(values[i]): int(n) for i, n in enumerate(raw) if n})


@dataclass
class _Pair:
    """One analyzed column pair: codes, pair population, marginals and domains."""

    columns: tuple[str, str]
    encoded: tuple[tuple[list[Any], np.ndarray], tuple[list[Any], np.ndarray]]
    base: np.ndarray
    margins: tuple[Counter[Key], Counter[Key]]
    domains: tuple[list[Key], list[Key]]
    sources: tuple[str, str]


_LIMITS = {"max_pairs": 15, "max_contexts": 32, "max_absence_cells": 1000}


@operation("pairs")
def pairs(
    df: pd.DataFrame,
    dimensions: Iterable[str],
    *,
    dropna: bool = False,
    include_absence: bool = False,
    reference_domains: Mapping[str, Iterable[Any]] | None = None,
    pair_contexts: Iterable[Mapping[str, Any]] | None = None,
    limits: PairLimits | None = None,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Measure how column pairs map onto each other, globally and within contexts.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    dimensions : iterable of str
        Nonempty columns; pairs follow their order.
    dropna : bool, optional
        True excludes rows missing either column (or a context column) from each
        pair; default False keeps missing values as a category.
    include_absence, reference_domains : optional
        Also count and sample value combinations never observed (default False),
        drawn from declared value domains by column (default observed values).
    pair_contexts : iterable of mappings or None, optional
        Extra analyses restricted to exact column values, such as
        ``{"site": "North"}``; context columns must not be in the pair.
    limits : PairLimits or None, optional
        Budgets, None unbounded: pairs (``max_pairs``, default 15), contexts
        including the global one (``max_contexts``, 32) and sampled absent cells
        (``max_absence_cells``, 1000). Omitted work is reported.
    scope, missing, table_id
        Source context shared by every analysis.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'pairs': one record per pair and context with its evaluated rows,
        relation ('1:1', '1:n', 'n:1', 'n:m'), uncorrected Cramér's V (None with
        a reason when undefined), marginals and optional absence summary.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.pairs(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), ["a", "b"])
    >>> result["pairs"][0]["relation"]
    '1:1'
    """
    from ..evidence import budgets, columns, prepare_values

    selected = columns(df, dimensions)
    if not selected:
        raise ValueError("dimensions must contain at least one column")
    caps = budgets(limits, _LIMITS, nullable=_LIMITS)
    max_contexts, max_pairs = caps["max_contexts"], caps["max_pairs"]
    requested = [_context(df, context) for context in pair_contexts or []]
    requested.sort(key=lambda context: tuple(sorted((c, _key(v)) for c, v in context.items())))
    contexts = [{}, *requested]
    contexts = contexts[:max_contexts] if max_contexts is not None else contexts
    context_columns = list(dict.fromkeys(c for context in contexts for c in context))
    frame, _, encoded, base = prepare_values(
        df,
        list(dict.fromkeys([*selected, *context_columns])),
        scope=scope,
        missing=missing,
        table_id=table_id,
    )
    domains = {str(c): values for c, values in (reference_domains or {}).items()}
    candidates = list(combinations(selected, 2))
    processed = candidates[:max_pairs] if max_pairs is not None else candidates
    budget: list[int | None] = [caps["max_absence_cells"]]
    records = []
    for a, b in processed:
        checkpoint()
        if {a, b} & set(context_columns):
            raise ValueError("pair context columns must be disjoint from the analyzed pair")
        pair = _pair(encoded, (a, b), dropna, domains, len(frame))
        for context in contexts:
            rows, eligible = _context_rows(encoded, context, pair.base, dropna)
            record, counts = _pair_record(pair, context, rows, eligible, len(frame))
            if include_absence:
                record["absence"] = _absence(pair, counts, budget)
            records.append(record)
    base["status"] = "computed" if records else "empty"
    base["parameters"] = {
        "dimensions": selected,
        "dropna": dropna,
        "include_absence": include_absence,
        "reference_domains": {
            c: [json_value(python_value(v)) for v in values] for c, values in domains.items()
        },
        "pair_contexts": [_as_mapping(context) for context in requested],
        "limits": caps,
    }
    base.update(
        pairs=records,
        features=selected,
        contexts=[_predicates(context) for context in contexts],
        absence_status="computed" if include_absence else "not_requested",
        requested_pairs=len(candidates),
        processed_pairs=len(processed),
        omitted_pairs=len(candidates) - len(processed),
        requested_contexts=1 + len(requested),
        processed_contexts=len(contexts),
        omitted_contexts=1 + len(requested) - len(contexts),
        warnings=[],
    )
    return Result("pairs", base)


def _pair(encoded, columns: tuple[str, str], dropna: bool, domains, size: int) -> _Pair:
    """Codes, pair population (dropna) and global value domains for one pair."""
    pair_encoded = (encoded[columns[0]], encoded[columns[1]])
    mask = _complete(list(pair_encoded), size) if dropna else np.ones(size, dtype=bool)
    rows = np.flatnonzero(mask)
    margins = (
        _support(pair_encoded[0][0], pair_encoded[0][1][rows]),
        _support(pair_encoded[1][0], pair_encoded[1][1][rows]),
    )
    domain_a, source_a = _declared_domain(domains, columns[0], set(margins[0]))
    domain_b, source_b = _declared_domain(domains, columns[1], set(margins[1]))
    return _Pair(columns, pair_encoded, mask, margins, (domain_a, domain_b), (source_a, source_b))


def _pair_record(
    pair: _Pair, context: Mapping[str, Any], rows: np.ndarray, eligible: int, size: int
) -> tuple[dict[str, Any], tuple[Counter, Counter, Counter]]:
    (a_values, a_codes), (b_values, b_codes) = pair.encoded
    ids, code_pairs = exact_pair_ids(a_codes[rows], b_codes[rows])
    sizes = np.bincount(ids, minlength=len(code_pairs))
    counts = Counter(
        {
            (_key(a_values[a]), _key(b_values[b])): int(n)
            for (a, b), n in zip(code_pairs, sizes)
            if n
        }
    )
    a_support = _support(a_values, a_codes[rows])
    b_support = _support(b_values, b_codes[rows])
    association, reason = _cramers_v(counts, a_support, b_support)
    record = {
        "columns": list(pair.columns),
        "context": _predicates(context),
        "evaluated_rows": len(rows),
        "missing_excluded_rows": size - eligible,
        "restriction_excluded_rows": eligible - len(rows),
        "relation": _relation(counts),
        "relation_reason": None if counts else "empty_population",
        "cramers_v": association,
        "cramers_v_reason": reason,
        "marginals": {
            "a_supported_levels": len(a_support),
            "b_supported_levels": len(b_support),
            "a": [{"value": _json(k), "count": n} for k, n in sorted(a_support.items())],
            "b": [{"value": _json(k), "count": n} for k, n in sorted(b_support.items())],
        },
        "observed_cells": len(counts),
        "domains": {
            "a_size": len(pair.domains[0]),
            "b_size": len(pair.domains[1]),
            "a_source": pair.sources[0],
            "b_source": pair.sources[1],
        },
    }
    return record, (counts, a_support, b_support)


def _absence(pair: _Pair, counts, budget: list[int | None]) -> dict[str, Any]:
    """Classify unobserved domain cells and sample them within the shared budget."""
    observed, a_support, b_support = counts
    (domain_a, domain_b), (global_a, global_b) = pair.domains, pair.margins
    total = len(domain_a) * len(domain_b)
    absent = total - len(observed)
    zero_a = sum(1 for v in domain_a if global_a[v] == 0)
    zero_b = sum(1 for v in domain_b if global_b[v] == 0)
    zero_support = zero_a * len(domain_b) + zero_b * len(domain_a) - zero_a * zero_b
    local_a = sum(1 for v in domain_a if global_a[v] > 0 and a_support[v] == 0)
    local_b = sum(1 for v in domain_b if global_b[v] > 0 and b_support[v] == 0)
    parent_absent = (
        local_a * (len(domain_b) - zero_b) + local_b * (len(domain_a) - zero_a) - local_a * local_b
    )
    cap = absent if budget[0] is None else max(0, budget[0])
    examples, scanned, limit = [], 0, max(1000, cap * 20)
    for a in domain_a:
        for b in domain_b:
            if len(examples) >= cap or scanned >= limit:
                break
            scanned += 1
            if (a, b) not in observed:
                examples.append({"a": _json(a), "b": _json(b)})
        if len(examples) >= cap or scanned >= limit:
            break
    if budget[0] is not None:
        budget[0] -= len(examples)
    return {
        "status": "computed",
        "total_cells": total,
        "observed_cells": len(observed),
        "absent_cells": absent,
        "classes": {
            "unobserved_zero_support": zero_support,
            "level_absent_under_parent": parent_absent,
            "unobserved_within_supported_margins": absent - zero_support - parent_absent,
        },
        "examples": examples,
        "examples_omitted": absent - len(examples),
    }


@operation("joint counts")
def joint_counts(
    df: pd.DataFrame,
    dimensions: Iterable[str],
    *,
    context: Mapping[str, Any] | None = None,
    dropna: bool = False,
    max_cells: int = 2500,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Count the observed value combinations of one column pair.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    dimensions : iterable of str
        Exactly two columns, in axis order.
    context : mapping or None, optional
        Exact values restricting rows, such as ``{"site": "North"}``; context
        columns must not be in the pair. Default None.
    dropna : bool, optional
        True excludes rows missing a pair or context column; default False.
    max_cells : int, optional
        Positive budget for the supported-value grid (blank cells included);
        default 2500. Exceeding it raises ValueError rather than dropping mass.
    scope, missing, table_id
        Source context shared by every analysis.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'joint_counts': axis values ``a`` and ``b`` in value order, observed
        ``cells`` indexing them, and the evaluated, missing-excluded and
        context-excluded rows.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> counts = fw.joint_counts(pd.DataFrame({"a": [1, 1], "b": ["x", "x"]}), ["a", "b"])
    >>> counts["cells"][0]["count"]
    2
    """
    from ..evidence import columns, prepare_values

    selected = columns(df, dimensions)
    if len(selected) != 2:
        raise ValueError("joint_counts requires exactly two dimensions")
    if max_cells is None:
        raise ValueError("max_cells must be a positive integer")
    validate_limit("max_cells", max_cells, zero=False)
    restriction = _context(df, context or {})
    if set(restriction) & set(selected):
        raise ValueError("context columns must be disjoint from the analyzed pair")
    frame, _, encoded, base = prepare_values(
        df, [*selected, *restriction], scope=scope, missing=missing, table_id=table_id
    )
    size = len(frame)
    mask = _complete([encoded[c] for c in selected], size) if dropna else np.ones(size, bool)
    rows, eligible = _context_rows(encoded, restriction, mask, dropna)
    (a_values, a_codes), (b_values, b_codes) = encoded[selected[0]], encoded[selected[1]]
    a_supported = np.unique(a_codes[rows]).tolist()  # code order is value order
    b_supported = np.unique(b_codes[rows]).tolist()
    if len(a_supported) * len(b_supported) > max_cells:
        raise ValueError(
            "Selected pair exceeds max_cells; narrow the context or increase the budget"
        )
    a_index = {code: i for i, code in enumerate(a_supported)}
    b_index = {code: i for i, code in enumerate(b_supported)}
    ids, code_pairs = exact_pair_ids(a_codes[rows], b_codes[rows])
    sizes = np.bincount(ids, minlength=len(code_pairs))
    cells = sorted(
        (
            {"a": a_index[a], "b": b_index[b], "count": int(n)}
            for (a, b), n in zip(code_pairs, sizes)
        ),
        key=lambda cell: (cell["a"], cell["b"]),
    )
    base["status"] = "computed" if len(rows) else "empty"
    base["parameters"] = {
        "dimensions": selected,
        "context": _as_mapping(restriction),
        "dropna": dropna,
        "max_cells": max_cells,
    }
    base.update(
        columns=selected,
        context=[{"column": c, "value": json_value(v)} for c, v in restriction.items()],
        evaluated_rows=len(rows),
        missing_excluded_rows=size - eligible,
        restriction_excluded_rows=eligible - len(rows),
        a=[json_value(a_values[c]) for c in a_supported],
        b=[json_value(b_values[c]) for c in b_supported],
        cells=cells,
    )
    return Result("joint_counts", base)
