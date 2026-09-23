"""Bounded populated-value summaries and evidence-labelled column families."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations, islice
from typing import Any, Unpack

import numpy as np
import pandas as pd

from ._explore._kernels import group_ids, modal_groups
from ._explore.encoding import encode_series
from ._runtime import checkpoint, operation, phase
from .evidence import (
    Scope,
    analyzable,
    bounded_rows,
    budgets,
    columns,
    finding,
    prepare,
    result,
)
from .result import Result
from .typing import PatternLimits, Runtime

_LIMITS = {"max_pairs": 100, "max_patterns": 10, "example_limit": 5}


@operation("value patterns")
def value_patterns(
    df: pd.DataFrame,
    *,
    features: Iterable[str] | None = None,
    by: Iterable[str] | None = None,
    limits: PatternLimits | None = None,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Summarize populated values: string formats, numeric ranges and relations.

    A column is numeric when every populated value is a non-boolean number,
    whatever its dtype. Formats, tolerances and tests are described in
    docs/algorithms.md.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    features : iterable of str or None, optional
        Columns to summarize; default None selects every column, skipping (and
        listing) columns with unsupported values.
    by : iterable of str or None, optional
        Context columns: each other feature is tested for being constant within
        each joint context (rows missing a context value are excluded).
    limits : PatternLimits or None, optional
        Budgets: pairs tested for constant offsets and ratios (``max_pairs``,
        100), formats, lengths and prefixes kept per string column
        (``max_patterns``, 10) and ``example_limit`` (5).
    scope, missing, table_id
        Source context shared by every analysis.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'value_patterns': per-column ``summaries``, indexed name
        ``families``, ``coverage`` and findings (string patterns, numeric ranges,
        constant offsets and ratios, context constancy, indexed families).

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"code": ["AB12", "CD34", None]})
    >>> result = fw.value_patterns(df)
    >>> result["summaries"][0]["formats"]
    [['A9', 2]]
    """
    budget = budgets(limits, _LIMITS)
    selected = columns(df, features)
    contexts = columns(df, by or [])
    frame, positions, codes, present, base = prepare(
        df,
        missing=missing,
        scope=scope,
        table_id=table_id,
        features=[*selected, *contexts] if contexts else [],
        presence_features=selected,
        optional=selected if features is None else (),
    )
    patterns = _Patterns(frame, positions, present, base, budget["example_limit"])
    selected = analyzable(selected, base)
    base["summaries"] = []
    with phase("value summaries", len(selected), "columns") as tracker:
        for c in selected:
            base["summaries"].append(_summary(patterns, c, budget["max_patterns"]))
            tracker.advance(detail=c)
    base["families"] = _indexed_families(patterns, selected)
    tested = _numeric_pairs(patterns, selected, budget["max_pairs"])
    if contexts:
        _context_constancy(patterns, codes, selected, contexts)
    base["coverage"] = {
        "pair_candidates": len(selected) * (len(selected) - 1) // 2,
        "pairs_evaluated": tested,
    }
    base["parameters"] = {
        "features": selected,
        "by": contexts,
        "limits": budget,
    }
    return result("value_patterns", base)


@dataclass
class _Patterns:
    frame: pd.DataFrame
    positions: np.ndarray
    present: dict[str, np.ndarray]
    base: dict[str, Any]
    example_limit: int
    numeric: dict[str, np.ndarray | None] = field(default_factory=dict)

    def number(
        self, c: str, series: pd.Series | None = None, kind: str | None = None
    ) -> np.ndarray | None:
        """The column as floats when numeric (see numbers), computed once."""
        if c not in self.numeric:
            series = self.frame[c] if series is None else series
            self.numeric[c] = numbers(series, self.present[c], kind)
        return self.numeric[c]

    def emit(self, kind, statement, features, metrics, rows, **extra) -> None:
        finding(
            self.base,
            kind,
            statement,
            features,
            metrics,
            rows,
            example_limit=self.example_limit,
            **extra,
        )


def numbers(series: pd.Series, present: np.ndarray, kind: str | None = None) -> np.ndarray | None:
    """The column as floats (NaN where absent) when every present value is a number.

    Numeric-ness follows the values, not the dtype: an object or categorical
    column of numbers is numeric, and booleans never are. A column with no
    present value is numeric when its dtype is. ``kind`` is the present values'
    ``pandas.api.types.infer_dtype``, when the caller already has it.
    """
    dtype = series.dtype
    native = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
    if _native_numbers(dtype):
        output = series.to_numpy(dtype=float, na_value=np.nan)
    else:
        if not isinstance(dtype, pd.CategoricalDtype):
            # Reject strings and other non-numbers without encoding the column.
            if kind is None:
                kind = pd.api.types.infer_dtype(series.to_numpy()[present], skipna=True)
            if kind not in ("integer", "floating", "mixed-integer-float", "empty"):
                return None
        values, codes = encode_series(series)
        used = [values[i] for i in np.unique(codes[present])]
        if not (all(type(v) in (int, float) for v in used) if used else native):
            return None
        table = np.array(
            [float(v) if type(v) in (int, float) else np.nan for v in values] or [np.nan]
        )
        output = table[codes] if len(values) else np.full(len(codes), np.nan)
    return np.where(present, output, np.nan)


def _native_numbers(dtype) -> bool:
    """A numeric, non-boolean, non-categorical dtype: its values are numbers."""
    return (
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not isinstance(dtype, pd.CategoricalDtype)
    )


def _summary(patterns: _Patterns, c: str, max_patterns: int) -> dict[str, Any]:
    present = patterns.present[c]
    series = patterns.frame[c]
    values = series.iloc[np.flatnonzero(present)]
    record = {"feature": c, "populated": len(values), "missing": len(patterns.frame) - len(values)}
    examples = bounded_rows(patterns.positions, present, patterns.example_limit)
    kind = pd.api.types.infer_dtype(values.to_numpy(copy=False), skipna=True)
    if len(values) and kind == "string":
        record.update(_string_summary(values, max_patterns))
        patterns.emit(
            "string_patterns", f"{c}: string formats, lengths and prefixes", [c], record, examples
        )
    if _native_numbers(series.dtype):
        number = values.to_numpy(dtype=float, na_value=np.nan)
    else:
        number = patterns.number(c, series, kind)
        number = None if number is None else number[present]
    if number is not None:
        record.update(_numeric_summary(number))
        patterns.emit(
            "numeric_range", f"{c}: numeric range and observed spacing", [c], record, examples
        )
    return record


def _string_summary(values: pd.Series, max_patterns: int) -> dict[str, Any]:
    ids, uniques = pd.factorize(values, sort=False)
    counts = np.bincount(ids)
    formats, lengths, prefixes = Counter(), Counter(), Counter()
    for i, (value, count) in enumerate(zip(uniques, counts)):
        if i % 8192 == 0:
            checkpoint()
        formats[re.sub(r"[A-Za-z]+", "A", re.sub(r"\d+", "9", value))] += int(count)
        lengths[len(value)] += int(count)
        prefixes[value[:3]] += int(count)
    # Lists, not most_common() tuples, so live and saved payloads match.
    return {
        "formats": [list(item) for item in formats.most_common(max_patterns)],
        "lengths": [list(item) for item in lengths.most_common(max_patterns)],
        "prefixes": [list(item) for item in prefixes.most_common(max_patterns)],
        "format_count": len(formats),
        "omitted_format_rows": sum(n for _, n in formats.most_common()[max_patterns:]),
    }


def _numeric_summary(numeric: np.ndarray) -> dict[str, Any]:
    finite = np.sort(np.unique(numeric[np.isfinite(numeric)]))
    differences = np.diff(finite)
    step = float(differences.min()) if len(differences) else None
    on_grid = None
    if step:
        offsets = (finite - finite[0]) / step
        on_grid = bool(np.allclose(offsets, np.round(offsets)))
    return {
        "minimum": float(finite[0]) if len(finite) else None,
        "maximum": float(finite[-1]) if len(finite) else None,
        "nonfinite": int((~np.isfinite(numeric)).sum()),
        "observed_step": step,
        "on_observed_step_grid": on_grid,
    }


def _indexed_families(patterns: _Patterns, selected: list[str]) -> list[dict[str, Any]]:
    """Columns sharing a name up to a trailing index, such as dose_1 and dose_2."""
    families = defaultdict(list)
    for c in selected:
        match = re.match(r"^(.*?)[_\-]?\d+$", c)
        if match:
            families[match.group(1)].append(c)
    records = []
    for prefix, group in families.items():
        if len(group) < 2:
            continue
        evidence = ["indexed_name"]
        present = patterns.present
        if all(np.array_equal(present[group[0]], present[c]) for c in group[1:]):
            evidence.append("identical_availability")
        records.append({"features": group, "prefix": prefix, "evidence": evidence})
        patterns.emit(
            "indexed_family",
            f"Indexed family: {', '.join(group)}",
            group,
            {"evidence": evidence},
            patterns.positions,
        )
    return records


def _numeric_pairs(patterns: _Patterns, selected: list[str], max_pairs: int) -> int:
    """Constant offsets (b - a) and ratios (b / a) between numeric columns."""
    tested = 0
    for a, b in islice(combinations(selected, 2), max_pairs):
        checkpoint()
        tested += 1
        eligible = patterns.present[a] & patterns.present[b]
        if not eligible.any():
            continue
        x, y = patterns.number(a), patterns.number(b)
        if x is None or y is None:
            continue
        x, y = x[eligible], y[eligible]
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
                patterns.emit(
                    f"numeric_{name}",
                    f"{b} has a constant {name} relative to {a}",
                    [a, b],
                    {
                        "value": float(values[0]),
                        "evaluated_rows": len(values),
                        "excluded_rows": len(patterns.frame) - len(values),
                        "rtol": 1e-5,
                        "atol": 1e-8,
                    },
                    patterns.positions[eligible][finite][mask],
                )
    return tested


def _context_constancy(patterns: _Patterns, codes, selected, contexts) -> None:
    """Whether each feature (other than the contexts) is constant within each context."""
    present = patterns.present
    valid_rows = np.flatnonzero(np.logical_and.reduce([present[c] for c in contexts]))
    context_ids = group_ids(codes[c][valid_rows] for c in contexts)
    features = [c for c in selected if c not in contexts]
    positions, limit_ = patterns.positions[valid_rows], patterns.example_limit
    with phase("context constancy", len(features), "columns") as tracker:
        for c in features:
            populated = present[c][valid_rows]
            valid_contexts = pd.unique(context_ids[populated])
            _, sizes, _, _, distinct = modal_groups(
                context_ids[populated], codes[c][valid_rows[populated]]
            )
            constant = np.isin(context_ids, valid_contexts[distinct == 1])
            nonconstant = np.isin(context_ids, valid_contexts[distinct > 1])
            count = int(np.count_nonzero(distinct == 1))
            patterns.emit(
                "context_constancy",
                f"{c}: constancy within {', '.join(contexts)}",
                [*contexts, c],
                {
                    "evaluated_groups": len(sizes),
                    "constant_groups": count,
                    "constant_group_fraction": count / len(sizes) if len(sizes) else None,
                },
                bounded_rows(positions, constant, limit_),
                exceptions=bounded_rows(positions, nonconstant, limit_),
            )
            tracker.advance(detail=c)
