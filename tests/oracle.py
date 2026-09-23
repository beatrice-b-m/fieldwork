"""Small independent oracles; deliberately import no production code.

Values are compared through ``token``: native missing scalars (None, NaN, pd.NA,
NaT) collapse to one token, integers and floats stay distinct, and the other
tokens mirror the tagged ``{"type", "value"}`` records in exports, so
``record_token`` can compare them directly.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any

import pandas as pd

MISSING = ("missing", None)


def token(value: Any) -> tuple[str, Any]:
    if value is None or value is pd.NA or value is pd.NaT:
        return MISSING
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, int):
        return ("integer", str(value))
    if isinstance(value, float):
        if math.isnan(value):
            return MISSING
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return ("float", (0.0 if value == 0 else value).hex())
    if isinstance(value, str):
        return ("string", value)
    raise TypeError(type(value).__name__)


def record_token(record: dict[str, Any]) -> tuple[str, Any]:
    """Token of a tagged scalar record from a Fieldwork export."""
    return (record["type"], record.get("value"))


def column_tokens(frame: pd.DataFrame, column: Any) -> list[tuple[str, Any]]:
    return [token(value) for value in frame[column].tolist()]


def level_counts(frame: pd.DataFrame, column: Any) -> Counter:
    return Counter(column_tokens(frame, column))


def prefix_counts(frame: pd.DataFrame, columns: list[Any]) -> list[Counter]:
    output = [Counter() for _ in columns]
    tokens = [column_tokens(frame, column) for column in columns]
    for path in zip(*tokens):
        for depth in range(1, len(path) + 1):
            output[depth - 1][path[:depth]] += 1
    return output


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _codes(values: Sequence[Any]) -> list[int]:
    lookup: dict[Any, int] = {}
    return [lookup.setdefault(value, len(lookup)) for value in values]


def grouping(
    frame: pd.DataFrame, determinant: Sequence[str], rows: Sequence[int], *, dropna: bool
) -> dict[str, Any]:
    """Recompute a candidate key's complete cases and group structure."""
    tokens = {column: column_tokens(frame, column) for column in determinant}
    complete = [i for i in rows if not dropna or all(tokens[c][i] != MISSING for c in determinant)]
    keys = [tuple(tokens[c][i] for c in determinant) for i in complete]
    sizes = pd.Series(_codes(keys), dtype="int64").value_counts()
    repeated = sizes[sizes >= 2]
    return {
        "evaluated_rows": len(complete),
        "missing_excluded_rows": len(rows) - len(complete),
        "groups": len(sizes),
        "unique": bool(len(complete)) and len(sizes) == len(complete),
        "uniqueness": _ratio(len(sizes), len(complete)),
        "repeated_groups": len(repeated),
        "repeated_rows": int(repeated.sum()),
    }


def dependency(
    frame: pd.DataFrame,
    determinant: Sequence[str],
    target: str,
    rows: Sequence[int],
    *,
    dropna: bool,
) -> dict[str, Any]:
    """Recompute one dependency test on ``rows`` with a pandas groupby.

    Returns the exported record fields plus ``groups``: evaluated group key ->
    (source positions, target token counter), for checking exceptions.
    """
    tokens = {column: column_tokens(frame, column) for column in (*determinant, target)}
    complete = [i for i in rows if not dropna or all(tokens[c][i] != MISSING for c in determinant)]
    observed = [i for i in complete if tokens[target][i] != MISSING]
    evaluated = observed if dropna else complete
    keys = [tuple(tokens[c][i] for c in determinant) for i in evaluated]
    table = pd.DataFrame(
        {
            "key": pd.Series(_codes(keys), dtype="int64"),
            "target": pd.Series(_codes([tokens[target][i] for i in evaluated]), dtype="int64"),
        }
    )
    sizes = table.groupby("key").size()
    pair_sizes = table.groupby(["key", "target"]).size()
    maxima = pair_sizes.groupby(level="key").max()
    distinct = pair_sizes.groupby(level="key").size()
    repair = int((sizes - maxima).sum())
    violating = distinct[distinct > 1].index
    repeated = sizes[sizes >= 2]
    groups: dict[tuple, tuple[list[int], Counter]] = {}
    for key, position in zip(keys, evaluated):
        positions, counts = groups.setdefault(key, ([], Counter()))
        positions.append(position)
        counts[tokens[target][position]] += 1
    return {
        "exact": repair == 0 if evaluated else None,
        "modal_accuracy": 1 - repair / len(evaluated) if evaluated else None,
        "repair_rows": repair,
        "evaluated_rows": len(evaluated),
        "determinant_evaluated_rows": len(complete),
        "target_observed_rows": len(observed),
        "target_coverage": _ratio(len(observed), len(complete)),
        "target_missing_excluded_rows": len(complete) - len(evaluated),
        "repeated_rows": int(repeated.sum()),
        "repeat_coverage": _ratio(int(repeated.sum()), len(evaluated)),
        "repeat_modal_accuracy": (1 - repair / int(repeated.sum())) if len(repeated) else None,
        "missing_excluded_rows": len(rows) - len(evaluated),
        "evaluated_groups": len(sizes),
        "violating_groups": len(violating),
        "affected_rows": int(sizes[violating].sum()),
        "group_violation_rate": _ratio(len(violating), len(sizes)),
        "repeated_groups": len(repeated),
        "groups": groups,
    }


def determinants(features: Sequence[str], max_key_size: int) -> Iterable[tuple[str, ...]]:
    """Candidate determinants in documented order: size, then input column order."""
    for size in range(1, max_key_size + 1):
        yield from combinations(features, size)
