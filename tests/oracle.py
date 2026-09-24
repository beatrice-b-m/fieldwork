"""Small independent oracles; deliberately import no production code.

Values are compared through ``token``: native missing scalars (None, NaN, pd.NA,
NaT) collapse to one token, and booleans, integers, floats and strings stay
distinct even when they print alike. Exports hold plain JSON values (None for
missing, infinities as "inf"/"-inf"); ``record_token`` maps one to its token.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from fractions import Fraction
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


def record_token(value: Any) -> tuple[str, Any]:
    """Token of an exported value; JSON keeps the Python types that ``token`` separates."""
    if value in ("inf", "-inf"):
        return ("float", value)
    return token(value)


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


def match_token(value: Any, mode: str = "typed") -> tuple[str, Any]:
    """Cross-table identity: numbers compare exactly as fractions, never as booleans.

    'text' mode writes integer-valued numbers as decimal strings.
    """
    if token(value) == MISSING:
        return MISSING
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if mode == "text" and math.isfinite(value) and value == int(value):
            return ("string", str(int(value)))
        return ("number", Fraction(value) if math.isfinite(value) else value)
    return token(value)


def _keys(frame: pd.DataFrame, columns: Sequence[str], mode: str) -> list[tuple | None]:
    rows = []
    for values in zip(*(frame[c].tolist() for c in columns), strict=True):
        key = tuple(match_token(value, mode) for value in values)
        rows.append(None if MISSING in key else key)
    return rows


def relation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: dict[str, str],
    compare: dict[str, str],
    *,
    mode: str = "typed",
    dropna: bool = True,
) -> dict[str, Any]:
    """Enumerate keys, coverage, cardinality and agreement of two frames row by row.

    Row lists hold positions within each frame, as [left, right]; ``agreement``
    maps each compared pair to its per-key state counts and rows per side.
    """
    keys = (_keys(left, list(on), mode), _keys(right, list(on.values()), mode))
    counts = [Counter(k for k in side if k is not None) for side in keys]
    matched = counts[0].keys() & counts[1].keys()

    def rows(keep) -> list[list[int]]:
        return [
            [i for i, key in enumerate(keys[s]) if key is not None and keep(s, key)] for s in (0, 1)
        ]

    output: dict[str, Any] = {
        "keys": [set(c) for c in counts],
        "matched": matched,
        "incomplete": [sum(k is None for k in side) for side in keys],
        "found_rows": rows(lambda s, k: k in matched),
        "unfound_rows": rows(lambda s, k: k not in matched),
        "many": [any(c[k] > 1 for k in matched) for c in counts],
        "joined_rows": sum(counts[0][k] * counts[1][k] for k in matched),
        "repeated_rows": rows(lambda s, k: k in matched and counts[s][k] > 1),
        "single_rows": rows(lambda s, k: k in matched and counts[s][k] == 1),
        "agreement": {},
    }
    for pair in compare.items():
        values: list[dict[tuple, set]] = [{}, {}]
        for side, (frame, column) in enumerate(((left, pair[0]), (right, pair[1]))):
            for key, value in zip(keys[side], frame[column].tolist(), strict=True):
                item = match_token(value, mode)
                if key in matched and not (dropna and item == MISSING):
                    values[side].setdefault(key, set()).add(item)
        state: dict[tuple, str] = {}
        for key in matched:
            a, b = values[0].get(key, set()), values[1].get(key, set())
            if not a or not b:
                state[key] = "unavailable"
            elif len(a) > 1 or len(b) > 1:
                state[key] = "ambiguous"
            else:
                state[key] = "agree" if a == b else "disagree"
        output["agreement"][pair] = {
            "counts": Counter(state.values()),
            "agree_rows": rows(lambda s, k, state=state: state.get(k) == "agree"),
            "disagree_rows": rows(lambda s, k, state=state: state.get(k) == "disagree"),
        }
    return output


def reciprocity(frame: pd.DataFrame, reference: str, key: str) -> dict[str, Any]:
    """Self-reference edges own key → reference, reciprocated edges and self-loops."""
    edges_by_row = list(
        zip(_keys(frame, [key], "typed"), _keys(frame, [reference], "typed"), strict=True)
    )
    targets = {own for own, _ in edges_by_row if own is not None}
    edges = {(o, r) for o, r in edges_by_row if o is not None and r is not None and r in targets}
    loops = {edge for edge in edges if edge[0] == edge[1]}
    tested = edges - loops
    reciprocated = {(o, r) for o, r in tested if (r, o) in edges}
    return {
        "edges": tested,
        "reciprocated": reciprocated,
        "loops": loops,
        "reciprocated_rows": [i for i, e in enumerate(edges_by_row) if e in reciprocated],
        "unreciprocated_rows": [
            i for i, e in enumerate(edges_by_row) if e in tested - reciprocated
        ],
    }
