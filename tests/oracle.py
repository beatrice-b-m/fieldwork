"""Small independent oracles; deliberately import no production code.

Values are compared through ``token``: native missing scalars (None, NaN, pd.NA,
NaT) collapse to one token, integers and floats stay distinct, and the other
tokens mirror the tagged ``{"type", "value"}`` records in exports, so
``record_token`` can compare them directly.
"""

from __future__ import annotations

import math
from collections import Counter
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
