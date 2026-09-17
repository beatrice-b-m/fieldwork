"""Small independent loop oracle; deliberately imports no production code."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd


def token(value: Any) -> tuple[str, Any]:
    if value is None or value is pd.NA or value is pd.NaT:
        return ("missing", None)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, int):
        return ("integer", str(value))
    if isinstance(value, float):
        if math.isnan(value):
            return ("missing", None)
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return ("float", (0.0 if value == 0 else value).hex())
    if isinstance(value, str):
        return ("string", value)
    raise TypeError(type(value).__name__)


def level_counts(frame: pd.DataFrame, column: Any) -> Counter:
    result: Counter = Counter()
    for value in frame[column].tolist():
        result[token(value)] += 1
    return result


def prefix_counts(frame: pd.DataFrame, columns: list[Any]) -> list[Counter]:
    output = [Counter() for _ in columns]
    for row in frame[columns].itertuples(index=False, name=None):
        path = tuple(token(value) for value in row)
        for depth in range(1, len(path) + 1):
            output[depth - 1][path[:depth]] += 1
    return output
