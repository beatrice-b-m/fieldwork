"""Private exact grouping kernels used by the explorer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np


def dense_counts(codes: np.ndarray, rows: np.ndarray) -> dict[int, int]:
    if rows.size == 0:
        return {}
    values, counts = np.unique(codes[rows], return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def exact_pair_ids(
    parents: np.ndarray, levels: np.ndarray, *, packing_limit: int | None = None
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Dense IDs for exact integer pairs, with an overflow-safe tuple fallback."""

    if parents.shape != levels.shape:
        raise ValueError("parent and level arrays must have the same shape")
    if parents.size == 0:
        return np.empty(0, dtype=np.int64), []
    width = int(levels.max()) + 1
    max_parent = int(parents.max())
    safe = np.iinfo(np.int64).max // max(width, 1)
    if packing_limit is not None:
        safe = min(safe, packing_limit)
    if max_parent <= safe:
        packed = parents.astype(np.int64) * width + levels.astype(np.int64)
        unique, inverse = np.unique(packed, return_inverse=True)
        pairs = [(int(item // width), int(item % width)) for item in unique]
        return inverse.astype(np.int64, copy=False), pairs
    lookup: dict[tuple[int, int], int] = {}
    pair_list: list[tuple[int, int]] = []
    inverse = np.empty(len(parents), dtype=np.int64)
    for index, pair in enumerate(zip(parents.tolist(), levels.tolist())):
        normalized = (int(pair[0]), int(pair[1]))
        dense = lookup.get(normalized)
        if dense is None:
            dense = len(pair_list)
            lookup[normalized] = dense
            pair_list.append(normalized)
        inverse[index] = dense
    ordered = sorted(pair_list)
    remap = np.empty(len(pair_list), dtype=np.int64)
    for new_id, pair in enumerate(ordered):
        remap[lookup[pair]] = new_id
    return remap[inverse], ordered


def python_prefix_counts(rows: Iterable[tuple[object, ...]]) -> list[Counter]:
    """Readable reference kernel used by differential tests."""

    counters: list[Counter] = []
    for row in rows:
        for depth in range(1, len(row) + 1):
            if len(counters) < depth:
                counters.append(Counter())
            counters[depth - 1][row[:depth]] += 1
    return counters
