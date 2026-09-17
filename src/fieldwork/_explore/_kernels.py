"""Private exact grouping kernels used by the explorer."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable

import numpy as np
import pandas as pd


def group_ids(arrays):
    """Dense first-observed integer-tuple groups without Python row tuples."""
    arrays = list(arrays)
    if not arrays:
        return np.empty(0, dtype=np.int64)
    ids = pd.factorize(arrays[0], sort=False)[0]
    for values in arrays[1:]:
        right = pd.factorize(values, sort=False)[0]
        ids = pair_groups(ids, right)
    return ids


def pair_groups(left, right):
    """Exact dense first-observed pair IDs, including negative input codes."""
    if not len(left):
        return np.empty(0, dtype=np.int64)
    a = pd.factorize(left, sort=False)[0]
    b = pd.factorize(right, sort=False)[0]
    width = int(b.max()) + 1
    # Factored IDs are bounded by the row count, not arbitrary input integers.
    if int(a.max()) <= (np.iinfo(np.int64).max - int(b.max())) // max(1, width):
        return pd.factorize(a * width + b, sort=False)[0]
    return pd.MultiIndex.from_arrays([a, b]).factorize(sort=False)[0]


def modal_groups(keys, targets):
    """Group sizes, modes, and distinct target counts; smallest code wins ties."""
    ids = pd.factorize(keys, sort=False)[0]
    if not len(ids):
        empty = np.empty(0, dtype=np.int64)
        return ids, empty, empty, empty, empty
    sizes = np.bincount(ids)
    if len(sizes) == len(ids):
        ones = np.ones(len(ids), dtype=np.int64)
        return ids, sizes, np.asarray(targets), ones, ones
    if np.all(targets == targets[0]):
        return (
            ids,
            sizes,
            np.full(len(sizes), targets[0], dtype=np.int64),
            sizes,
            np.ones(len(sizes), dtype=np.int64),
        )
    # Target factorization is sorted to preserve canonical-code mode ties.
    target_ids, target_values = pd.factorize(targets, sort=True)
    width = len(target_values)
    if int(ids.max()) > (np.iinfo(np.int64).max - (width - 1)) // width:
        pairs, counts = np.unique(np.column_stack([ids, target_ids]), axis=0, return_counts=True)
        parents, children = pairs.T
    else:
        pairs, counts = np.unique(ids * width + target_ids, return_counts=True)
        parents, children = pairs // width, pairs % width
    starts = np.r_[0, np.flatnonzero(np.diff(parents)) + 1]
    distinct = np.diff(np.r_[starts, len(counts)])
    maxima = np.maximum.reduceat(counts, starts)
    winners = np.minimum.reduceat(
        np.where(counts == np.repeat(maxima, distinct), np.arange(len(counts)), len(counts)), starts
    )
    return ids, sizes, np.asarray(target_values)[children[winners]], maxima, distinct


class PackedMask:
    """Read-only population membership; materialize booleans only when needed."""

    def __init__(self, data, size):
        self.data, self.size = data, size

    def __array__(self, dtype=None, copy=None):
        values = np.unpackbits(np.frombuffer(self.data, dtype=np.uint8), count=self.size).astype(
            bool
        )
        return values.astype(dtype, copy=False) if dtype is not None else values

    def copy(self):
        return np.asarray(self)


class MaskPool:
    def __init__(self):
        self.values = {}

    def intern(self, mask):
        data = np.packbits(mask).tobytes()
        return self.values.setdefault(data, PackedMask(data, len(mask)))


def same_mask(left, right):
    if isinstance(left, PackedMask) and isinstance(right, PackedMask):
        return left.size == right.size and left.data == right.data
    if isinstance(left, PackedMask):
        return left.size == len(right) and left.data == np.packbits(right).tobytes()
    if isinstance(right, PackedMask):
        return same_mask(right, left)
    return np.array_equal(left, right)


class FDCache:
    """Full-population metrics plus bounded exact subset-population reuse."""

    def __init__(self):
        self.global_records = {}
        self.subsets = OrderedDict()
        self.bytes = 0

    def get(self, key, mask):
        full = self.global_records.get(key)
        # _fd_record always intersects a restriction with this pair's global
        # eligibility. Equal counts of nested populations mean equal membership.
        if full is not None and full["evaluated_rows"] == int(mask.sum()):
            return full
        token = (key, np.packbits(mask).tobytes())
        cached = self.subsets.get(token)
        if cached is not None:
            self.subsets.move_to_end(token)
        return cached

    def put(self, key, mask, record, *, global_population=False):
        if global_population:
            self.global_records[key] = record
            return
        token = (key, np.packbits(mask).tobytes())
        if token not in self.subsets:
            self.bytes += len(token[1])
        self.subsets[token] = record
        while len(self.subsets) > 512 or self.bytes > 16 * 1024 * 1024:
            removed, _ = self.subsets.popitem(last=False)
            self.bytes -= len(removed[1])


class EncodedColumns(dict):
    def __init__(self, values, cache=None):
        super().__init__(values)
        self.fd_cache = cache if cache is not None else FDCache()


def first_indices(mask, limit):
    """First matching rows without allocating every matching position."""
    if limit <= 0:
        return np.empty(0, dtype=np.int64)
    if limit >= len(mask):
        return np.flatnonzero(mask)
    pieces = []
    found = 0
    for start in range(0, len(mask), 8192):
        chunk = np.flatnonzero(mask[start : start + 8192])[: limit - found] + start
        pieces.append(chunk)
        found += len(chunk)
        if found >= limit:
            break
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


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
