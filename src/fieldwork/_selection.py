"""Evaluate one saved selector without replaying unrelated findings or graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._explore._kernels import group_ids, modal_groups
from ._explore.encoding import json_value, same_json
from ._runtime import phase
from .evidence import normalized_encoding, prepare, saved_context

_AVAILABILITY = {
    "availability",
    "availability_signature",
    "availability_family",
    "similar_availability",
    "presence_implication",
    "mutually_exclusive",
    "context_availability",
    "context_summary",
    "entity_availability",
    "entity_summary",
}
_VALUES = {"string_patterns", "numeric_range", "indexed_family", "numeric_offset", "numeric_ratio"}
_DEPENDENCIES = {"exact_dependency", "approximate_dependency"}


@dataclass
class _Rows:
    """The verified source, prepared under a saved analysis's scope and conventions."""

    record: dict[str, Any]
    parameters: dict[str, Any]
    features: list[str]
    frame: pd.DataFrame
    positions: np.ndarray
    codes: dict[str, np.ndarray]
    present: dict[str, np.ndarray]
    mask: np.ndarray


def select_rows(df, analysis, record, exceptions):
    """Source positions matching a saved finding, or None when replay is needed."""
    pattern = record["pattern"]
    if pattern not in _AVAILABILITY | _VALUES | _DEPENDENCIES | {"context_constancy"}:
        return None
    rows = _prepare(df, analysis, record)
    with phase("selecting matches"):
        if pattern in _DEPENDENCIES:
            return _dependency(rows, exceptions)
        if pattern == "context_constancy":
            return _constancy(rows, exceptions)
        if pattern in _VALUES:
            return _values(rows, pattern, exceptions)
        return _availability(rows, pattern, exceptions)


def _prepare(df, analysis, record) -> _Rows:
    parameters = analysis["parameters"]
    features = [f["column"] for f in record["features"]]
    needed = [*features, *parameters.get("entity", []), *parameters.get("by", [])]
    frame, positions, codes, present, _ = prepare(df, features=needed, **saved_context(analysis))
    mask = np.ones(len(frame), dtype=bool)
    context = record.get("structure", {}).get("context", {})
    if context:
        encoded = normalized_encoding(frame, {c: codes[c] for c in context}, present)
        for c, saved in context.items():
            values, dense = encoded[c]
            code = next((i for i, v in enumerate(values) if same_json(json_value(v), saved)), None)
            mask &= dense == code if code is not None else False
    return _Rows(record, parameters, features, frame, positions, codes, present, mask)


def _dependency(rows: _Rows, exceptions: bool) -> np.ndarray:
    """Rows keeping (or, as exceptions, repairing) their group's modal target."""
    metrics = rows.record["measurements"]
    key, target = metrics["determinant"], metrics["target"]
    mask, present, codes = rows.mask, rows.present, rows.codes
    if rows.parameters["dropna"]:
        for c in [*key, target]:
            mask = mask & present[c]
    selected = np.flatnonzero(mask)
    key_ids = group_ids(np.where(present[c][selected], codes[c][selected], -1) for c in key)
    targets = np.where(present[target][selected], codes[target][selected], -1)
    groups, _, modes, _, _ = modal_groups(key_ids, targets)
    good = targets == modes[groups]
    return rows.positions[selected[~good if exceptions else good]]


def _constancy(rows: _Rows, exceptions: bool) -> np.ndarray:
    """Rows of contexts where the feature is constant (or, as exceptions, varies)."""
    mask = rows.mask
    contexts = rows.parameters.get("by", [])
    for c in contexts:
        mask = mask & rows.present[c]
    selected = np.flatnonzero(mask)
    ids = group_ids(rows.codes[c][selected] for c in contexts)
    c = rows.features[-1]
    populated = rows.present[c][selected]
    valid = pd.unique(ids[populated])
    _, _, _, _, distinct = modal_groups(ids[populated], rows.codes[c][selected[populated]])
    groups = valid[distinct > 1 if exceptions else distinct == 1]
    return rows.positions[selected[np.isin(ids, groups)]]


def _values(rows: _Rows, pattern: str, exceptions: bool) -> np.ndarray:
    """Rows a value summary describes; these findings have no exceptions."""
    from .patterns import numbers

    if exceptions:
        return np.empty(0, dtype=np.int64)
    if pattern == "indexed_family":
        return rows.positions.copy()
    present, features = rows.present, rows.features
    numeric = {c: numbers(rows.frame[c], present[c]) for c in features}
    if pattern != "string_patterns" and any(value is None for value in numeric.values()):
        raise ValueError("Saved finding does not resolve to one matching population")
    if pattern in {"string_patterns", "numeric_range"}:
        return rows.positions[present[features[0]]]
    a, b = features
    x, y = numeric[a], numeric[b]
    valid = np.isfinite(x) & np.isfinite(y) & present[a] & present[b]
    if pattern == "numeric_ratio":
        valid &= x != 0
    return rows.positions[valid]


def _availability(rows: _Rows, pattern: str, exceptions: bool) -> np.ndarray:
    """Rows (or all rows of the entities) matching an availability predicate."""
    record, parameters = rows.record, rows.parameters
    entities = parameters.get("entity", [])
    entity_mode = pattern in {"entity_availability", "entity_summary"}
    entity_mode = entity_mode or parameters.get("unit") == "entities"
    mask = rows.mask
    for c in entities if entity_mode else ():
        mask = mask & rows.present[c]
    selected = np.flatnonzero(mask)
    ids = group_ids(rows.codes[c][selected] for c in entities) if entity_mode else None
    sizes = np.bincount(ids) if ids is not None else None
    count = len(sizes) if sizes is not None else len(selected)

    def counts(c):
        return np.bincount(ids, weights=rows.present[c][selected], minlength=count).astype(np.int64)

    def available(c):
        if not entity_mode:
            return rows.present[c][selected]
        populated = counts(c)
        any_row = parameters.get("entity_presence", "any") == "any"
        return populated > 0 if any_row else populated == sizes

    matches = _predicate(
        record, pattern, exceptions, rows.features, count, counts, sizes, available
    )
    if matches is None:
        return np.empty(0, dtype=np.int64)
    return rows.positions[selected[matches[ids] if entity_mode else matches]]


def _predicate(record, pattern, exceptions, features, count, counts, sizes, available):
    """Units matching a saved availability finding; None when it has no exceptions."""
    if pattern in {"availability", "context_availability"}:
        matches = available(record["selector"].get("feature", features[0]))
        return ~matches if exceptions else matches
    if pattern in {"context_summary", "entity_summary"}:
        return np.zeros(count, dtype=bool) if exceptions else np.ones(count, dtype=bool)
    if exceptions and pattern in {
        "entity_availability",
        "availability_signature",
        "availability_family",
    }:
        return None
    if pattern == "entity_availability":
        values = counts(record["selector"]["feature"])
        return {
            "any": values > 0,
            "all": values == sizes,
            "one": values == 1,
            "some": (values > 0) & (values < sizes),
            "none": values == 0,
        }[record["selector"]["presence_pattern"]]
    if pattern == "availability_signature":
        matches = np.ones(count, dtype=bool)
        for c in record["structure"]["present"]:
            matches &= available(c)
        for c in record["structure"]["absent"]:
            matches &= ~available(c)
        return matches
    if pattern == "availability_family":
        return available(features[0])
    a, b = (available(c) for c in features)
    if pattern == "presence_implication":
        return a & ~b if exceptions else a & b
    if pattern == "similar_availability":
        return a != b if exceptions else a & b
    return np.zeros(count, dtype=bool) if exceptions else a | b
