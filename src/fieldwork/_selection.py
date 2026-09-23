"""Evaluate one saved selector without replaying unrelated findings or graphs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._explore._kernels import group_ids, modal_groups
from ._explore.encoding import json_value, same_json
from ._runtime import phase
from .evidence import normalized_encoding, prepare, saved_context


def select_rows(df, analysis, record, exceptions):
    pattern = record["pattern"]
    supported = {
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
        "exact_dependency",
        "approximate_dependency",
        "string_patterns",
        "numeric_range",
        "indexed_family",
        "numeric_offset",
        "numeric_ratio",
        "context_constancy",
    }
    if pattern not in supported:
        return None
    params = analysis["parameters"]
    features = [f["column"] for f in record["features"]]
    entities = params.get("entity", [])
    contexts = params.get("by", [])
    frame, positions, codes, present, _ = prepare(
        df, features=[*features, *entities, *contexts], **saved_context(analysis)
    )
    n = len(frame)
    empty = np.empty(0, dtype=np.int64)
    context = record.get("structure", {}).get("context", {})
    row_mask = np.ones(n, dtype=bool)
    if context:
        encoded = normalized_encoding(frame, {c: codes[c] for c in context}, present)
        for c, saved in context.items():
            values, dense = encoded[c]
            code = next((i for i, v in enumerate(values) if same_json(json_value(v), saved)), None)
            row_mask &= dense == code if code is not None else False
    with phase("selecting matches"):
        if pattern in {"exact_dependency", "approximate_dependency"}:
            metrics = record["measurements"]
            key, target = metrics["determinant"], metrics["target"]
            if params["dropna"]:
                for c in [*key, target]:
                    row_mask &= present[c]
            rows = np.flatnonzero(row_mask)
            key_ids = group_ids(np.where(present[c][rows], codes[c][rows], -1) for c in key)
            targets = np.where(present[target][rows], codes[target][rows], -1)
            groups, _, modes, _, _ = modal_groups(key_ids, targets)
            good = targets == modes[groups]
            return positions[rows[~good if exceptions else good]]
        if pattern == "context_constancy":
            for c in contexts:
                row_mask &= present[c]
            rows = np.flatnonzero(row_mask)
            ids = group_ids(codes[c][rows] for c in contexts)
            c = features[-1]
            populated = present[c][rows]
            valid = pd.unique(ids[populated])
            _, _, _, _, distinct = modal_groups(ids[populated], codes[c][rows[populated]])
            groups = valid[distinct > 1 if exceptions else distinct == 1]
            return positions[rows[np.isin(ids, groups)]]
        if pattern in {
            "string_patterns",
            "numeric_range",
            "indexed_family",
            "numeric_offset",
            "numeric_ratio",
        }:
            if exceptions:
                return empty
            if pattern == "indexed_family":
                return positions.copy()
            if pattern in {"string_patterns", "numeric_range"}:
                if pattern == "numeric_range" and (
                    not pd.api.types.is_numeric_dtype(frame[features[0]].dtype)
                    or pd.api.types.is_bool_dtype(frame[features[0]].dtype)
                ):
                    raise ValueError("Saved finding does not resolve to one matching population")
                return positions[present[features[0]]]
            a, b = features
            if not all(pd.api.types.is_numeric_dtype(frame[c].dtype) for c in (a, b)):
                raise ValueError("Saved finding does not resolve to one matching population")
            rows = np.flatnonzero(present[a] & present[b])
            x, y = (frame[c].iloc[rows].to_numpy(dtype=float) for c in (a, b))
            valid = np.isfinite(x) & np.isfinite(y)
            if pattern == "numeric_ratio":
                valid &= x != 0
            return positions[rows[valid]]

        entity_pattern = pattern in {"entity_availability", "entity_summary"}
        entity_mode = entity_pattern or params.get("unit") == "entities"
        if entity_mode:
            for c in entities:
                row_mask &= present[c]
        rows = np.flatnonzero(row_mask)
        ids = group_ids(codes[c][rows] for c in entities) if entity_mode else None
        sizes = np.bincount(ids) if entity_mode else None
        count = len(sizes) if entity_mode else len(rows)

        def counts(c):
            return np.bincount(ids, weights=present[c][rows], minlength=count).astype(np.int64)

        def availability(c):
            if not entity_mode:
                return present[c][rows]
            populated = counts(c)
            return (
                populated > 0
                if params.get("entity_presence", "any") == "any"
                else populated == sizes
            )

        if pattern in {"availability", "context_availability"}:
            selected = availability(record["selector"].get("feature", features[0]))
            if exceptions:
                selected = ~selected
        elif pattern in {"context_summary", "entity_summary"}:
            selected = np.zeros(count, dtype=bool) if exceptions else np.ones(count, dtype=bool)
        elif pattern == "entity_availability":
            if exceptions:
                return empty
            selector = record["selector"]
            values = counts(selector["feature"])
            selected = {
                "any": values > 0,
                "all": values == sizes,
                "one": values == 1,
                "some": (values > 0) & (values < sizes),
                "none": values == 0,
            }[selector["presence_pattern"]]
        elif pattern == "availability_signature":
            if exceptions:
                return empty
            selected = np.ones(count, dtype=bool)
            for c in record["structure"]["present"]:
                selected &= availability(c)
            for c in record["structure"]["absent"]:
                selected &= ~availability(c)
        elif pattern == "availability_family":
            if exceptions:
                return empty
            selected = availability(features[0])
        else:
            a, b = (availability(c) for c in features)
            if pattern == "presence_implication":
                selected = a & ~b if exceptions else a & b
            elif pattern == "similar_availability":
                selected = a != b if exceptions else a & b
            else:
                selected = np.zeros(count, dtype=bool) if exceptions else a | b
        return positions[rows[selected[ids] if entity_mode else selected]]
