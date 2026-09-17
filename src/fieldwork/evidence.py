"""Portable findings, position-based inspection, and reusable population scopes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._explore.encoding import MISSING, encode_series, normalize_scalar, validate_frame
from ._explore.result import ExplorerResult


def fingerprint(df: pd.DataFrame) -> str:
    """Identify ordered source values and labels, including duplicate indexes."""
    validate_frame(df)
    digest = hashlib.sha256()
    for values in (df.columns, df.index, *[df[c].array for c in df.columns]):
        digest.update(b"[")
        for value in values:
            digest.update(
                json.dumps(
                    normalize_scalar(value, label=isinstance(value, tuple)).to_dict(),
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            )
            digest.update(b"\n")
        digest.update(b"]")
    return digest.hexdigest()


@dataclass(frozen=True)
class Scope:
    dataset_id: str
    positions: tuple[int, ...]
    name: str = "selection"
    parent: str | None = None

    def __post_init__(self):
        positions = tuple(self.positions)
        if any(
            isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 0 for p in positions
        ):
            raise ValueError("Scope positions must be nonnegative integers")
        if len(set(positions)) != len(positions):
            raise ValueError("Scope positions must not repeat")
        object.__setattr__(self, "positions", tuple(sorted(int(p) for p in positions)))

    @classmethod
    def from_positions(cls, df: pd.DataFrame, positions: Iterable[int], *, name="selection"):
        selected = tuple(positions)
        if any(
            isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 0 or p >= len(df)
            for p in selected
        ):
            raise ValueError("Scope positions must be valid nonnegative row positions")
        if len(set(selected)) != len(selected):
            raise ValueError("Scope positions must not repeat")
        return cls(fingerprint(df), tuple(sorted(int(p) for p in selected)), name)

    def refine(self, df: pd.DataFrame, positions: Iterable[int], *, name="refined"):
        child = Scope.from_positions(df, positions, name=name)
        if child.dataset_id != self.dataset_id or not set(child.positions) <= set(self.positions):
            raise ValueError("Refinement must select positions from its parent scope")
        return Scope(child.dataset_id, child.positions, name, self.name)


class InvestigationResult(ExplorerResult):
    """Saved evidence with dataframe projections and validated source-row inspection."""

    def to_frame(self, section: str = "findings") -> pd.DataFrame:
        return pd.json_normalize(self.payload.get(section, []))

    def inspect(self, df: pd.DataFrame, finding: str | int, *, exceptions: bool = False):
        if fingerprint(df) != self.payload["source"]["dataset_id"]:
            raise ValueError("Source dataset differs from the ordered analysis source")
        records = self.payload["findings"]
        record = (
            records[finding]
            if isinstance(finding, int)
            else next((r for r in records if r["id"] == finding), None)
        )
        if record is None:
            raise KeyError(finding)
        selection = record["exceptions" if exceptions else "examples"]
        return df.iloc[selection["positions"]].copy()

    def recompute(self, df: pd.DataFrame, **overrides):
        """Reapply saved conventions and scope to the same source, with explicit budget overrides."""
        from .availability import missingness
        from .discovery import discover_dependencies
        from .navigation import suggest_paths
        from .patterns import value_patterns

        operations = {
            "missingness": missingness,
            "dependencies": discover_dependencies,
            "paths": suggest_paths,
            "value_patterns": value_patterns,
        }
        if self.kind not in operations:
            raise ValueError(
                "Recompute an individual analysis section, not an overview or comparison"
            )
        if fingerprint(df) != self.payload["source"]["dataset_id"]:
            raise ValueError("Source dataset differs; use a Recipe for a new delivery")
        scope_data = self.payload["scope"]
        scoped = scope_data.get("selection_positions")
        scope = (
            Scope(
                self.payload["source"]["dataset_id"],
                tuple(scoped),
                scope_data["name"],
                scope_data.get("parent"),
            )
            if scoped is not None
            else None
        )
        missing = {
            c: [_restore_scalar(v) for v in values]
            for c, values in self.payload["missing_convention"]["sentinels"].items()
        }
        parameters = {
            **self.payload["parameters"],
            "scope": scope,
            "missing": missing,
            "table_id": self.payload["source"]["table_id"],
            **overrides,
        }
        return operations[self.kind](df, **parameters)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if data.get("schema_version") != "1.0":
            raise ValueError("Unsupported investigation schema version")
        if data["kind"] == "paths":
            from .navigation import PathResult

            cls = PathResult
        return cls(
            data["kind"],
            {k: v for k, v in data.items() if k not in {"kind", "schema_version", "stability"}},
            schema_version="1.0",
        )

    def __repr__(self):
        from .presentation import render_plaintext

        return render_plaintext(self, max_lines=40)

    __str__ = __repr__


def columns(df, selected=None):
    validate_frame(df)
    if not all(isinstance(c, str) for c in df.columns):
        raise TypeError(
            "Discovery requires string column names; foundation operations accept typed labels"
        )
    selected = list(df.columns if selected is None else selected)
    if len(set(selected)) != len(selected):
        raise ValueError("Columns must not repeat")
    for c in selected:
        if c not in df:
            raise KeyError(c)
    return selected


def limit(name, value, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def prepare(df, *, scope=None, missing=None, table_id="table"):
    columns(df)
    if not isinstance(table_id, str) or not table_id:
        raise ValueError("table_id must be a nonempty string")
    identity = fingerprint(df)
    if scope is not None and scope.dataset_id != identity:
        raise ValueError("Scope belongs to a different ordered dataset")
    if scope is not None and any(p >= len(df) for p in scope.positions):
        raise ValueError("Scope positions exceed the source population")
    positions = np.array(scope.positions if scope else range(len(df)), dtype=np.int64)
    frame = df.iloc[positions]
    missing = missing or {}
    columns(df, missing)
    encoded, available = {}, {}
    conventions = {}
    for c in df:
        values, codes = encode_series(frame[c])
        sentinels = {normalize_scalar(v) for v in missing.get(c, [])}

        def sentinel_key(v):
            if v.kind == "integer":
                return ("number", int(v.value))
            if v.kind == "float":
                return ("number", float.fromhex(v.value))
            return v

        sentinel_keys = {sentinel_key(v) for v in sentinels}
        mask = np.array(
            [v != MISSING and sentinel_key(v) not in sentinel_keys for v in values], dtype=bool
        )
        available[c] = mask[codes]
        encoded[c] = codes
        conventions[c] = [v.to_dict() for v in sorted(sentinels, key=lambda v: v.sort_key())]
    base = {
        "status": "computed" if len(frame) else "empty",
        "source": {"dataset_id": identity, "table_id": table_id, "input_rows": len(df)},
        "scope": {
            "name": scope.name if scope else "input",
            "parent": scope.parent if scope else None,
            "evaluated_rows": len(frame),
            "restriction_excluded_rows": len(df) - len(frame),
            "positions_are": "zero_based_source_positions",
            "selection_positions": list(scope.positions) if scope else None,
        },
        "missing_convention": {
            "native_missing": True,
            "sentinels": conventions,
            "numeric_sentinel_equality": True,
        },
        "features": [{"table": table_id, "column": c} for c in df],
        "findings": [],
    }
    return frame, positions, encoded, available, base


def selection(positions, total, example_limit):
    return {
        "positions": [int(p) for p in positions[:example_limit]],
        "total": int(total),
        "omitted": max(0, int(total) - example_limit),
        "method": "first_in_source_order",
        "limit": example_limit,
    }


def finding(
    base,
    kind,
    statement,
    features,
    metrics,
    positions,
    *,
    exceptions=(),
    example_limit=5,
    unit="rows",
    selector=None,
):
    record = {
        "id": f"f{len(base['findings'])}",
        "pattern": kind,
        "statement": statement,
        "features": [{"table": base["source"]["table_id"], "column": c} for c in features],
        "counting_unit": unit,
        "measurements": metrics,
        "examples": selection(positions, len(positions), example_limit),
        "exceptions": selection(exceptions, len(exceptions), example_limit),
        "selector": {
            "dataset_id": base["source"]["dataset_id"],
            "scope_ref": "scope",
            "parameters_ref": "parameters",
            "missing_convention_ref": "missing_convention",
            "finding_id": f"f{len(base['findings'])}",
            **(selector or {}),
        },
    }
    base["findings"].append(record)
    return record


def result(kind, base):
    return InvestigationResult(kind, base, schema_version="1.0")


def _restore_scalar(value):
    kind = value["type"]
    raw = value.get("value")
    if kind == "missing":
        return None
    if kind in {"boolean", "string"}:
        return raw
    if kind == "integer":
        return int(raw)
    if kind == "float":
        return float.fromhex(raw)
    if kind == "date":
        from datetime import date

        return date.fromisoformat(raw)
    if kind in {"datetime_naive", "datetime_aware"}:
        return pd.Timestamp(raw)
    if kind == "timedelta":
        return pd.Timedelta(int(raw), unit="ns")
    raise ValueError(f"Unsupported saved sentinel: {kind}")


def saved_context(base):
    """Restore source-bound scope and typed missing conventions from saved evidence."""
    data = base["scope"]
    positions = data.get("selection_positions")
    return {
        "scope": Scope(
            base["source"]["dataset_id"], tuple(positions), data["name"], data.get("parent")
        )
        if positions is not None
        else None,
        "missing": {
            c: [_restore_scalar(v) for v in values]
            for c, values in base["missing_convention"]["sentinels"].items()
        },
        "table_id": base["source"]["table_id"],
    }


def foundation_context(df, operation, *args, scope=None, missing=None, table_id="table", **options):
    """Normalize a private frame and retain original-source accounting in every derived scope."""
    from copy import deepcopy
    from ._explore.census import _source

    frame, _, _, present, base = prepare(df, scope=scope, missing=missing, table_id=table_id)
    normalized = frame.copy()
    for c in normalized:
        normalized[c] = normalized[c].astype(object).where(present[c], None)
    analysis = operation(normalized, *args, **options)
    payload = deepcopy(analysis.payload)
    excluded = len(df) - len(frame)
    source = {**_source(df), **base["source"]}

    def rebase(value):
        if isinstance(value, dict):
            if "scope_id" in value and "input_rows" in value:
                value["input_rows"] += excluded
                value["restriction_excluded_rows"] += excluded
                value["conditional"] = value["conditional"] or bool(excluded)
                value["lineage"] = [base["scope"]["name"], *value["lineage"]]
            for key, child in list(value.items()):
                if key == "source":
                    value[key] = deepcopy(source)
                else:
                    rebase(child)
        elif isinstance(value, list):
            for child in value:
                rebase(child)

    rebase(payload)
    payload["analysis_context"] = {k: base[k] for k in ("source", "scope", "missing_convention")}
    return ExplorerResult(analysis.kind, payload, schema_version=analysis.schema_version)
