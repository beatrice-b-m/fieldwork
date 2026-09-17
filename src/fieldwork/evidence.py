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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if data.get("schema_version") != "1.0":
            raise ValueError("Unsupported investigation schema version")
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
        "selector": {"dataset_id": base["source"]["dataset_id"], **(selector or {})},
    }
    base["findings"].append(record)
    return record


def result(kind, base):
    return InvestigationResult(kind, base, schema_version="1.0")
