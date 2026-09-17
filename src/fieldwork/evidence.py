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

    def relationships(self, feature=None, *, kinds=None):
        """Browse typed feature connections, with finding IDs for evidence inspection."""
        records = self.payload.get("feature_network", {}).get("relationships", [])
        return pd.json_normalize(
            [
                record
                for record in records
                if (feature is None or any(f["column"] == feature for f in record["features"]))
                and (kinds is None or record["kind"] in kinds)
            ]
        )

    def _finding(self, df, finding):
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
        return record

    def inspect(self, df: pd.DataFrame, finding: str | int, *, exceptions=False, all_matches=False):
        """Return saved examples, or recompute the complete matching source population."""
        record = self._finding(df, finding)
        if all_matches:
            return df.iloc[list(self.select(df, finding, exceptions=exceptions).positions)].copy()
        return df.iloc[record["exceptions" if exceptions else "examples"]["positions"]].copy()

    def select(self, df, finding, *, exceptions=False, name="finding selection"):
        """Recover all matching source positions as a reusable, source-bound Scope."""
        record = self._finding(df, finding)
        selector = record["selector"]
        analysis = self
        if self.kind == "overview":
            analysis = InvestigationResult.from_dict(self["sections"][selector["analysis_section"]])
        if analysis.kind == "paths":
            selected = [] if exceptions else analysis["scope"].get("selection_positions")
            return Scope(
                self["source"]["dataset_id"],
                tuple(selected if selected is not None else range(len(df))),
                name,
                analysis["scope"]["name"],
            )
        replay = analysis.recompute(df, example_limit=len(df))
        # Match semantic selectors, not ordinal IDs: older saved results can have
        # different finding orders after new evidence types are introduced.
        bookkeeping = {
            "dataset_id",
            "scope_ref",
            "parameters_ref",
            "missing_convention_ref",
            "finding_id",
            "analysis_section",
        }
        predicate = {k: v for k, v in selector.items() if k not in bookkeeping}
        matches = [
            f
            for f in replay["findings"]
            if f["pattern"] == record["pattern"]
            and (predicate or f["features"] == record["features"])
            and all(f["selector"].get(k) == v for k, v in predicate.items())
        ]
        if len(matches) != 1:
            raise ValueError("Saved finding does not resolve to one matching population")
        complete = matches[0]
        positions = complete["exceptions" if exceptions else "examples"]["positions"]
        return Scope(
            self["source"]["dataset_id"], tuple(positions), name, analysis["scope"]["name"]
        )

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
        result_class = cls
        if data["kind"] == "paths":
            from .navigation import PathResult

            result_class = PathResult
        return result_class(
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
    structure=None,
):
    record = {
        "id": f"f{len(base['findings'])}",
        "pattern": kind,
        "statement": statement,
        "features": [{"table": base["source"]["table_id"], "column": c} for c in features],
        "counting_unit": unit,
        "structure": structure or {},
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
    frame, _, _, present, base = prepare(df, scope=scope, missing=missing, table_id=table_id)
    normalized = frame.copy()
    for c in normalized:
        normalized[c] = normalized[c].astype(object).where(present[c], None)
    analysis = operation(normalized, *args, **options)
    return contextual_result(analysis, df, base)


def contextual_result(analysis, df, base):
    """Attach discovery lineage to a foundation result computed on its prepared frame."""
    from copy import deepcopy

    from ._explore.census import _source

    payload = deepcopy(analysis.payload)
    excluded = base["scope"]["restriction_excluded_rows"]
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


def context_statement(context):
    return ", ".join(
        f"{feature} = {_restore_scalar(value)!r} ({value['type']})"
        for feature, value in context.items()
    )
