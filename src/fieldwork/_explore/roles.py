"""Reviewable schema-role proposals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .census import _source
from .encoding import encode_series, normalize_scalar, validate_frame
from .result import ExplorerResult


@dataclass(frozen=True)
class SchemaProposal:
    column: dict[str, Any]
    proposed_role: str
    reasons: tuple[dict[str, Any], ...]
    fd_evidence: Any = "not_evaluated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "proposed_role": self.proposed_role,
            "reasons": list(self.reasons),
            "fd_evidence": self.fd_evidence,
        }


def infer_schema(df: pd.DataFrame, candidate_keys: Iterable[Any] | None = None) -> ExplorerResult:
    """Suggest roles without silently choosing an analysis configuration."""

    validate_frame(df)
    proposals: list[SchemaProposal] = []
    rows = len(df)
    for column in df.columns:
        series = df[column]
        cardinality = len(encode_series(series)[0])
        ratio = cardinality / rows if rows else 0.0
        name = str(column).lower()
        reasons = (
            {"code": "CARDINALITY", "value": cardinality},
            {"code": "CARDINALITY_RATIO", "value": ratio},
            {"code": "DTYPE", "value": str(series.dtype)},
            {"code": "MISSING_ROWS", "value": int(series.isna().sum())},
        )
        if rows and cardinality == rows:
            role = "id"
        elif pd.api.types.is_numeric_dtype(series.dtype) and cardinality > min(20, rows / 2):
            role = "continuous"
        elif cardinality <= max(20, int(rows * 0.1)):
            role = "categorical"
        else:
            role = "unknown"
        if name.endswith(("id", "_id")):
            reasons = (*reasons, {"code": "NAME_HINT_ID", "value": True})
        proposals.append(
            SchemaProposal(normalize_scalar(column, label=True).to_dict(), role, reasons)
        )
    if candidate_keys is not None:
        from .grain import grain

        evidence = grain(df, candidate_keys).payload["dependencies"]
        by_target: dict[str, list[dict[str, Any]]] = {}
        for dependency in evidence:
            by_target.setdefault(str(dependency["target"]), []).append(dependency)
        proposals = [
            SchemaProposal(
                proposal.column,
                proposal.proposed_role,
                proposal.reasons,
                {
                    "status": "evaluated",
                    "keys": [
                        {
                            "name": dependency["key_name"],
                            "holds": dependency["holds"],
                            "evaluated_groups": dependency["evaluated_groups"],
                        }
                        for dependency in by_target.get(str(proposal.column), [])
                    ],
                },
            )
            for proposal in proposals
        ]
    return ExplorerResult(
        "schema_proposal",
        {
            "status": "empty" if rows == 0 else "computed",
            "source": _source(df),
            "proposals": [proposal.to_dict() for proposal in proposals],
            "suggested_dimensions": [
                proposal.column for proposal in proposals if proposal.proposed_role == "categorical"
            ],
            "suggested_keys": [
                proposal.column for proposal in proposals if proposal.proposed_role == "id"
            ],
            "candidate_keys_received": candidate_keys is not None,
            "warnings": [],
        },
    )
