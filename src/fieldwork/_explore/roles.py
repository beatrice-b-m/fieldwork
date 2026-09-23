"""Reviewable schema-role proposals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, Unpack

import pandas as pd

from .._runtime import operation, phase
from ..typing import ColumnLabel, Runtime, SchemaRole
from .census import _source
from .encoding import encode_column, normalize_scalar, validate_frame
from .result import ExplorerResult, KeySpec


@dataclass(frozen=True)
class SchemaProposal:
    """A reviewable role suggestion with the evidence used to form it.

    Parameters
    ----------
    column : dict[str, Any]
        Tagged column identity, not a bare column label.
    proposed_role : str
        'id', 'categorical', 'continuous', or 'unknown'.
    reasons : tuple of dict
        Evidence records, including cardinality, dtype, missing rows, and optional
        column-name hints.
    fd_evidence : mapping or str, optional
        Default 'not_evaluated'. With explicit candidate keys, an evaluated
        status and per-key dependency evidence.

    Attributes
    ----------
    column : dict[str, Any]
        Tagged column identity.
    proposed_role : str
        Suggested role; requires user review.
    reasons : tuple[dict[str, Any], ...]
        Evidence behind the heuristic suggestion.
    fd_evidence : mapping or str
        Optional observed dependency evidence.

    Notes
    -----
    The record is shallowly frozen; nested evidence remains mutable. infer_schema
    returns an ExplorerResult with serialized proposals, not live instances.
    Suggestions do not modify source values or automatically configure analyses.
    """

    column: dict[str, Any]
    proposed_role: SchemaRole
    reasons: tuple[dict[str, Any], ...]
    fd_evidence: Literal["not_evaluated"] | dict[str, Any] = "not_evaluated"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this proposal into ordinary evidence fields.

        Returns
        -------
        dict[str, Any]
            Column identity, proposed_role, reasons, and fd_evidence. The reasons
            tuple becomes a list; nested dictionaries remain shared.
        """
        return {
            "column": self.column,
            "proposed_role": self.proposed_role,
            "reasons": list(self.reasons),
            "fd_evidence": self.fd_evidence,
        }


@operation("schema inference")
def infer_schema(
    df: pd.DataFrame,
    candidate_keys: Iterable[ColumnLabel | KeySpec] | None = None,
    **runtime: Unpack[Runtime],
) -> ExplorerResult:
    """Suggest reviewable column roles without changing analysis settings.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation. Column labels must be unique strings,
        non-boolean integers, or recursively tuple-valued labels. Native missing
        scalars share one identity; integer and float values remain distinct.
        Unsupported column labels or scalar objects raise TypeError.
    candidate_keys : iterable of column labels or KeySpec or None, optional
        Optional explicit determinants for exact dependency evidence; default None
        leaves fd_evidence unevaluated. Composite keys require KeySpec.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    ExplorerResult
        Kind 'schema_proposal', with serialized SchemaProposal records, suggested
        dimensions/keys, source metadata, and warnings. The result is a mapping,
        not a list of SchemaProposal instances.

    Raises
    ------
    KeyError
        A requested column is unknown.
    ValueError
        Columns, limits, thresholds, constraints, or source scope are invalid.
    TypeError
        The frame, column labels, or scalar values are unsupported.
    AnalysisCancelled
        Cancellation or the cooperative timeout stops analysis.

    Notes
    -----
    Roles ('id', 'categorical', 'continuous', 'unknown') are heuristics based on
    observed cardinality and dtype. Name hints are evidence rather than overrides.
    Proposals do not cast values, select dimensions, or establish semantic IDs;
    review them before passing a schema or keys to other analyses.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> proposals = fw.infer_schema(pd.DataFrame({"id": [1, 2, 3]}))
    >>> proposals["proposals"][0]["proposed_role"]
    'id'
    """

    validate_frame(df)
    proposals: list[SchemaProposal] = []
    rows = len(df)
    with phase("schema columns", len(df.columns), "columns") as tracker:
        for column in df.columns:
            series = df[column]
            cardinality = len(encode_column(df, column)[0])
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
            tracker.advance(detail=str(column))
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
