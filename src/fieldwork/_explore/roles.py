"""Reviewable schema-role proposals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Unpack

import pandas as pd

from .._runtime import operation, phase
from ..result import Result
from ..typing import Runtime, SchemaRole
from .grain import KeySpec, grain, key_specs

if TYPE_CHECKING:
    from ..evidence import Scope


@dataclass(frozen=True)
class SchemaProposal:
    """A reviewable role suggestion with the evidence used to form it.

    Parameters
    ----------
    column : str
        Column name (non-string labels are named by str()).
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
    column : str
        Column name.
    proposed_role : str
        Suggested role; requires user review.
    reasons : tuple[dict[str, Any], ...]
        Evidence behind the heuristic suggestion.
    fd_evidence : mapping or str
        Optional observed dependency evidence.

    Notes
    -----
    The record is shallowly frozen; nested evidence remains mutable. infer_schema
    returns an Result with serialized proposals, not live instances.
    Suggestions do not modify source values or automatically configure analyses.
    """

    column: str
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
    candidate_keys: Iterable[str | KeySpec] | None = None,
    *,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Suggest reviewable column roles from cardinality, dtype and name hints.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    candidate_keys : iterable of str or KeySpec or None, optional
        Keys to test with grain, adding exact dependency evidence to each
        proposal; default None leaves fd_evidence 'not_evaluated'.
    scope, missing, table_id
        Source context shared by every analysis.
    **runtime : Unpack[Runtime]
        Optional runtime controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'schema_proposal': one serialized SchemaProposal per column, plus
        suggested dimensions (categorical) and keys (id).

    Notes
    -----
    Roles are heuristics: every value distinct is 'id'; a numeric dtype with more
    than min(20, rows / 2) values is 'continuous'; at most max(20, 10% of rows)
    values is 'categorical'; otherwise 'unknown'. Nothing is cast or configured.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> proposals = fw.infer_schema(pd.DataFrame({"id": [1, 2, 3]}))
    >>> proposals["proposals"][0]["proposed_role"]
    'id'
    """
    from ..evidence import columns, prepare_values

    names = columns(df)
    frame, _, encoded, base = prepare_values(
        df, names, scope=scope, missing=missing, table_id=table_id
    )
    rows = len(frame)
    proposals = []
    with phase("schema columns", len(names), "columns") as tracker:
        for column in names:
            values, codes = encoded[column]
            proposals.append(_proposal(column, frame[column].dtype, values, codes, rows))
            tracker.advance(detail=column)
    if candidate_keys is not None:
        evidence = grain(
            df, candidate_keys, scope=scope, missing=missing, table_id=table_id
        ).payload["dependencies"]
        for proposal in proposals:
            tests = [d for d in evidence if d["target"] == proposal["column"]]
            proposal["fd_evidence"] = {
                "status": "evaluated",
                "keys": [
                    {
                        "name": d["key_name"],
                        "holds": d["holds"],
                        "evaluated_groups": d["evaluated_groups"],
                    }
                    for d in tests
                ],
            }
    base["parameters"] = {
        "candidate_keys": None
        if candidate_keys is None
        else [
            {"name": s.name, "columns": list(s.columns)} for s in key_specs(frame, candidate_keys)
        ]
    }
    base.update(
        proposals=proposals,
        suggested_dimensions=[
            p["column"] for p in proposals if p["proposed_role"] == "categorical"
        ],
        suggested_keys=[p["column"] for p in proposals if p["proposed_role"] == "id"],
        warnings=[],
    )
    return Result("schema_proposal", base)


def _proposal(column: str, dtype: Any, values: list[Any], codes: Any, rows: int) -> dict:
    cardinality = len(values)
    missing_rows = int((codes == len(values) - 1).sum()) if values and values[-1] is None else 0
    reasons: tuple[dict[str, Any], ...] = (
        {"code": "CARDINALITY", "value": cardinality},
        {"code": "CARDINALITY_RATIO", "value": cardinality / rows if rows else 0.0},
        {"code": "DTYPE", "value": str(dtype)},
        {"code": "MISSING_ROWS", "value": missing_rows},
    )
    if rows and cardinality == rows:
        role = "id"
    elif pd.api.types.is_numeric_dtype(dtype) and cardinality > min(20, rows / 2):
        role = "continuous"
    elif cardinality <= max(20, int(rows * 0.1)):
        role = "categorical"
    else:
        role = "unknown"
    if column.lower().endswith(("id", "_id")):
        reasons = (*reasons, {"code": "NAME_HINT_ID", "value": True})
    return SchemaProposal(column, role, reasons).to_dict()
