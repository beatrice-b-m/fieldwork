"""Standard-library result models for feature exploration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from ..typing import ColumnLabel

SCHEMA_VERSION = "0.3"


@dataclass(frozen=True, repr=False)
class ExplorerResult(Mapping[str, Any]):
    """A versioned mapping of analytical evidence with shallowly frozen attributes.

    Parameters
    ----------
    kind : str
        Operation kind, such as 'levels', 'census', 'grain', 'pairs',
        'joint_counts', 'schema_proposal', or 'explore'.
    payload : dict, optional
        Analytical sections; default is a new empty dictionary. Usually supplied
        by an analysis rather than constructed manually.
    schema_version : str, optional
        Evidence schema version; default '0.3' for foundation results. This is
        independent of the package version.

    Attributes
    ----------
    kind : str
        Producer operation kind.
    payload : dict[str, Any]
        Mutable nested evidence. Field meanings depend on kind: counts in levels,
        observed prefix tree in census, dependencies/graph in grain, pair
        measurements in pairs, and axis dictionaries/cells in joint_counts.
        Combined explore results contain a sections mapping. Foundation values
        are tagged identities; dictionaries resolve feature and level references.
    schema_version : str
        Serialized evidence schema version.

    Notes
    -----
    Indexing exposes metadata and payload keys: result['kind'] and
    result['scopes'], for example. Frozen attributes do not make nested lists and
    dictionaries immutable. Ordinary exports also share nested containers.
    Generated evidence is compatible with json.dumps(..., allow_nan=False);
    nonfinite scalar values use tagged encodings. Rendering is bounded and does
    not require the original dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.levels(pd.DataFrame({'site': ['A', 'B']}))
    >>> restored = fw.ExplorerResult.from_dict(result.to_dict())
    >>> restored.kind
    'levels'
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self, *, resolve_references: bool = False) -> dict[str, Any]:
        """Export analytical evidence as ordinary or resolved JSON data.

        Parameters
        ----------
        resolve_references : bool, optional
            Default False keeps local IDs. True adds labels and typed values beside
            references in an independent resolved copy; IDs remain intact.

        Returns
        -------
        dict[str, Any]
            Strict-JSON-compatible evidence. The ordinary default export is a new
            top-level dictionary but shares nested containers with the result.

        Notes
        -----
        Both modes retain quantitative evidence; use
        visualization_data(detail='topology') for a structural projection. Restore
        with the corresponding result class's from_dict method. JSON serialization
        converts tuples to lists.

        Examples
        --------
        >>> import json
        >>> import pandas as pd
        >>> import fieldwork as fw
        >>> result = fw.levels(pd.DataFrame({'x': [1, 1]}))
        >>> data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        >>> fw.ExplorerResult.from_dict(data).kind
        'levels'
        """
        data = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            **self.payload,
        }
        if resolve_references:
            from .resolved import resolve_result

            data = resolve_result(data)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Restore foundation evidence from a to_dict export.

        Parameters
        ----------
        data : mapping
            Foundation schema 0.3 export. JSON-decoded input is accepted.
            Discovery schema 1.0 uses InvestigationResult.

        Returns
        -------
        Self
            Restored result. Ordinary nested containers are reused; no source frame
            or original analysis is needed.

        Raises
        ------
        ValueError
            The schema version is unsupported.
        KeyError
            Required export fields are missing.

        Notes
        -----
        This restores saved evidence rather than verifying it against source data.
        It does not validate every nested analytical record or migrate old schemas.
        """
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported foundation schema version")
        return cls(
            data["kind"],
            {k: v for k, v in data.items() if k not in {"kind", "schema_version"}},
            schema_version=data["schema_version"],
        )

    def __repr__(self) -> str:
        from .render import render_plaintext

        return render_plaintext(self, width=100, max_lines=40, max_nodes=100)

    def __str__(self) -> str:
        return repr(self)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        """Use the same bounded, escaped text in IPython and notebooks."""
        printer.text("ExplorerResult(...)" if cycle else repr(self))

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class KeySpec:
    """Declare an explicitly named single-column or composite determinant.

    Parameters
    ----------
    name : str
        Nonempty unique name within one candidate collection.
    columns : tuple of column labels
        Nonempty determinant columns, normalized to a tuple. Supported labels
        are strings, non-boolean integers, or recursively nested tuples.

    Attributes
    ----------
    name : str
        Candidate identifier used in evidence.
    columns : tuple of column labels
        Ordered determinant components; the record is frozen.

    Raises
    ------
    ValueError
        The name or columns are empty. Analysis also rejects repeated or unknown
        components and duplicate candidate names.

    Notes
    -----
    A bare tuple passed as a grain candidate names one tuple-labeled column.
    Use KeySpec to make composite intent explicit. This Python object cannot be
    persisted directly in a Recipe's strict JSON parameters.

    Examples
    --------
    >>> import fieldwork as fw
    >>> key = fw.KeySpec('visit', ('site', 'participant', 'visit_number'))
    >>> key.name
    'visit'
    """

    name: str
    columns: tuple[ColumnLabel, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("KeySpec.name must be a non-empty string")
        columns = tuple(self.columns)
        if not columns:
            raise ValueError("KeySpec.columns must not be empty")
        object.__setattr__(self, "columns", columns)
