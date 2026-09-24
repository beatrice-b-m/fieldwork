"""The one result model shared by every analysis."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self, Unpack

import pandas as pd

from ._runtime import operation
from .typing import Runtime, Side

if TYPE_CHECKING:
    from .evidence import Scope
    from .navigation import Path

SCHEMA_VERSION = "2.0"


def overview_findings(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """An overview's ranked leads, resolved to the section findings they reference.

    Each record is the section's finding with the lead's ID and rank, and a
    selector naming its section, so it can be inspected through the overview.
    """
    sections = payload["sections"]
    found = {
        (name, record["id"]): record
        for name, section in sections.items()
        for record in section.get("findings", [])
    }
    output = []
    for lead in payload.get("leads", []):
        name = lead["section"]
        record = found[(name, lead["finding_id"])]
        output.append(
            {
                **record,
                "id": lead["id"],
                "lead": lead["lead"],
                "selector": {
                    **record["selector"],
                    "analysis_section": name,
                    "scope_ref": f"sections.{name}.scope",
                    "parameters_ref": f"sections.{name}.parameters",
                    "missing_convention_ref": f"sections.{name}.missing_convention",
                },
            }
        )
    return output


@dataclass(frozen=True, repr=False)
class Result(Mapping[str, Any]):
    """Saved evidence from any analysis, browsable without the source frame.

    Parameters
    ----------
    kind : str
        Producing analysis: 'levels', 'census', 'grain', 'pairs', 'joint_counts',
        'schema_proposal', 'profile', 'missingness', 'dependencies', 'paths',
        'value_patterns', 'overview', 'comparison' or 'relation'.
    payload : dict, optional
        JSON-compatible evidence (source identity, scope, parameters and
        kind-specific records); default empty. Analyses and from_dict build it.

    Notes
    -----
    Indexing reads payload keys (``result["scope"]``); ``schema_version`` is the
    export format, '2.0'. Methods that read source rows (inspect, select,
    recompute) first verify the identical ordered source; a relation's left
    source is ``df`` and its right source is passed as ``right``. Overviews and profiles
    hold their parts in ``sections``; ``section(name)`` returns one as a Result.
    See docs/contracts.md for the payload fields.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> fw.Result.from_dict(result.to_dict()).kind
    'missingness'
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: ClassVar[str] = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Export the evidence as strict-JSON-compatible data.

        Returns
        -------
        dict[str, Any]
            ``schema_version``, ``kind`` and the payload fields, in a new top-level
            dictionary whose nested containers are shared with the result.
        """
        return {"schema_version": self.schema_version, "kind": self.kind, **self.payload}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Restore a result saved with to_dict (schema 2.0).

        Parameters
        ----------
        data : mapping
            A to_dict export, JSON-decoded or not; nested containers are reused.

        Returns
        -------
        Result
            The saved evidence; no source frame is needed.

        Raises
        ------
        ValueError
            The schema version is not 2.0.
        """
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported result schema {data.get('schema_version')!r}")
        payload = {k: v for k, v in data.items() if k not in {"kind", "schema_version"}}
        return cls(data["kind"], payload)

    def section(self, name: str) -> Result:
        """Return one part of a composite result (overview or profile).

        Parameters
        ----------
        name : str
            Section name, such as 'dependencies' or 'census'.

        Returns
        -------
        Result
            The saved section, sharing its containers with this result.

        Raises
        ------
        KeyError
            The section does not exist.
        ValueError
            The section was not requested.
        """
        data = self.payload.get("sections", {})[name]
        if data.get("status") == "not_requested":
            raise ValueError(f"Section {name!r} was not requested")
        return Result.from_dict(data)

    @property
    def findings(self) -> list[dict[str, Any]]:
        """Findings in rank order, each with an ID usable by inspect and select.

        An overview stores ranked references to its sections' findings; this
        resolves them. Other kinds return their ``findings`` list (empty for
        analyses that report measurements only).
        """
        if self.kind == "overview":
            return overview_findings(self.payload)
        return self.payload.get("findings", [])

    def to_frame(self, section: str = "findings") -> pd.DataFrame:
        """Tabulate a saved list, such as findings, availability or dependencies.

        Parameters
        ----------
        section : str, optional
            Payload list to project; default 'findings'. A missing list gives an
            empty frame.

        Returns
        -------
        pandas.DataFrame
            ``pandas.json_normalize`` of the records (nested fields become dotted
            columns). This is evidence, not source rows.
        """
        records = self.findings if section == "findings" else self.payload.get(section, [])
        return pd.json_normalize(records)

    def relationships(
        self, feature: str | None = None, *, kinds: Iterable[str] | None = None
    ) -> pd.DataFrame:
        """Tabulate saved feature connections and the findings supporting them.

        Parameters
        ----------
        feature : str or None, optional
            Keep connections involving this column; default None keeps all.
        kinds : iterable of str or None, optional
            Keep these relationship kinds; default None keeps all.

        Returns
        -------
        pandas.DataFrame
            Feature-network relationships; empty when none are saved. A connection
            is a lead to inspect, not an equivalence or a composed dependency.
        """
        records = self.payload.get("feature_network", {}).get("relationships", [])
        return pd.json_normalize(
            [
                record
                for record in records
                if (feature is None or any(f["column"] == feature for f in record["features"]))
                and (kinds is None or record["kind"] in kinds)
            ]
        )

    @property
    def best(self) -> Path | None:
        """The top-ranked census path of a paths result, or None if there is none."""
        return self.path(0) if self._paths()["paths"] else None

    def path(self, index: int = 0) -> Path:
        """Return a ranked census recommendation with its source context.

        Parameters
        ----------
        index : int, optional
            Position among the ranked paths; default 0. Negative indexes count
            from the end.

        Returns
        -------
        Path
            Dimensions plus the saved source, scope and missing conventions, so
            ``path.census(df)`` evaluates the same population.

        Raises
        ------
        ValueError
            The result holds no path recommendations.
        IndexError
            The path does not exist.
        """
        from .navigation import Path

        paths = self._paths()
        return Path(paths["paths"][index]["dimensions"], paths)

    def _paths(self) -> Mapping[str, Any]:
        if self.kind == "paths":
            return self.payload
        if self.kind == "overview" and "paths" in self.payload.get("sections", {}):
            return self.section("paths").payload
        raise ValueError(f"A {self.kind} result has no census path recommendations")

    def _finding(
        self,
        df: pd.DataFrame,
        finding: str | int,
        side: str = "left",
        right: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        from .evidence import fingerprint

        if self.kind != "relation" and (side != "left" or right is not None):
            raise ValueError("side and right apply to relation results only")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if fingerprint(df) != self.payload["source"]["dataset_id"]:
            raise ValueError("Source dataset differs from the ordered analysis source")
        records = self.findings
        record = (
            records[finding]
            if isinstance(finding, int)
            else next((r for r in records if r["id"] == finding), None)
        )
        if record is None:
            raise KeyError(finding)
        return record

    @operation("inspection")
    def inspect(
        self,
        df: pd.DataFrame,
        finding: str | int,
        *,
        exceptions: bool = False,
        all_matches: bool = False,
        side: Side = "left",
        right: pd.DataFrame | None = None,
        **runtime: Unpack[Runtime],
    ) -> pd.DataFrame:
        """Return a finding's source rows, as a copy of the original frame's rows.

        Parameters
        ----------
        df : pandas.DataFrame
            The identical ordered source (labels, index and values are verified).
        finding : str or int
            Finding ID such as 'f0', or a position in ``findings``.
        exceptions : bool, optional
            Default False returns supporting rows; True returns counterexamples.
        all_matches : bool, optional
            Default False returns the saved examples (at most example_limit);
            True recovers the complete matching population.
        side : {'left', 'right'}, optional
            Relation results only: the table whose rows are returned; default
            'left'.
        right : pandas.DataFrame or None, optional
            Relation results between two tables only: the identical right
            source, needed for right rows and for ``all_matches``.
        **runtime : Unpack[Runtime]
            Optional runtime controls; see fieldwork.typing.Runtime.

        Returns
        -------
        pandas.DataFrame
            Source rows in source order, selected by position (``iloc``).

        Raises
        ------
        ValueError
            The source differs, or a saved selector cannot resolve its population.
        KeyError, IndexError
            The finding ID or position does not exist.
        """
        record = self._finding(df, finding, side, right)
        if self.kind == "relation":
            from .relate import right_source, saved_rows

            source = df if side == "left" else right_source(self.payload, df, right)
            if all_matches:
                selected = self.select(df, finding, exceptions=exceptions, side=side, right=right)
                return source.iloc[list(selected.positions)].copy()
            return source.iloc[saved_rows(record, side, exceptions)].copy()
        if all_matches:
            return df.iloc[list(self.select(df, finding, exceptions=exceptions).positions)].copy()
        return df.iloc[record["exceptions" if exceptions else "examples"]["positions"]].copy()

    @operation("selection")
    def select(
        self,
        df: pd.DataFrame,
        finding: str | int,
        *,
        exceptions: bool = False,
        name: str = "finding selection",
        side: Side = "left",
        right: pd.DataFrame | None = None,
        **runtime: Unpack[Runtime],
    ) -> Scope:
        """Recover a finding's complete matching source population as a Scope.

        Parameters
        ----------
        df : pandas.DataFrame
            The identical ordered source (labels, index and values are verified).
        finding : str or int
            Finding ID such as 'f0', or a position in ``findings``.
        exceptions : bool, optional
            Default False selects supporting rows; True selects counterexamples.
        name : str, optional
            Name of the returned scope; default 'finding selection'.
        side : {'left', 'right'}, optional
            Relation results only: the table to select from; default 'left'.
            The Scope belongs to that table.
        right : pandas.DataFrame or None, optional
            Relation results between two tables only: the identical right
            source, required because matching needs both tables.
        **runtime : Unpack[Runtime]
            Optional runtime controls; see fieldwork.typing.Runtime.

        Returns
        -------
        Scope
            Every matching source position, under the saved scope and missing
            conventions, whatever the saved example limit.

        Raises
        ------
        ValueError
            The source differs, or a saved selector cannot resolve its population.
        KeyError, IndexError
            The finding ID or position does not exist.

        Examples
        --------
        >>> import pandas as pd
        >>> import fieldwork as fw
        >>> df = pd.DataFrame({'x': [1, 2, None]})
        >>> fw.missingness(df, limits={"example_limit": 1}).select(df, 0).positions
        (0, 1)
        """
        from ._selection import select_rows
        from .evidence import Scope

        record = self._finding(df, finding, side, right)
        if self.kind == "relation":
            from .relate import matching_rows, right_source

            other = right_source(self.payload, df, right)
            source = self.payload["sides"][side]
            positions = matching_rows(self.payload, record, df, other, side, exceptions)
            return Scope(
                source["source"]["dataset_id"], tuple(positions), name, source["scope"]["name"]
            )
        selector = record["selector"]
        analysis = self
        if self.kind == "overview":
            analysis = self.section(selector["analysis_section"])
        dataset = self.payload["source"]["dataset_id"]
        parent = analysis["scope"]["name"]
        if analysis.kind == "paths":
            selected = [] if exceptions else analysis["scope"].get("selection_positions")
            return Scope(
                dataset, tuple(selected if selected is not None else range(len(df))), name, parent
            )
        selected = select_rows(df, analysis, record, exceptions)
        if selected is not None:
            return Scope(dataset, tuple(selected), name, parent)
        replay = analysis.recompute(df, limits={"example_limit": len(df)})
        # Match semantic selectors, not ordinal IDs, which depend on parameters.
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
        positions = matches[0]["exceptions" if exceptions else "examples"]["positions"]
        return Scope(dataset, tuple(positions), name, parent)

    @operation("recomputation")
    def recompute(self, df: pd.DataFrame, **overrides: Any) -> Result:
        """Rerun this analysis on its verified source, optionally with new options.

        Parameters
        ----------
        df : pandas.DataFrame
            The identical ordered source (labels, index and values are verified).
        **overrides : Any
            Options replacing saved parameters, such as
            ``limits={"example_limit": 20}`` (budgets merge with the saved ones),
            and the runtime controls of fieldwork.typing.Runtime. A relation
            between two tables also needs ``right``, its identical right source.

        Returns
        -------
        Result
            A new result under the saved scope, missing conventions and table ID.
            To analyze a new delivery, use a Recipe.

        Raises
        ------
        ValueError
            The source differs, or the kind cannot be recomputed (overview,
            profile, comparison).
        TypeError
            An override is not accepted by the analysis.
        """
        from .evidence import fingerprint, saved_context
        from .workflow import Recipe

        name = {"schema_proposal": "infer_schema", "relation": "relate"}.get(self.kind, self.kind)
        operations = Recipe.operations()
        if name not in operations or "parameters" not in self.payload:
            raise ValueError(f"A {self.kind} result cannot be recomputed; recompute its sections")
        if fingerprint(df) != self.payload["source"]["dataset_id"]:
            raise ValueError("Source dataset differs; use a Recipe for a new delivery")
        saved = self.payload["parameters"]
        if self.kind == "relation":
            from .relate import right_source
            from .relate import saved_context as relation_context

            right = right_source(self.payload, df, overrides.pop("right", None))
            own = None if self.payload["self_reference"] else right
            context = {**relation_context(self.payload), "right": own}
        else:
            context = saved_context(self.payload)
        parameters = {**saved, **context, **overrides}
        if isinstance(saved.get("limits"), Mapping) and isinstance(
            overrides.get("limits"), Mapping
        ):
            # A budget override keeps the other saved budgets.
            parameters["limits"] = {**saved["limits"], **overrides["limits"]}
        return operations[name](df, **parameters)

    def __repr__(self) -> str:
        from .presentation import render_plaintext

        return render_plaintext(self, max_lines=40)

    def __str__(self) -> str:
        return repr(self)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        """Use the same bounded, escaped text in IPython and notebooks."""
        printer.text("Result(...)" if cycle else repr(self))

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())
