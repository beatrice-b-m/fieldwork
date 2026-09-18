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
from ._runtime import checkpoint, current_session, operation, phase
from .progress import CancellationToken, Progress


def fingerprint(df: pd.DataFrame) -> str:
    """Identify ordered source values and labels, including duplicate indexes."""
    validate_frame(df)
    session = current_session()
    if session and id(df) in session.fingerprints:
        checkpoint()
        return session.fingerprints[id(df)][1]
    with phase("fingerprinting", len(df.columns), "columns") as tracker:
        identity = _fingerprint(df, tracker)
    if session:
        session.fingerprints[id(df)] = (df, identity)
    return identity


def _fingerprint(df, progress):
    digest = hashlib.sha256()
    for values in (df.columns, df.index):
        digest.update(b"[")
        for i, value in enumerate(values):
            if i % 8192 == 0:
                checkpoint()
            digest.update(
                json.dumps(
                    normalize_scalar(value, label=isinstance(value, tuple)).to_dict(),
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            )
            digest.update(b"\n")
        digest.update(b"]")
    # Bound allocation even for continuous/unique columns. Serialize canonical
    # values once per chunk; preserve the original byte stream and saved IDs.
    for column in df:
        digest.update(b"[")
        for start in range(0, len(df), 8192):
            checkpoint()
            chunk = df[column].iloc[start : start + 8192]
            try:
                values, codes = encode_series(chunk)
            except TypeError:
                # Fingerprints historically allow tuple-valued labels/cells even
                # where the analytical scalar encoder rejects tuple cells.
                serialized = [
                    json.dumps(
                        normalize_scalar(v, label=isinstance(v, tuple)).to_dict(),
                        sort_keys=True,
                        allow_nan=False,
                    ).encode()
                    + b"\n"
                    for v in chunk.array
                ]
            else:
                dictionary = np.array(
                    [
                        json.dumps(v.to_dict(), sort_keys=True, allow_nan=False).encode() + b"\n"
                        for v in values
                    ],
                    dtype=object,
                )
                serialized = dictionary[codes].tolist()
            digest.update(b"".join(serialized))
        digest.update(b"]")
        progress.advance(detail=str(column))
    return digest.hexdigest()


@dataclass(frozen=True)
class Scope:
    """Identify a reusable population by positions in one ordered source frame.

    Parameters
    ----------
    dataset_id : str
        Canonical source fingerprint. Prefer from_positions to compute it.
    positions : tuple of int
        Unique nonnegative source row positions, sorted into source order.
        from_positions additionally validates bounds against the dataframe.
    name : str, optional
        Population label; default 'selection'.
    parent : str or None, optional
        Parent scope name recording lineage; default None.

    Attributes
    ----------
    dataset_id : str
        Identity of ordered column labels, index labels, and cell values.
    positions : tuple[int, ...]
        Absolute source positions, never dataframe index labels or offsets within
        a parent selection. Duplicate dataframe indexes are therefore safe.
    name : str
        Displayed population label.
    parent : str or None
        Parent scope name, when refined or selected from saved evidence.

    Raises
    ------
    ValueError
        Positions repeat or are negative/noninteger (booleans are invalid).

    Notes
    -----
    Scopes are frozen and source-bound. Reordering or changing values/labels
    invalidates reuse; dtype metadata alone is not part of identity. Scopes store
    positions, not source cells. Search/display budgets do not modify a scope.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({'x': [10, 20, 30]}, index=['a', 'a', 'b'])
    >>> selected = fw.Scope.from_positions(df, [2, 0], name='eligible')
    >>> selected.positions
    (0, 2)
    >>> selected.refine(df, [2]).parent
    'eligible'
    """

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
    @operation("scope selection")
    def from_positions(
        cls,
        df: pd.DataFrame,
        positions: Iterable[int | np.integer[Any]],
        *,
        name: str = "selection",
        progress: Progress = None,
        cancel: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> Scope:
        """Create a source-bound scope from absolute row positions.

        Parameters
        ----------
        df : pandas.DataFrame
            Source frame, read without mutation. Labels may be unique strings, integers,
            or recursively nested tuples. Duplicate index labels are supported;
            selections use integer row positions. Unsupported scalars raise TypeError.
        positions : iterable of int
            Unique, nonnegative, in-bounds positions in the original source frame.
            Input order is normalized to source order; an empty iterable is valid.
        name : str, optional
            Scope label; default 'selection'.
        progress : bool or callable, optional
            Default None is silent; True uses the built-in display. A callback receives
            ProgressEvent objects synchronously. False is also silent. Callback errors
            propagate unchanged; do not mutate the frame from a callback.
        cancel : CancellationToken or None, optional
            Cooperative cancellation token; default None. A cancelled token raises
            AnalysisCancelled at the next checkpoint, with no partial result.
        timeout : float or None, optional
            Finite nonnegative seconds from call start; default None disables the
            deadline. Expiration raises AnalysisCancelled cooperatively, after the
            current pandas/NumPy work item returns, rather than at a hard deadline.

        Returns
        -------
        Scope
            Frozen selection with the full source fingerprint and sorted positions.

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
        Reads and fingerprints the whole frame without mutation. Positions are not
        index labels; duplicate dataframe indexes are allowed.
        """
        selected = tuple(positions)
        if any(
            isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 0 or p >= len(df)
            for p in selected
        ):
            raise ValueError("Scope positions must be valid nonnegative row positions")
        if len(set(selected)) != len(selected):
            raise ValueError("Scope positions must not repeat")
        return cls(fingerprint(df), tuple(sorted(int(p) for p in selected)), name)

    @operation("scope refinement")
    def refine(
        self,
        df: pd.DataFrame,
        positions: Iterable[int | np.integer[Any]],
        *,
        name: str = "refined",
        progress: Progress = None,
        cancel: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> Scope:
        """Select a subset of this scope using absolute source positions.

        Parameters
        ----------
        df : pandas.DataFrame
            Source frame, read without mutation. Labels may be unique strings, integers,
            or recursively nested tuples. Duplicate index labels are supported;
            selections use integer row positions. Unsupported scalars raise TypeError.
        positions : iterable of int
            Unique absolute source positions contained in this scope, not offsets
            within its selected rows. An empty iterable creates an empty child.
        name : str, optional
            Child scope label; default 'refined'.
        progress : bool or callable, optional
            Default None is silent; True uses the built-in display. A callback receives
            ProgressEvent objects synchronously. False is also silent. Callback errors
            propagate unchanged; do not mutate the frame from a callback.
        cancel : CancellationToken or None, optional
            Cooperative cancellation token; default None. A cancelled token raises
            AnalysisCancelled at the next checkpoint, with no partial result.
        timeout : float or None, optional
            Finite nonnegative seconds from call start; default None disables the
            deadline. Expiration raises AnalysisCancelled cooperatively, after the
            current pandas/NumPy work item returns, rather than at a hard deadline.

        Returns
        -------
        Scope
            Child scope with this scope's name as parent; the parent is unchanged.

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
        The supplied frame must match the parent's ordered source. A valid row
        position outside the parent is rejected with ValueError.
        """
        child = Scope.from_positions(df, positions, name=name)
        if child.dataset_id != self.dataset_id or not set(child.positions) <= set(self.positions):
            raise ValueError("Refinement must select positions from its parent scope")
        return Scope(child.dataset_id, child.positions, name, self.name)


class InvestigationResult(ExplorerResult):
    """Browse saved discovery evidence and recover verified source populations.

    Parameters
    ----------
    kind : str
        'missingness', 'dependencies', 'paths', 'value_patterns', 'overview', or
        'comparison'. Public analyses and from_dict construct these results.
    payload : dict, optional
        Saved evidence; default is a new empty dictionary. Analytical constructors
        populate source, scope, missing_convention, findings, and kind-specific
        sections. Empty hand-built payloads do not support inspection.
    schema_version : str, optional
        Discovery analyses and from_dict use '1.0'. The inherited raw constructor
        defaults to foundation '0.3'; use the producer/loader for discovery data.
    stability : str, optional
        Inherited stability marker; default 'unstable'.

    Attributes
    ----------
    kind : str
        Discovery operation kind.
    payload : dict[str, Any]
        Mutable nested evidence. Findings contain id, pattern, statement,
        features, metrics, counting_unit/analysis_unit, structural predicates,
        and bounded examples/exceptions. Row selections record positions,
        total, omitted, and limit. IDs identify findings within this result.
    schema_version : str
        '1.0' for supported discovery exports.
    stability : str
        Evidence stability marker.

    Notes
    -----
    Use to_frame for findings or list sections such as availability, candidates,
    changes, or summaries. Overview sections hold ordinary result exports; restore
    one with from_dict before calling its methods. Each finding retains its own
    population and counting unit. Search omissions are not negative findings.
    Dependency records include observed target coverage on determinant-eligible
    rows and repeat-only consistency on the target-specific evaluated rows.
    Undefined fractions are None. Candidate determines_with_repeated_support
    lists global exact targets with repeated groups; global_targets_tested and
    global_targets_possible disclose completed versus selected global tests.
    Candidate repeated_rows describes the determinant population, which can be
    larger than any individual dependency population. These are row-weighted
    observations, not entity validity or reliability guarantees.

    Top-level attributes are frozen, but nested payloads and ordinary exports are
    mutable. Findings contain representative positions, not source rows. inspect,
    select, and recompute require the identical ordered source. Use Recipe to
    reapply parameters to a new delivery. Methods inherited from ExplorerResult
    provide mapping access and ordinary/resolved/compact exports.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({'x': [1, None]})
    >>> result = fw.missingness(df)
    >>> result.to_frame().empty
    False
    >>> fw.InvestigationResult.from_dict(result.to_dict()).kind
    'missingness'
    """

    def to_frame(self, section: str = "findings") -> pd.DataFrame:
        """Project a saved list section into a normalized dataframe.

        Parameters
        ----------
        section : str, optional
            List section name; default 'findings'. Common alternatives include
            'availability', 'dependencies', 'candidates', 'changes', and 'summaries',
            depending on kind. Missing sections yield an empty dataframe.

        Returns
        -------
        pandas.DataFrame
            pandas.json_normalize projection with nested mapping fields flattened
            into dotted column names. This contains evidence, not source rows.

        Notes
        -----
        The result is a projection of saved data; it does not rerun analysis or
        validate source identity. Nested object values may remain shared.
        The dependencies section includes completed tests below min_accuracy;
        findings contains only emitted findings. Legacy exports retain their
        original fields; to_frame does not fabricate missing measurements.
        """
        return pd.json_normalize(self.payload.get(section, []))

    def relationships(
        self, feature: str | None = None, *, kinds: Iterable[str] | None = None
    ) -> pd.DataFrame:
        """Browse saved feature connections and their supporting finding references.

        Parameters
        ----------
        feature : str or None, optional
            Restrict to connections containing this column; default None includes all.
        kinds : iterable of str or None, optional
            Relationship kinds to retain; default None includes all saved kinds.
            Use returned kind values to discover available categories.

        Returns
        -------
        pandas.DataFrame
            Normalized feature-network relationship records. Empty when no matching
            network is saved. Records retain their units and evidence references.

        Notes
        -----
        Connectedness means reachability, not equivalence or a composed functional
        dependency. No dataframe or recomputation is needed.
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

    @operation("inspection")
    def inspect(
        self,
        df: pd.DataFrame,
        finding: str | int,
        *,
        exceptions: bool = False,
        all_matches: bool = False,
        progress: Progress = None,
        cancel: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> pd.DataFrame:
        """Return representative or complete matching source rows as a copy.

        Parameters
        ----------
        df : pandas.DataFrame
            Original ordered source frame; labels, index, and values must match the
            saved fingerprint. Duplicate index labels are supported.
        finding : str or int
            Finding ID such as "f0", or zero-based position in this result's findings
            list. Integer indexing follows Python list rules, including negative indices.
        exceptions : bool, optional
            Default False selects supporting rows. True selects saved counterexample
            rows or their complete matching population.
        all_matches : bool, optional
            Default False returns only saved representative examples/exceptions,
            bounded by example_limit. True recovers the entire matching population.
        progress : bool or callable, optional
            Default None is silent; True uses the built-in display. A callback receives
            ProgressEvent objects synchronously. False is also silent. Callback errors
            propagate unchanged; do not mutate the frame from a callback.
        cancel : CancellationToken or None, optional
            Cooperative cancellation token; default None. A cancelled token raises
            AnalysisCancelled at the next checkpoint, with no partial result.
        timeout : float or None, optional
            Finite nonnegative seconds from call start; default None disables the
            deadline. Expiration raises AnalysisCancelled cooperatively, after the
            current pandas/NumPy work item returns, rather than at a hard deadline.

        Returns
        -------
        pandas.DataFrame
            Source rows in source order, selected with iloc and copied. Original
            column labels and index labels, including duplicates, are preserved.

        Raises
        ------
        ValueError
            The source differs or a saved selector cannot resolve a population.
        KeyError
            The string finding ID is unknown or required saved fields are missing.
        IndexError
            An integer finding position is outside the findings list.
        AnalysisCancelled
            Cancellation or the cooperative timeout stops source verification or selection.

        Notes
        -----
        all_matches=True uses the saved selector and conventions; unsupported saved
        selector forms can fall back to recomputing the owning section. Overview
        findings resolve through their section. Entity selections include all rows of
        matching entities within the analyzed scope/context, including absent values.
        Comparison findings do not have a single recoverable source population.
        """
        record = self._finding(df, finding)
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
        progress: Progress = None,
        cancel: CancellationToken | None = None,
        timeout: float | None = None,
    ) -> Scope:
        """Recover all matching source positions as a reusable Scope.

        Parameters
        ----------
        df : pandas.DataFrame
            Original ordered source frame; labels, index, and values must match the
            saved fingerprint. Duplicate index labels are supported.
        finding : str or int
            Finding ID such as "f0", or zero-based position in this result's findings
            list. Integer indexing follows Python list rules, including negative indices.
        exceptions : bool, optional
            Default False selects supporting rows. True selects saved counterexample
            rows or their complete matching population.
        name : str, optional
            Returned scope label; default 'finding selection'.
        progress : bool or callable, optional
            Default None is silent; True uses the built-in display. A callback receives
            ProgressEvent objects synchronously. False is also silent. Callback errors
            propagate unchanged; do not mutate the frame from a callback.
        cancel : CancellationToken or None, optional
            Cooperative cancellation token; default None. A cancelled token raises
            AnalysisCancelled at the next checkpoint, with no partial result.
        timeout : float or None, optional
            Finite nonnegative seconds from call start; default None disables the
            deadline. Expiration raises AnalysisCancelled cooperatively, after the
            current pandas/NumPy work item returns, rather than at a hard deadline.

        Returns
        -------
        Scope
            Complete matching source positions with parent scope lineage. Selection
            is independent of saved example limits and does not mutate the source.

        Raises
        ------
        ValueError
            The source differs or a saved selector cannot resolve a population.
        KeyError
            The string finding ID is unknown or required saved fields are missing.
        IndexError
            An integer finding position is outside the findings list.
        AnalysisCancelled
            Cancellation or the cooperative timeout stops source verification or selection.

        Notes
        -----
        Evaluates the saved predicate under its scope and missing conventions;
        unsupported selector forms may fall back to section recomputation. Overview
        finding IDs are routed to their owning section. Entity selectors retain all
        source rows of matching entities inside that population. Comparison findings
        do not support source selection.

        Examples
        --------
        >>> import pandas as pd
        >>> import fieldwork as fw
        >>> df = pd.DataFrame({'x': [1, 2, 3]})
        >>> result = fw.missingness(df, example_limit=1)
        >>> scope = result.select(df, 0)
        >>> isinstance(scope, fw.Scope)
        True
        """
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
        from ._selection import select_rows

        selected = select_rows(df, analysis, record, exceptions)
        if selected is not None:
            return Scope(
                self["source"]["dataset_id"], tuple(selected), name, analysis["scope"]["name"]
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

    @operation("recomputation")
    def recompute(
        self,
        df: pd.DataFrame,
        *,
        progress: Progress = None,
        cancel: CancellationToken | None = None,
        timeout: float | None = None,
        **overrides: Any,
    ) -> InvestigationResult:
        """Reapply a saved individual analysis to its verified source.

        Parameters
        ----------
        df : pandas.DataFrame
            Original ordered source frame; labels, index, and values must match the
            saved fingerprint. Duplicate index labels are supported.
        progress : bool or callable, optional
            Default None is silent; True uses the built-in display. A callback receives
            ProgressEvent objects synchronously. False is also silent. Callback errors
            propagate unchanged; do not mutate the frame from a callback.
        cancel : CancellationToken or None, optional
            Cooperative cancellation token; default None. A cancelled token raises
            AnalysisCancelled at the next checkpoint, with no partial result.
        timeout : float or None, optional
            Finite nonnegative seconds from call start; default None disables the
            deadline. Expiration raises AnalysisCancelled cooperatively, after the
            current pandas/NumPy work item returns, rather than at a hard deadline.
        **overrides : Any
            Keyword options accepted by the saved operation. Explicit values replace
            saved parameters/context; use, for example, example_limit=20 to save more
            representatives. Options depend on the result kind.

        Returns
        -------
        InvestigationResult
            New result with the overrides applied. Paths restore as PathResult.

        Raises
        ------
        ValueError
            The source differs, kind is overview/comparison, or overrides are invalid.
        TypeError
            An override is not accepted by the saved operation.
        AnalysisCancelled
            Cancellation or the cooperative timeout stops recomputation.

        Notes
        -----
        Supports missingness, dependencies, paths, and value_patterns. Restore and
        recompute an individual overview section rather than the composition. To
        analyze another delivery use Recipe; recompute checks the saved fingerprint.
        The original result is not modified.
        """
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
    def from_dict(cls, data: Mapping[str, Any]) -> InvestigationResult:
        """Restore discovery evidence, including path behavior, from saved data.

        Parameters
        ----------
        data : mapping
            Ordinary discovery schema 1.0 export or a fieldwork.compact envelope
            containing one. JSON-decoded data is accepted.

        Returns
        -------
        InvestigationResult
            Restored evidence; kind='paths' produces PathResult with best/path/census
            handoff behavior. Ordinary nested containers are reused.

        Raises
        ------
        ValueError
            The schema/envelope version or compact reference graph is unsupported.
        KeyError
            Required saved fields are missing.

        Notes
        -----
        Does not recompute analyses or verify source identity. Source checks happen
        when inspecting, selecting, recomputing, or handing a path to census. This is
        not a full validator of every nested evidence field.
        """
        from ._serialization import expand_result

        data = expand_result(data)
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


def prepare(df, *, scope=None, missing=None, table_id="table", features=None, presence_features=()):
    columns(df)
    return prepare_context(
        df,
        scope=scope,
        missing=missing,
        table_id=table_id,
        features=features,
        presence_features=presence_features,
    )


def prepare_context(
    df, *, scope=None, missing=None, table_id="table", features=None, presence_features=()
):
    """Prepare source context independently of discovery's column-label contract."""
    validate_frame(df)
    if not isinstance(table_id, str) or not table_id:
        raise ValueError("table_id must be a nonempty string")
    identity = fingerprint(df)
    if scope is not None and scope.dataset_id != identity:
        raise ValueError("Scope belongs to a different ordered dataset")
    if scope is not None and any(p >= len(df) for p in scope.positions):
        raise ValueError("Scope positions exceed the source population")
    missing = missing or {}
    labels = {normalize_scalar(c, label=True) for c in df.columns}
    for c in missing:
        if normalize_scalar(c, label=True) not in labels:
            raise KeyError(c)
    sentinel_values = {
        c: sorted({normalize_scalar(v) for v in missing.get(c, [])}, key=lambda v: v.sort_key())
        for c in df
    }
    conventions = {c: [v.to_dict() for v in values] for c, values in sentinel_values.items()}
    cache_key = (
        id(df),
        scope.positions if scope else None,
        tuple(
            (normalize_scalar(c, label=True), tuple(values))
            for c, values in sentinel_values.items()
        ),
    )
    session = current_session()
    cached = session.prepared.get(cache_key) if session else None
    if cached is None:
        positions = np.array(scope.positions, dtype=np.int64) if scope else np.arange(len(df))
        frame = df.iloc[positions] if scope else df
        cached = (df, frame, positions, {}, {})
        if session:
            session.prepared[cache_key] = cached
    _, frame, positions, all_encoded, all_available = cached
    selected = list(dict.fromkeys(df.columns if features is None else features))
    presence_columns = list(dict.fromkeys([*selected, *presence_features]))
    needed = [
        c
        for c in presence_columns
        if c not in all_available or (c in selected and c not in all_encoded)
    ]

    def sentinel_key(v):
        if v.kind == "integer":
            return ("number", int(v.value))
        if v.kind == "float":
            return ("number", float.fromhex(v.value))
        return v

    with phase("encoding", len(needed), "columns") as tracker:
        for c in needed:
            dtype = frame[c].dtype
            native_only = (
                pd.api.types.is_numeric_dtype(dtype)
                or pd.api.types.is_datetime64_any_dtype(dtype)
                or pd.api.types.is_timedelta64_dtype(dtype)
                or isinstance(dtype, pd.StringDtype)
            )
            if c not in selected and not sentinel_values[c] and native_only:
                all_available[c] = frame[c].notna().to_numpy(dtype=bool)
                tracker.advance(detail=str(c))
                continue
            values, codes = encode_series(frame[c])
            sentinel_keys = {sentinel_key(v) for v in sentinel_values[c]}
            mask = np.array(
                [v != MISSING and sentinel_key(v) not in sentinel_keys for v in values], dtype=bool
            )
            all_available[c] = mask[codes]
            if c in selected:
                all_encoded[c] = codes
                if session:
                    session.remember_encoding((id(frame), c), (frame, values, codes))
            tracker.advance(detail=str(c))
    encoded = {c: all_encoded[c] for c in selected}
    available = {c: all_available[c] for c in presence_columns}
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


def normalized_encoding(frame, codes, present):
    """Foundation dictionaries with native/sentinel absence in one missing level."""
    session = current_session()
    output = {}
    for c, code in codes.items():
        checkpoint()
        cached = session.encodings.get((id(frame), c)) if session else None
        values = cached[1] if cached else encode_series(frame[c])[0]
        absent = np.unique(code[~present[c]])
        if all(values[i] == MISSING for i in absent):
            output[c] = (values, code)
            continue
        cache_key = (id(frame), c, id(present[c]))
        normalized = session.encodings.get(cache_key) if session else None
        if normalized is None:
            tokens = list(values)
            for i in absent:
                tokens[i] = MISSING
            dictionary = sorted(set(tokens), key=lambda v: v.sort_key())
            lookup = {v: i for i, v in enumerate(dictionary)}
            remap = np.fromiter((lookup[v] for v in tokens), dtype=np.int64)
            normalized = (frame, dictionary, remap[code])
            if session:
                session.remember_encoding(cache_key, normalized)
        output[c] = normalized[1:]
    return output


def selection(positions, total, example_limit):
    return {
        "positions": [int(p) for p in positions[:example_limit]],
        "total": int(total),
        "omitted": max(0, int(total) - example_limit),
        "method": "first_in_source_order",
        "limit": example_limit,
    }


@dataclass(frozen=True)
class EvidenceRows:
    """Bounded examples with the full source-row count, private to finding assembly."""

    positions: Any
    total: int

    def __len__(self):
        return self.total

    def __getitem__(self, key):
        return self.positions[key]


def bounded_rows(positions, mask, limit):
    from ._explore._kernels import first_indices

    return EvidenceRows(positions[first_indices(mask, limit)], int(np.count_nonzero(mask)))


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


def qualitative_analysis_unit(base, record):
    """Describe a finding's own unit without exporting support or population sizes."""
    section = record.get("selector", {}).get("analysis_section")
    if section:
        base = base["sections"][section]
    unit = record["counting_unit"]
    analysis = base.get("analysis_unit", {})
    output = {"counting_unit": unit}
    if unit == analysis.get("counting_unit"):
        output.update(
            {k: analysis[k] for k in ("entity_keys", "presence_aggregation") if k in analysis}
        )
    elif unit == "rows":
        output.update(entity_keys=[], presence_aggregation="per_row")
    else:
        output["entity_keys"] = record.get("structure", {}).get("entity_keys", [])
    if record["pattern"] in {"entity_availability", "entity_summary"}:
        output.pop("presence_aggregation", None)
    return output


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
    from ._explore.census import _census

    if operation is _census:
        # Consume a dimensions generator once, before both preparation and census.
        args = (tuple(args[0]), *args[1:])
    frame, _, codes, present, base = prepare_context(
        df,
        scope=scope,
        missing=missing,
        table_id=table_id,
        features=args[0] if operation is _census else None,
    )
    if not all(isinstance(c, str) for c in df.columns):
        # JSON object keys cannot preserve integer identities or encode tuples.
        conventions = base["missing_convention"]
        conventions["sentinels_by_column"] = [
            {"column": normalize_scalar(c, label=True).to_dict(), "values": values}
            for c, values in conventions.pop("sentinels").items()
        ]
    if operation is _census:
        analysis = operation(
            frame, *args, _encoded=normalized_encoding(frame, codes, present), **options
        )
    else:
        normalized = frame.copy()
        for c in normalized:
            checkpoint()
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

    def rebase_sources(result_payload):
        # Only result roots and their analytical sections own dataset metadata.
        # Elsewhere, `source` can be a graph node reference or a feature name.
        if "source" in result_payload:
            result_payload["source"] = deepcopy(source)
        for section in result_payload.get("sections", {}).values():
            rebase_sources(section)

    visited = set()

    def rebase(value):
        if not isinstance(value, (dict, list)) or id(value) in visited:
            return
        # deepcopy preserves aliases, including the census scope shared by pair
        # and grain lineage metadata. Rebase each container once, not each path.
        visited.add(id(value))
        if isinstance(value, dict):
            if "scope_id" in value and "input_rows" in value:
                value["input_rows"] += excluded
                value["restriction_excluded_rows"] += excluded
                value["conditional"] = value["conditional"] or bool(excluded)
                value["lineage"] = [base["scope"]["name"], *value["lineage"]]
            for child in value.values():
                rebase(child)
        else:
            for child in value:
                rebase(child)

    rebase_sources(payload)
    rebase(payload)
    payload["analysis_context"] = {k: base[k] for k in ("source", "scope", "missing_convention")}
    return ExplorerResult(analysis.kind, payload, schema_version=analysis.schema_version)


def context_statement(context):
    return ", ".join(
        f"{feature} = {_restore_scalar(value)!r} ({value['type']})"
        for feature, value in context.items()
    )
