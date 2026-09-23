"""Observed functional-dependency and grain evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Unpack

import numpy as np
import pandas as pd

from .._runtime import checkpoint, operation, phase
from ..typing import ColumnLabel, Runtime
from ._kernels import EncodedColumns, MaskPool, same_mask
from .census import _scope, _source
from .encoding import (
    MissingCode,
    encode_series,
    missing_code,
    normalize_scalar,
    resolve_columns,
    validate_frame,
)
from .grain_graph import build_grain_graph
from .result import ExplorerResult, KeySpec


def _key_specs(df: pd.DataFrame, candidate_keys: Iterable[Any]) -> tuple[KeySpec, ...]:
    validate_frame(df)
    specs: list[KeySpec] = []
    for index, item in enumerate(candidate_keys):
        if isinstance(item, KeySpec):
            spec = item
        else:
            resolved = resolve_columns(df, [item], argument="candidate_keys")
            spec = KeySpec(str(item), (resolved[0],))
        columns = resolve_columns(df, spec.columns, argument=f"key {spec.name!r}")
        tokens = [normalize_scalar(column, label=True) for column in columns]
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"Key {spec.name!r} has repeated components")
        specs.append(KeySpec(spec.name, columns))
    if not specs:
        raise ValueError("candidate_keys must contain at least one key")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("candidate key names must be unique")
    return tuple(specs)


def _fd_record(
    df: pd.DataFrame,
    spec: KeySpec,
    target: Any,
    *,
    dropna: bool,
    scope_prefix: str,
    encoded: dict[Any, tuple[list[Any], np.ndarray]],
    row_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    mask = np.ones(len(df), dtype=bool) if row_mask is None else row_mask.copy()
    if dropna:
        for column in (*spec.columns, target):
            values, codes = encoded[column]
            absent = missing_code(values)
            if absent is not None:
                mask &= codes != absent
    checkpoint()
    cache = getattr(encoded, "fd_cache", None)
    cache_key = (spec.columns, target, dropna)
    cached = cache.get(cache_key, mask) if cache is not None else None
    if cached is None:
        key_names = [f"k{index}" for index in range(len(spec.columns))]
        table = pd.DataFrame(
            {
                **{name: encoded[column][1][mask] for name, column in zip(key_names, spec.columns)},
                "target": encoded[target][1][mask],
            }
        )
        if len(table):
            grouped = table.groupby(key_names, sort=False, observed=True)["target"].agg(
                ["nunique", "size"]
            )
            violations = grouped["nunique"] > 1
            evaluated_groups = len(grouped)
            violating_groups = int(violations.sum())
            affected_rows = int(grouped.loc[violations, "size"].sum())
            singleton_groups = int((grouped["size"] == 1).sum())
        else:
            evaluated_groups = violating_groups = affected_rows = singleton_groups = 0
        cached = {
            "evaluated_groups": evaluated_groups,
            "violating_groups": violating_groups,
            "affected_rows": affected_rows,
            "singleton_groups": singleton_groups,
            "evaluated_rows": int(mask.sum()),
        }
        if cache is not None:
            cache.put(cache_key, mask, cached, global_population=row_mask is None)
    evaluated_groups = cached["evaluated_groups"]
    violating_groups = cached["violating_groups"]
    affected_rows = cached["affected_rows"]
    singleton_groups = cached["singleton_groups"]
    evaluated_rows = cached["evaluated_rows"]
    missing_excluded = len(df) - evaluated_rows
    scope_id = f"{scope_prefix}:{spec.name}:{normalize_scalar(target, label=True).sort_key()}"
    record = {
        "key_name": spec.name,
        "key_columns": [normalize_scalar(c, label=True).to_dict() for c in spec.columns],
        "target": normalize_scalar(target, label=True).to_dict(),
        "scope_id": scope_id,
        "holds": None if evaluated_groups == 0 else violating_groups == 0,
        "undefined_reason": "no_evaluated_groups" if evaluated_groups == 0 else None,
        "evaluated_groups": evaluated_groups,
        "violating_groups": violating_groups,
        "group_rate": violating_groups / evaluated_groups if evaluated_groups else None,
        "affected_rows": affected_rows,
        "evaluated_rows": evaluated_rows,
        "row_rate": affected_rows / evaluated_rows if evaluated_rows else None,
        "singleton_groups": singleton_groups,
        "repeated_groups": evaluated_groups - singleton_groups,
        "scope": _scope(scope_id, len(df), missing_excluded, 0, False),
    }
    return record, mask


def _grain(
    df: pd.DataFrame,
    candidate_keys: Iterable[Any],
    *,
    dropna: bool = False,
    scope_metadata: Mapping[str, Any] | None = None,
    _encoded=None,
    _cache=None,
) -> ExplorerResult:
    """Evaluate exact observed FDs for explicit determinant candidates."""

    specs = _key_specs(df, candidate_keys)
    records: list[dict[str, Any]] = []
    scopes: list[dict[str, Any]] = []
    holds_by_target: dict[Any, list[str]] = defaultdict(list)
    encoded = _encoded
    if encoded is None:
        encoded = {}
        with phase("grain encoding", len(df.columns), "columns") as tracker:
            for column in df.columns:
                values, codes = encode_series(df[column])
                encoded[column] = (MissingCode(missing_code(values)), codes)
                tracker.advance(detail=str(column))
    encoded = EncodedColumns(encoded, _cache)
    pool = MaskPool()
    evaluated_sets = {}
    with phase(
        "exact dependencies", sum(len(df.columns) - len(s.columns) for s in specs), "tests"
    ) as tracker:
        for spec in specs:
            checkpoint()
            components = {normalize_scalar(c, label=True) for c in spec.columns}
            for target in df.columns:
                if normalize_scalar(target, label=True) in components:
                    continue
                record, evaluated = _fd_record(
                    df,
                    spec,
                    target,
                    dropna=dropna,
                    scope_prefix="s3",
                    encoded=encoded,
                )
                records.append(record)
                scope = record.pop("scope")
                scopes.append(
                    _scope(
                        scope["scope_id"],
                        len(df),
                        scope["missing_excluded_rows"],
                        0,
                        bool((scope_metadata or {}).get("conditional")),
                        parent_scope=(scope_metadata or {}).get("scope"),
                    )
                )
                evaluated_sets[(spec.name, target)] = pool.intern(evaluated)
                if record["holds"] is True:
                    holds_by_target[target].append(spec.name)
                tracker.advance(detail=f"{spec.name} → {target}")
    target_summaries: list[dict[str, Any]] = []
    specs_by_name = {spec.name: spec for spec in specs}
    holds_lookup = {
        (record["key_name"], str(record["target"])): record["holds"] is True for record in records
    }

    def determines_key(left: str, right: str, mask: np.ndarray) -> bool:
        left_columns = {
            normalize_scalar(column, label=True) for column in specs_by_name[left].columns
        }
        for component in specs_by_name[right].columns:
            if normalize_scalar(component, label=True) in left_columns:
                continue
            if same_mask(evaluated_sets[(left, component)], mask):
                holds = holds_lookup[(left, str(normalize_scalar(component, label=True).to_dict()))]
            else:
                # Key-to-key evidence must use the same rows as the target FDs.
                evidence, _ = _fd_record(
                    df,
                    specs_by_name[left],
                    component,
                    dropna=dropna,
                    scope_prefix="comparison",
                    encoded=encoded,
                    row_mask=mask,
                )
                holds = evidence["holds"] is True
            if not holds:
                return False
        return True

    for target in df.columns:
        relevant = [
            record
            for record in records
            if record["target"] == normalize_scalar(target, label=True).to_dict()
        ]
        if not relevant:
            continue
        determining = holds_by_target.get(target, [])
        comparable = True
        if dropna and len(specs) > 1:
            sets = [
                evaluated_sets[(spec.name, target)]
                for spec in specs
                if (spec.name, target) in evaluated_sets
            ]
            comparable = all(same_mask(item, sets[0]) for item in sets[1:]) if sets else True
        equivalent: list[list[str]] = []
        incomparable: list[list[str]] = []
        coarsest = list(determining)
        for index, left in enumerate(determining):
            if not comparable:
                break
            for right in determining[index + 1 :]:
                mask = evaluated_sets[(left, target)]
                left_right = determines_key(left, right, mask)
                right_left = determines_key(right, left, mask)
                if left_right and right_left:
                    equivalent.append([left, right])
                elif not left_right and not right_left:
                    incomparable.append([left, right])
                elif left_right and left in coarsest:
                    coarsest.remove(left)
                elif right_left and right in coarsest:
                    coarsest.remove(right)
        target_summaries.append(
            {
                "target": normalize_scalar(target, label=True).to_dict(),
                "determining_keys": determining,
                "assignment": "compatible" if determining else "undetermined",
                "cross_key_comparison": "comparable" if comparable else "not_comparable",
                "coarsest_candidates": coarsest if comparable else [],
                "equivalent_determinants": equivalent if comparable else [],
                "incomparable_candidates": incomparable if comparable else [],
            }
        )
    payload: dict[str, Any] = {
        "status": "empty" if len(df) == 0 else "computed",
        "source": _source(df),
        "scopes": scopes,
        "keys": [
            {
                "name": spec.name,
                "columns": [normalize_scalar(c, label=True).to_dict() for c in spec.columns],
            }
            for spec in specs
        ],
        "dependencies": records,
        "graph": build_grain_graph(
            df,
            specs,
            encoded,
            records,
            evaluated_sets,
            dropna=dropna,
            scope_metadata=scope_metadata,
        ),
        "targets": target_summaries,
        "warnings": [],
        "scope_metadata": scope_metadata,
    }
    return ExplorerResult("grain", payload)


@operation("grain")
def grain(
    df: pd.DataFrame,
    candidate_keys: Iterable[ColumnLabel | KeySpec],
    *,
    dropna: bool = False,
    scope_metadata: Mapping[str, Any] | None = None,
    **runtime: Unpack[Runtime],
) -> ExplorerResult:
    """Evaluate exact observed dependencies for explicitly supplied keys.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation. Column labels must be unique strings,
        non-boolean integers, or recursively tuple-valued labels. Native missing
        scalars share one identity; integer and float values remain distinct.
        Unsupported column labels or scalar objects raise TypeError.
    candidate_keys : iterable of column labels or KeySpec
        Nonempty determinant candidates. A label means a single-column key; use
        KeySpec(name, columns) for composites. A tuple label denotes one column,
        not a composite. Candidate names and each key's columns must be unique.
    dropna : bool, optional
        Default False treats missing values as a category. True evaluates each
        determinant/target pair on its complete cases and records that population.
    scope_metadata : mapping or None, optional
        Optional descriptive lineage supplied by composition; default None. This
        does not select rows. Use a Scope with census/explore for row selection.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    ExplorerResult
        Kind 'grain', with explicit keys, dependency evidence, target placements,
        scopes, and a graph over compatible candidate populations. Graph aliases
        indicate equivalent observed partitions.

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
    Exact dependencies describe the observed delivery, not future guarantees.
    Singleton groups satisfy a dependency trivially; repeated and violating groups
    are reported separately. Candidate key uniqueness and dependency accuracy are
    different questions. Graph relationships use compatible populations rather
    than composing dependencies across differing complete-case cohorts.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"site": ["A", "A"], "visit": [1, 2], "value": [3, 4]})
    >>> result = fw.grain(df, ["site", fw.KeySpec("visit_key", ("site", "visit"))])
    >>> result.kind
    'grain'
    """
    return _grain(
        df,
        candidate_keys,
        dropna=dropna,
        scope_metadata=scope_metadata,
    )
