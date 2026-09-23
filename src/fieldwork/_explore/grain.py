"""Exact observed dependencies of explicit candidate keys, and their grain graph."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Unpack

import numpy as np
import pandas as pd

from .._runtime import checkpoint, operation, phase
from ..result import Result
from ..typing import Runtime
from ._kernels import FDCache, MaskPool, same_mask
from .grain_graph import build_grain_graph

if TYPE_CHECKING:
    from ..evidence import Scope


@dataclass(frozen=True)
class KeySpec:
    """Declare an explicitly named single-column or composite determinant.

    Parameters
    ----------
    name : str
        Nonempty name, unique among the candidates of one call.
    columns : tuple of str
        Nonempty ordered determinant columns (non-string labels are named by str()).

    Raises
    ------
    ValueError
        The name or columns are empty. Analyses also reject repeated or unknown
        components and duplicate candidate names.

    Examples
    --------
    >>> import fieldwork as fw
    >>> fw.KeySpec('visit', ('site', 'participant', 'visit_number')).columns
    ('site', 'participant', 'visit_number')
    """

    name: str
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("KeySpec.name must be a non-empty string")
        columns = tuple(self.columns)
        if not columns:
            raise ValueError("KeySpec.columns must not be empty")
        object.__setattr__(self, "columns", columns)


def key_specs(df: pd.DataFrame, candidate_keys: Iterable[Any]) -> tuple[KeySpec, ...]:
    """Validate candidates: a column name is a single-column key named after it.

    A saved ``{"name", "columns"}`` record (as in result parameters) is a KeySpec.
    """
    from ..evidence import columns

    specs = []
    for item in candidate_keys:
        if isinstance(item, Mapping):
            item = KeySpec(item["name"], tuple(item["columns"]))
        spec = item if isinstance(item, KeySpec) else KeySpec(str(item), (item,))
        specs.append(KeySpec(spec.name, tuple(columns(df, spec.columns))))
    if not specs:
        raise ValueError("candidate_keys must contain at least one key")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("candidate key names must be unique")
    return tuple(specs)


class Encoded:
    """Row codes and the missing code of each column, plus reusable test results.

    Grain never displays cell values, so it needs no value dictionaries.
    """

    def __init__(
        self,
        codes: Mapping[str, np.ndarray],
        missing: Mapping[str, int | None],
        cache: FDCache | None = None,
    ):
        self.codes, self.missing = codes, missing
        self.cache = cache if cache is not None else FDCache()

    def complete(self, columns: Iterable[str], mask: np.ndarray) -> np.ndarray:
        """Rows of ``mask`` where none of ``columns`` is missing."""
        mask = mask.copy()
        for column in columns:
            if self.missing[column] is not None:
                mask &= self.codes[column] != self.missing[column]
        return mask


def check_dependency(
    size: int,
    spec: KeySpec,
    target: str,
    *,
    dropna: bool,
    encoded: Encoded,
    row_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Test spec -> target exactly, on complete cases when dropna; returns rows used."""
    mask = np.ones(size, dtype=bool) if row_mask is None else np.asarray(row_mask).copy()
    if dropna:
        mask = encoded.complete((*spec.columns, target), mask)
    checkpoint()
    cache_key = (spec.columns, target, dropna)
    counts = encoded.cache.get(cache_key, mask)
    if counts is None:
        counts = _group_counts(encoded, spec.columns, target, mask)
        encoded.cache.put(cache_key, mask, counts, global_population=row_mask is None)
    groups, violating = counts["evaluated_groups"], counts["violating_groups"]
    evaluated = counts["evaluated_rows"]
    record = {
        "key_name": spec.name,
        "key_columns": list(spec.columns),
        "target": target,
        "holds": None if groups == 0 else violating == 0,
        "undefined_reason": "no_evaluated_groups" if groups == 0 else None,
        "evaluated_groups": groups,
        "violating_groups": violating,
        "affected_rows": counts["affected_rows"],
        "evaluated_rows": evaluated,
        "missing_excluded_rows": size - evaluated,
        "singleton_groups": counts["singleton_groups"],
        "repeated_groups": groups - counts["singleton_groups"],
    }
    return record, mask


def _group_counts(
    encoded: Encoded, key: tuple[str, ...], target: str, mask: np.ndarray
) -> dict[str, int]:
    names = [f"k{index}" for index in range(len(key))]
    table = pd.DataFrame(
        {
            **{name: encoded.codes[column][mask] for name, column in zip(names, key)},
            "target": encoded.codes[target][mask],
        }
    )
    counts = {"evaluated_rows": int(mask.sum())}
    if not len(table):
        zero = ("evaluated_groups", "violating_groups", "affected_rows", "singleton_groups")
        return {**counts, **dict.fromkeys(zero, 0)}
    grouped = table.groupby(names, sort=False, observed=True)["target"].agg(["nunique", "size"])
    violations = grouped["nunique"] > 1
    return {
        **counts,
        "evaluated_groups": len(grouped),
        "violating_groups": int(violations.sum()),
        "affected_rows": int(grouped.loc[violations, "size"].sum()),
        "singleton_groups": int((grouped["size"] == 1).sum()),
    }


def record_on(
    size: int,
    spec: KeySpec,
    target: str,
    mask: Any,
    *,
    known: dict,
    dropna: bool,
    encoded: Encoded,
) -> tuple[dict[str, Any], Any]:
    """Test spec -> target on the rows of ``mask``, reusing a test on that population.

    Returns the record and its evaluated rows, which can be fewer than ``mask``
    when dropna excludes further incomplete cases.
    """
    saved = known.get((spec.name, target))
    if saved is not None and same_mask(saved[1], mask):
        return saved[0], mask
    return check_dependency(
        size, spec, target, dropna=dropna, encoded=encoded, row_mask=np.asarray(mask)
    )


@operation("grain")
def grain(
    df: pd.DataFrame,
    candidate_keys: Iterable[str | KeySpec | Mapping[str, Any]],
    *,
    dropna: bool = False,
    scope: Scope | None = None,
    missing: Mapping[str, Iterable[Any]] | None = None,
    table_id: str = "table",
    **runtime: Unpack[Runtime],
) -> Result:
    """Test which columns each candidate key determines, and relate the keys.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame, read without mutation.
    candidate_keys : iterable of str, KeySpec or mapping
        Nonempty candidates. A column name is a single-column key; use
        KeySpec(name, columns), or its JSON form ``{"name": ..., "columns": [...]}``,
        for composite keys. Names must be unique.
    dropna : bool, optional
        Default False treats missing values as a category. True tests each
        key/target pair on its complete cases, so populations can differ.
    scope, missing, table_id
        Source context shared by every analysis.
    **runtime : Unpack[Runtime]
        Optional progress, cancel and timeout controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'grain': one exact test per key and non-key column (``dependencies``),
        per-column placement summaries (``targets``), and ``graph``, which merges
        equivalent keys, links coarser to finer keys and places each column at
        the coarsest keys determining it. Exactness describes this delivery only;
        see docs/algorithms.md for singleton and repeated support.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> df = pd.DataFrame({"site": ["A", "A"], "visit": [1, 2], "value": [3, 4]})
    >>> result = fw.grain(df, ["site", fw.KeySpec("visit_key", ("site", "visit"))])
    >>> [t["determining_keys"] for t in result["targets"] if t["target"] == "value"]
    [['visit_key']]
    """
    from ..evidence import columns, prepare_values

    names = columns(df)
    frame, _, values, base = prepare_values(
        df, names, scope=scope, missing=missing, table_id=table_id
    )
    specs = key_specs(frame, candidate_keys)
    encoded = Encoded(
        {c: codes for c, (_, codes) in values.items()},
        {c: len(v) - 1 if v and v[-1] is None else None for c, (v, _) in values.items()},
    )
    base["parameters"] = {
        "candidate_keys": [{"name": s.name, "columns": list(s.columns)} for s in specs],
        "dropna": dropna,
    }
    base.update(evaluate(len(frame), names, specs, dropna=dropna, encoded=encoded))
    return Result("grain", base)


def evaluate(
    size: int, names: list[str], specs: tuple[KeySpec, ...], *, dropna: bool, encoded: Encoded
) -> dict[str, Any]:
    """Grain payload fields: keys, dependency tests, target summaries and graph."""
    pool, known, records = MaskPool(), {}, []
    total = sum(len(names) - len(spec.columns) for spec in specs)
    with phase("exact dependencies", total, "tests") as tracker:
        for spec in specs:
            for target in names:
                if target in spec.columns:
                    continue
                record, evaluated = check_dependency(
                    size, spec, target, dropna=dropna, encoded=encoded
                )
                records.append(record)
                known[(spec.name, target)] = (record, pool.intern(evaluated))
                tracker.advance(detail=f"{spec.name} → {target}")
    return {
        "keys": [{"name": s.name, "columns": list(s.columns)} for s in specs],
        "dependencies": records,
        "targets": _targets(size, names, specs, records, known, dropna, encoded),
        "graph": build_grain_graph(size, names, specs, encoded, known, dropna=dropna),
        "warnings": [],
    }


def _targets(size, names, specs, records, known, dropna, encoded) -> list[dict[str, Any]]:
    """Per column: determining keys and how they compare on the same rows."""
    by_name = {spec.name: spec for spec in specs}
    determining = defaultdict(list)
    for record in records:
        if record["holds"] is True:
            determining[record["target"]].append(record["key_name"])

    def determines(left: str, right: str, mask: Any) -> bool:
        # Key-to-key evidence must use the same rows as the target tests.
        spec = by_name[left]
        return all(
            component in spec.columns
            or record_on(size, spec, component, mask, known=known, dropna=dropna, encoded=encoded)[
                0
            ]["holds"]
            is True
            for component in by_name[right].columns
        )

    summaries = []
    for target in names:
        sets = [known[(spec.name, target)][1] for spec in specs if (spec.name, target) in known]
        if not sets:
            continue
        keys = determining[target]
        comparable = not dropna or all(same_mask(item, sets[0]) for item in sets[1:])
        equivalent, incomparable, coarsest = [], [], list(keys)
        for index, left in enumerate(keys if comparable else []):
            for right in keys[index + 1 :]:
                mask = known[(left, target)][1]
                left_right = determines(left, right, mask)
                right_left = determines(right, left, mask)
                if left_right and right_left:
                    equivalent.append([left, right])
                elif not left_right and not right_left:
                    incomparable.append([left, right])
                elif left_right and left in coarsest:
                    coarsest.remove(left)
                elif right_left and right in coarsest:
                    coarsest.remove(right)
        summaries.append(
            {
                "target": target,
                "determining_keys": keys,
                "cross_key_comparison": "comparable" if comparable else "not_comparable",
                "coarsest_candidates": coarsest if comparable else [],
                "equivalent_determinants": equivalent,
                "incomparable_candidates": incomparable,
            }
        )
    return summaries
