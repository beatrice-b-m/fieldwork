"""Structural checks across two tables: key coverage, cardinality, agreement, references.

The tables are never joined: each side is prepared under its own scope and
sentinels, key and compared values are mapped into shared match identities, and
every measurement is computed from per-key row counts. Selection recomputes the
same arrays from both verified sources.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Unpack

import numpy as np
import pandas as pd

from ._explore.encoding import value_key
from ._runtime import focus, operation, phase
from .evidence import Scope, bounded_rows, budgets, columns, fingerprint, prepare_values, selection
from .result import Result
from .typing import MatchMode, RelateLimits, Runtime, Side

_LIMITS = {"example_limit": 5}
_SIDES: tuple[Side, Side] = ("left", "right")
_KINDS = {
    0: "boolean",
    1: "number",
    3: "string",
    4: "date",
    5: "datetime",
    6: "aware datetime",
    7: "timedelta",
}
_RELATIONS = {
    (False, False): "1:1",
    (False, True): "1:n",
    (True, False): "n:1",
    (True, True): "n:m",
}

Sentinels = Mapping[str, Iterable[Any]]


@operation("relate")
def relate(
    left: pd.DataFrame,
    right: pd.DataFrame | None = None,
    *,
    on: Mapping[str, str],
    compare: Mapping[str, str] | None = None,
    match: MatchMode = "typed",
    dropna: bool = True,
    scopes: tuple[Scope | None, Scope | None] | None = None,
    missing: tuple[Sentinels | None, Sentinels | None] | None = None,
    table_ids: tuple[str, str] | None = None,
    limits: RelateLimits | None = None,
    **runtime: Unpack[Runtime],
) -> Result:
    """Relate two tables through a key: coverage, cardinality and agreement.

    Parameters
    ----------
    left : pandas.DataFrame
        Left source frame, read without mutation.
    right : pandas.DataFrame or None, optional
        Right source frame; default None relates ``left`` to itself, so each
        row's reference columns (the left columns of ``on``) are resolved
        against the key columns (the right columns) of the same table.
    on : mapping of str to str
        Nonempty key pairs, left column to right column; several pairs form a
        composite key. Rows with a missing key column are excluded and counted.
    compare : mapping of str to str or None, optional
        Attribute pairs, left column to right column, tested for agreement on
        matched keys; default None compares nothing.
    match : {'typed', 'text'}, optional
        How values match across tables. 'typed' (default) keeps booleans,
        numbers, strings and temporal kinds apart and matches numbers
        numerically; 'text' also matches integer-valued numbers to their
        decimal text.
    dropna : bool, optional
        Default True ignores missing compared values; False compares missing as
        a value.
    scopes, missing, table_ids : pair or None, optional
        Per-side source context as ``(left, right)``: Scopes (default None
        evaluates every row), sentinel mappings (default None) and table IDs
        (default ('left', 'right'), or ('table', 'table') for a self-reference).
    limits : RelateLimits or None, optional
        Output budget: ``example_limit`` (5) source rows per finding and side.
    **runtime : Unpack[Runtime]
        Optional runtime controls; see fieldwork.typing.Runtime.

    Returns
    -------
    Result
        Kind 'relation': per-side ``sides`` accounting, ``key_coverage`` in each
        direction, the matched-key ``relation``, one ``agreement`` record per
        compared pair, ``reciprocity`` for a self-reference (else None),
        ``warnings`` and findings. Inspect or select rows with ``side`` and
        ``right``. Units and states are defined in docs/algorithms.md.

    Raises
    ------
    KeyError
        A named column does not exist on its side.
    ValueError
        ``on`` is empty, a column repeats within ``on`` or ``compare``, or a
        per-side argument does not have two entries.
    TypeError
        ``on`` or ``compare`` is not a mapping, or a per-side argument is not a
        sequence.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> images = pd.DataFrame({"acc": [1, 1, 2, 3], "patient": ["a", "a", "b", "c"]})
    >>> clinical = pd.DataFrame({"acc": [1.0, 2.0, 4.0], "patient": ["a", "x", "d"]})
    >>> link = fw.relate(images, clinical, on={"acc": "acc"}, compare={"patient": "patient"})
    >>> link["key_coverage"]["left_to_right"]["state"], link["relation"]["type"]
    ('some', 'n:1')
    >>> link["agreement"][0]["state"]
    'some'
    """
    budget = budgets(limits, _LIMITS)
    keys = _column_pairs("on", on, required=True)
    compared = _column_pairs("compare", compare, required=False)
    if match not in ("typed", "text"):
        raise ValueError("match must be 'typed' or 'text'")
    if not isinstance(dropna, bool):
        raise TypeError("dropna must be boolean")
    self_reference = right is None
    ids = _pair("table_ids", table_ids, ("table",) * 2 if self_reference else _SIDES)
    if any(not isinstance(t, str) or not t for t in ids):
        raise ValueError("table_ids must be nonempty strings")
    evaluation = _Evaluation.build(
        left,
        right,
        keys,
        compared,
        match,
        dropna,
        _pair("scopes", scopes, (None, None)),
        _pair("missing", missing, (None, None)),
        ids,
    )
    base = evaluation.payload(budget["example_limit"])
    base["parameters"] = {
        "on": dict(keys),
        "compare": dict(compared),
        "match": match,
        "dropna": dropna,
        "limits": budget,
    }
    return Result("relation", base)


def _column_pairs(argument: str, mapping: Any, *, required: bool) -> list[tuple[str, str]]:
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{argument} must map left columns to right columns")
    pairs = [(str(a), str(b)) for a, b in mapping.items()]
    if required and not pairs:
        raise ValueError(f"{argument} must name at least one column pair")
    if len({a for a, _ in pairs}) != len(pairs) or len({b for _, b in pairs}) != len(pairs):
        raise ValueError(f"{argument} columns must not repeat on either side")
    return pairs


def _pair(argument: str, value: Any, default: tuple[Any, Any]) -> tuple[Any, Any]:
    """A per-side argument as (left, right); lists are accepted for saved recipes."""
    if value is None:
        return default
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{argument} must be a (left, right) pair")
    if len(value) != 2:
        raise ValueError(f"{argument} must be a (left, right) pair")
    return value[0], value[1]


def _match_key(value: Any, mode: str) -> tuple[int, Any]:
    """Cross-table identity: numbers match numerically; 'text' writes integers as text."""
    kind, identity = value_key(value)
    if kind in (1, 2):
        integral = kind == 1 or (math.isfinite(identity) and float(identity).is_integer())
        if mode == "text" and integral:
            return (3, str(int(identity)))
        # 1 and 1.0 are equal dictionary keys, so they share one identity.
        return (1, identity)
    return (kind, identity)


def _ids(encoded, lookup: dict, mode: str, kinds: set[str]) -> np.ndarray:
    """Match identity per row (-1 when missing), recording the value kinds seen."""
    values, codes = encoded
    table = np.full(len(values), -1, dtype=np.int64)
    for i, value in enumerate(values):
        if value is not None:
            key = _match_key(value, mode)
            table[i] = lookup.setdefault(key, len(lookup))
            kinds.add(_KINDS[key[0]])
    return table[codes]


def _joint(blocks: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """Joint ids of complete key tuples across row blocks; -1 marks an incomplete key."""
    stacked = np.concatenate(blocks)
    complete = (stacked >= 0).all(axis=1)
    ids = np.full(len(stacked), -1, dtype=np.int64)
    count = 0
    if complete.any():
        uniques, inverse = np.unique(stacked[complete], axis=0, return_inverse=True)
        ids[complete] = inverse.reshape(-1)
        count = len(uniques)
    return np.split(ids, np.cumsum([len(b) for b in blocks])[:-1]), count


def _counts(key: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(key[key >= 0], minlength=size)


def _at(per_key: np.ndarray, key: np.ndarray) -> np.ndarray:
    """A per-key flag for each row; rows with an incomplete key are False."""
    return (key >= 0) & per_key[np.maximum(key, 0)] if len(per_key) else np.zeros(len(key), bool)


def _values_per_key(key: np.ndarray, values: np.ndarray, size: int):
    """Distinct values per key, and the value of keys that have exactly one."""
    rows = (key >= 0) & (values >= 0)
    pairs = np.unique(np.stack([key[rows], values[rows]], axis=1), axis=0)
    count = np.bincount(pairs[:, 0], minlength=size)
    value = np.full(size, -1, dtype=np.int64)
    value[pairs[:, 0]] = pairs[:, 1]
    return count, value


def _state(hits: int, total: int) -> str:
    return "empty" if total == 0 else "all" if hits == total else "none" if hits == 0 else "some"


def _label(table: str, cols: Sequence[str]) -> str:
    return f"{table}.{cols[0]}" if len(cols) == 1 else f"{table}.({', '.join(cols)})"


@dataclass
class _Side:
    """One prepared side: source positions, key ids and compared value ids per row."""

    base: dict[str, Any]
    positions: np.ndarray
    key: np.ndarray
    values: list[np.ndarray]
    # Self-reference only: each row's key under the other side's columns (the
    # row's own key on the left, its reference on the right).
    other: np.ndarray | None


@dataclass
class _Evaluation:
    sides: tuple[_Side, _Side]
    size: int
    keys: list[tuple[str, str]]
    compared: list[tuple[str, str]]
    self_reference: bool
    warnings: list[dict[str, Any]]

    @classmethod
    def build(cls, left, right, keys, compared, mode, dropna, scopes, missing, table_ids):
        self_reference = right is None
        frames = (left, left if self_reference else right)
        named = []
        for i in range(2):
            own = [pair[i] for pair in keys]
            attributes = [pair[i] for pair in compared]
            columns(frames[i], own)
            columns(frames[i], attributes)
            extra = [pair[1 - i] for pair in keys] if self_reference else []
            named.append(list(dict.fromkeys([*own, *attributes, *extra])))
        prepared = []
        for i in range(2):
            if scopes[i] is not None and not isinstance(scopes[i], Scope):
                raise TypeError("scopes must hold Scope objects or None")
            prepared.append(
                prepare_values(
                    frames[i], named[i], scope=scopes[i], missing=missing[i], table_id=table_ids[i]
                )
            )
        warnings = []

        def mapped(pairs, which):
            lookups = [{} for _ in pairs]
            kinds = [(set(), set()) for _ in pairs]
            output = [[], []]
            with phase("matching values", len(pairs) * 2, "columns") as tracker:
                for j, pair in enumerate(pairs):
                    for i in range(2):
                        with focus(pair[i]):
                            values = prepared[i][2][pair[i]]
                            output[i].append(_ids(values, lookups[j], mode, kinds[j][i]))
                        tracker.advance(detail=pair[i])
            for j, pair in enumerate(pairs):
                if kinds[j][0] and kinds[j][1] and not kinds[j][0] & kinds[j][1]:
                    warnings.append(
                        {
                            "code": "VALUE_KIND_MISMATCH",
                            "role": which,
                            "column": pair[0],
                            "right_column": pair[1],
                            "left_kinds": sorted(kinds[j][0]),
                            "right_kinds": sorted(kinds[j][1]),
                        }
                    )
            return lookups, output

        lookups, key_ids = mapped(keys, "key")
        blocks = [np.column_stack(key_ids[0]), np.column_stack(key_ids[1])]
        if self_reference:
            # Each row's key under the other side's columns, in the same identities.
            for i in range(2):
                values = prepared[i][2]
                blocks.append(
                    np.column_stack(
                        [
                            _ids(values[pair[1 - i]], lookups[j], mode, set())
                            for j, pair in enumerate(keys)
                        ]
                    )
                )
        joint, size = _joint(blocks)
        value_lookups, value_ids = mapped(compared, "compare")
        if not dropna:
            # Missing is one more value, shared by both sides.
            for j, lookup in enumerate(value_lookups):
                for i in range(2):
                    value_ids[i][j] = np.where(value_ids[i][j] < 0, len(lookup), value_ids[i][j])

        def side(i: int) -> _Side:
            other = joint[2 + i] if self_reference else None
            return _Side(prepared[i][3], prepared[i][1], joint[i], value_ids[i], other)

        return cls((side(0), side(1)), size, keys, compared, self_reference, warnings)

    @cached_property
    def counts(self) -> tuple[np.ndarray, np.ndarray]:
        return _counts(self.sides[0].key, self.size), _counts(self.sides[1].key, self.size)

    @cached_property
    def matched(self) -> np.ndarray:
        return (self.counts[0] > 0) & (self.counts[1] > 0)

    @cached_property
    def agreements(self) -> list[dict[str, np.ndarray]]:
        """Per-key agreement states of each compared pair (only matched keys are set)."""
        output = []
        for j in range(len(self.compared)):
            (lc, lv), (rc, rv) = (
                _values_per_key(side.key, side.values[j], self.size) for side in self.sides
            )
            unavailable = self.matched & ((lc == 0) | (rc == 0))
            decisive = self.matched & (lc == 1) & (rc == 1)
            output.append(
                {
                    "agree": decisive & (lv == rv),
                    "disagree": decisive & (lv != rv),
                    "ambiguous": self.matched & ~unavailable & ~decisive,
                    "unavailable": unavailable,
                }
            )
        return output

    @cached_property
    def references(self) -> dict[str, np.ndarray]:
        """Self-reference edges own → reference of the referencing (left) rows."""
        left, right = self.sides
        assert left.other is not None and right.other is not None
        ref, own, size = left.key, left.other, max(self.size, 1)
        resolved = _at(self.counts[1] > 0, ref) & (own >= 0)
        loop = resolved & (own == ref)
        candidate = resolved & ~loop
        # A target row (key b, reference a) reciprocates the edge a → b.
        targets = (right.key >= 0) & (right.other >= 0)
        reverse = np.unique(right.key[targets] * size + right.other[targets])
        reciprocated = candidate & np.isin(ref * size + own, reverse)
        edges = own * size + ref
        return {
            "candidate": candidate,
            "reciprocated": reciprocated,
            "edges": np.unique(edges[candidate]),
            "reciprocated_edges": np.unique(edges[reciprocated]),
            "loops": np.unique(own[loop]),
            "incomplete_own": (ref >= 0) & (own < 0),
        }

    def masks(self, selector: Mapping[str, Any], side: int) -> tuple[np.ndarray, np.ndarray]:
        """(examples, exceptions) row masks of a finding on one side."""
        key = self.sides[side].key
        part = selector["part"]
        if part == "coverage":
            found = _at(self.matched, key)
            return found, (key >= 0) & ~found
        if part == "relation":
            matched = _at(self.matched, key)
            repeated = _at(self.counts[side] > 1, key)
            return matched & repeated, matched & ~repeated
        if part == "agreement":
            states = self.agreements[self.compared.index(tuple(selector["compare"]))]
            return _at(states["agree"], key), _at(states["disagree"], key)
        references = self.references
        return references["reciprocated"], references["candidate"] & ~references["reciprocated"]

    def payload(self, example_limit: int) -> dict[str, Any]:
        left, right = (side.base for side in self.sides)
        ids = [base["source"]["table_id"] for base in (left, right)]
        names = [[pair[i] for pair in self.keys] for i in range(2)]
        base = {
            **left,
            "status": "computed"
            if left["scope"]["evaluated_rows"] or right["scope"]["evaluated_rows"]
            else "empty",
            "self_reference": self.self_reference,
            "sides": {},
            "warnings": self.warnings,
            "findings": [],
        }
        for i, name in enumerate(_SIDES):
            key, counts = self.sides[i].key, self.counts[i]
            present = counts > 0
            matched_rows = int(_at(self.matched, key).sum())
            base["sides"][name] = {
                "source": self.sides[i].base["source"],
                "scope": self.sides[i].base["scope"],
                "missing_convention": self.sides[i].base["missing_convention"],
                "key": names[i],
                "compare": [pair[i] for pair in self.compared],
                "rows": {
                    "evaluated": len(key),
                    "incomplete_key": int((key < 0).sum()),
                    "matched": matched_rows,
                    "unmatched": int((key >= 0).sum()) - matched_rows,
                },
                "keys": {
                    "distinct": int(present.sum()),
                    "matched": int(self.matched.sum()),
                    "unmatched": int((present & ~self.matched).sum()),
                    "repeated": int((counts > 1).sum()),
                },
            }
        emit = _Findings(self, base, ids, example_limit)
        base["key_coverage"] = {}
        for direction, source in (("left_to_right", 0), ("right_to_left", 1)):
            base["key_coverage"][direction] = emit.coverage(direction, source)
        base["relation"] = emit.relation()
        base["agreement"] = [emit.agreement(j) for j in range(len(self.compared))]
        base["reciprocity"] = emit.reciprocity() if self.self_reference else None
        return base


class _Findings:
    """Measurement records and their findings, in a fixed order."""

    def __init__(self, evaluation: _Evaluation, base, ids, example_limit):
        self.evaluation, self.base, self.ids, self.limit = evaluation, base, ids, example_limit
        self.keys = [_label(ids[i], [pair[i] for pair in evaluation.keys]) for i in range(2)]

    def emit(self, part, statement, features, structure, measurements, sides, extra=None):
        identity = f"f{len(self.base['findings'])}"
        samples = []
        for side in sides:
            positions = self.evaluation.sides[side].positions
            examples, exceptions = self.evaluation.masks({"part": part, **(extra or {})}, side)
            samples.append(
                {
                    key: selection(rows, len(rows), self.limit)
                    for key, rows in (
                        ("examples", bounded_rows(positions, examples, self.limit)),
                        ("exceptions", bounded_rows(positions, exceptions, self.limit)),
                    )
                }
            )
        record = {
            "id": identity,
            "pattern": {
                "coverage": "key_coverage",
                "relation": "key_relation",
                "agreement": "value_agreement",
                "reciprocity": "reference_reciprocity",
            }[part],
            "statement": statement,
            "features": [{"table": self.ids[i], "column": c} for i, c in features],
            "counting_unit": "references" if part == "reciprocity" else "keys",
            "structure": structure,
            "measurements": measurements,
            **samples[0],
            "selector": {
                "dataset_id": self.base["source"]["dataset_id"],
                "scope_ref": "scope",
                "parameters_ref": "parameters",
                "missing_convention_ref": "missing_convention",
                "finding_id": identity,
                "part": part,
                "side": _SIDES[sides[0]],
                "sides": [_SIDES[side] for side in sides],
                **(extra or {}),
            },
        }
        if len(sides) > 1:
            record["other_side"] = {"side": _SIDES[sides[1]], **samples[1]}
        self.base["findings"].append(record)

    def key_features(self):
        return [(i, pair[i]) for i in range(2) for pair in self.evaluation.keys]

    def coverage(self, direction, source):
        evaluation = self.evaluation
        key = evaluation.sides[source].key
        present = evaluation.counts[source] > 0
        total, found = int(present.sum()), int(evaluation.matched.sum())
        found_rows = int(_at(evaluation.matched, key).sum())
        record = {
            "state": _state(found, total),
            "keys": total,
            "found": found,
            "not_found": total - found,
            "found_rows": found_rows,
            "not_found_rows": int((key >= 0).sum()) - found_rows,
        }
        source_key, target_key = self.keys[source], self.keys[1 - source]
        if evaluation.self_reference and source == 0:
            statements = {
                "all": f"Every {source_key} reference resolves to a {target_key} row",
                "some": f"Some {source_key} references resolve to a {target_key} row",
                "none": f"No {source_key} reference resolves to a {target_key} row",
                "empty": f"No row has a complete {source_key} reference",
            }
        elif evaluation.self_reference:
            statements = {
                "all": f"Every {source_key} key is referenced by {target_key}",
                "some": f"Some {source_key} keys are referenced by {target_key}",
                "none": f"No {source_key} key is referenced by {target_key}",
                "empty": f"No row has a complete {source_key} key",
            }
        else:
            statements = {
                "all": f"Every {source_key} key is found in {target_key}",
                "some": f"Some {source_key} keys are found in {target_key}",
                "none": f"No {source_key} key is found in {target_key}",
                "empty": f"No complete {source_key} key to look up in {target_key}",
            }
        statement = statements[record["state"]]
        self.emit(
            "coverage",
            statement,
            self.key_features(),
            {"direction": direction, "coverage": record["state"]},
            record,
            [source],
            {"direction": direction},
        )
        return record

    def relation(self):
        evaluation = self.evaluation
        matched = evaluation.matched
        left, right = (counts[matched] for counts in evaluation.counts)
        many = (bool((left > 1).any()), bool((right > 1).any()))
        record = {
            "type": _RELATIONS[many] if matched.any() else None,
            "matched_keys": int(matched.sum()),
            "joined_rows": int((left * right).sum()),
            "max_left_rows_per_key": int(left.max(initial=0)),
            "max_right_rows_per_key": int(right.max(initial=0)),
            "left_repeated_keys": int((left > 1).sum()),
            "right_repeated_keys": int((right > 1).sum()),
        }
        if record["type"] is not None:
            self.emit(
                "relation",
                f"{self.keys[0]} to {self.keys[1]} is {record['type']} on matched keys",
                self.key_features(),
                {"relation": record["type"]},
                record,
                [0, 1],
            )
        return record

    def agreement(self, j):
        evaluation = self.evaluation
        pair = evaluation.compared[j]
        counts = {state: int(mask.sum()) for state, mask in evaluation.agreements[j].items()}
        agree, disagree = counts["agree"], counts["disagree"]
        record = {
            "left": pair[0],
            "right": pair[1],
            "state": _state(agree, agree + disagree),
            "ambiguity": "some" if counts["ambiguous"] else "none",
            "matched_keys": int(evaluation.matched.sum()),
            "agreeing_keys": agree,
            "disagreeing_keys": disagree,
            "ambiguous_keys": counts["ambiguous"],
            "unavailable_keys": counts["unavailable"],
        }
        if record["matched_keys"]:
            names = f"{self.ids[0]}.{pair[0]} and {self.ids[1]}.{pair[1]}"
            statement = {
                "all": f"{names} agree on every comparable matched key",
                "some": f"{names} agree on some comparable matched keys",
                "none": f"{names} agree on no comparable matched key",
                "empty": f"{names} have no matched key with one value on each side",
            }[record["state"]]
            self.emit(
                "agreement",
                statement,
                [(0, pair[0]), (1, pair[1])],
                {"agreement": record["state"], "ambiguity": record["ambiguity"]},
                record,
                [0, 1],
                {"compare": list(pair)},
            )
        return record

    def reciprocity(self):
        references = self.evaluation.references
        total = len(references["edges"])
        reciprocated = len(references["reciprocated_edges"])
        record = {
            "state": _state(reciprocated, total),
            "self_references": "some" if len(references["loops"]) else "none",
            "resolved_references": total,
            "reciprocated": reciprocated,
            "not_reciprocated": total - reciprocated,
            "self_referencing_keys": len(references["loops"]),
            "incomplete_own_key_rows": int(references["incomplete_own"].sum()),
        }
        if total or len(references["loops"]):
            reference, key = self.keys
            statement = {
                "all": f"Every resolved {reference} reference is reciprocated by its {key} row",
                "some": f"Some resolved {reference} references are reciprocated",
                "none": f"No resolved {reference} reference is reciprocated",
                "empty": f"Every resolved {reference} reference points to its own row's key",
            }[record["state"]]
            self.emit(
                "reciprocity",
                statement,
                self.key_features(),
                {"reciprocity": record["state"], "self_references": record["self_references"]},
                record,
                [0],
            )
        return record


# Source-bound access for Result.inspect, select and recompute.


def saved_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Per-side scopes, sentinels and table IDs restored from saved evidence."""
    scopes, missing, ids = [], [], []
    for name in _SIDES:
        side = payload["sides"][name]
        scope, positions = side["scope"], side["scope"].get("selection_positions")
        scopes.append(
            Scope(side["source"]["dataset_id"], tuple(positions), scope["name"], scope["parent"])
            if positions is not None
            else None
        )
        missing.append(dict(side["missing_convention"]["sentinels"]))
        ids.append(side["source"]["table_id"])
    return {"scopes": tuple(scopes), "missing": tuple(missing), "table_ids": tuple(ids)}


def right_source(payload: Mapping[str, Any], left: pd.DataFrame, right: pd.DataFrame | None):
    """The verified right frame: ``left`` for a self-reference, else ``right``."""
    if payload["self_reference"]:
        if right is not None:
            raise ValueError("A self-reference has no separate right source; omit right")
        return left
    if right is None:
        raise ValueError("Pass right=, the right source frame, for this relation")
    if fingerprint(right) != payload["sides"]["right"]["source"]["dataset_id"]:
        raise ValueError("Right source dataset differs from the ordered analysis source")
    return right


def check_side(record: Mapping[str, Any], side: str) -> None:
    if side not in _SIDES:
        raise ValueError("side must be 'left' or 'right'")
    if side not in record["selector"]["sides"]:
        raise ValueError(f"Finding {record['id']!r} has no {side} rows")


def saved_rows(record: Mapping[str, Any], side: str, exceptions: bool) -> list[int]:
    """A finding's saved sample positions on one side."""
    check_side(record, side)
    samples = record if record["selector"]["side"] == side else record["other_side"]
    return samples["exceptions" if exceptions else "examples"]["positions"]


def matching_rows(
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
    left: pd.DataFrame,
    right: pd.DataFrame,
    side: str,
    exceptions: bool,
) -> list[int]:
    """Every source position on ``side`` matching a finding, recomputed from both sources."""
    check_side(record, side)
    parameters = payload["parameters"]
    context = saved_context(payload)
    evaluation = _Evaluation.build(
        left,
        None if payload["self_reference"] else right,
        list(parameters["on"].items()),
        list(parameters["compare"].items()),
        parameters["match"],
        parameters["dropna"],
        context["scopes"],
        context["missing"],
        context["table_ids"],
    )
    index = 0 if side == "left" else 1
    examples, rejected = evaluation.masks(record["selector"], index)
    positions = evaluation.sides[index].positions
    return positions[rejected if exceptions else examples].tolist()
