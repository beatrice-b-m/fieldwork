"""Portable recipes and delivery comparisons, delegating to public analyses."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from ._runtime import operation
from .availability import missingness
from .discovery import discover_dependencies
from .evidence import finding, result
from .navigation import suggest_paths
from .patterns import value_patterns
from .result import Result


@dataclass(frozen=True)
class Recipe:
    """Store reusable, strict-JSON parameters for an allowlisted operation.

    Parameters
    ----------
    operation : str
        One of 'missingness', 'dependencies', 'paths', 'value_patterns', 'explore',
        'profile', 'census', 'grain', 'levels', 'pairs', 'joint_counts' or
        'infer_schema'.
        Use 'dependencies' for discover_dependencies and 'paths' for suggest_paths.
    parameters : dict[str, Any], optional
        Operation keyword arguments; default is a new empty dictionary. Must be
        strict JSON. Source-bound scope and progress/cancel/timeout controls
        belong in run overrides, not persisted parameters.
    notes : str, optional
        Free-text notes; default empty string.
    version : str, optional
        Recipe format version; default and only supported value is '1.0'.

    Attributes
    ----------
    operation : str
        Allowlisted operation identifier.
    parameters : dict[str, Any]
        Saved options. Nested parameters are mutable despite the frozen record.
    notes : str
        User notes retained in saved JSON.
    version : str
        Recipe format version, independent of evidence/package versions.

    Raises
    ------
    ValueError
        Version/operation is unsupported, runtime controls/scope are persisted,
        or parameters contain nonfinite JSON numbers.
    TypeError
        Parameters contain values not serializable as JSON, such as KeySpec,
        Scope, timestamps, or callbacks.

    Notes
    -----
    Recipes reapply parameters to new deliveries; evidence and scopes retain old
    source identities. Construction validates serializability, not every operation
    argument. Operation-specific validation occurs at run time. JSON decoding
    converts tuples to lists. Runtime overrides do not mutate saved parameters.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> recipe = fw.Recipe('missingness', {'features': ['value']})
    >>> recipe.run(pd.DataFrame({'value': [1, None]})).kind
    'missingness'
    """

    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if self.version != "1.0" or self.operation not in self.operations():
            raise ValueError("Unsupported recipe version or operation")
        if {"progress", "cancel", "timeout"} & self.parameters.keys():
            raise ValueError(
                "Runtime controls belong in Recipe.run overrides, not saved parameters"
            )
        if "scope" in self.parameters:
            raise ValueError(
                "Recipes reapply to deliveries; pass a scope when running, not in the recipe"
            )
        json.dumps(self.to_dict(), allow_nan=False)

    @staticmethod
    def operations() -> dict[str, Callable[..., Result]]:
        """Return the supported recipe operation registry.

        Returns
        -------
        dict[str, Callable[..., Result]]
            New mapping from persisted operation names to public callables. Editing
            this returned dictionary does not register or replace operations.
        """
        from ._explore import census, grain, infer_schema, joint_counts, levels, pairs, profile
        from .overview import explore

        return {
            "missingness": missingness,
            "dependencies": discover_dependencies,
            "paths": suggest_paths,
            "value_patterns": value_patterns,
            "explore": explore,
            "profile": profile,
            "census": census,
            "grain": grain,
            "levels": levels,
            "pairs": pairs,
            "joint_counts": joint_counts,
            "infer_schema": infer_schema,
        }

    @operation("recipe")
    def run(
        self,
        df: pd.DataFrame,
        **overrides: Any,
    ) -> Result:
        """Apply saved parameters to a delivery, with explicit overrides.

        Parameters
        ----------
        df : pandas.DataFrame
            Delivery to analyze; may differ from prior recipe runs. Supply any Scope
            override created from this delivery.
        **overrides : Any
            Operation-specific keyword overrides, such as scope or example_limit, and
            the progress, cancel and timeout controls of fieldwork.typing.Runtime.
            Explicit keys replace saved parameters without modifying the recipe. For
            automatic explore, scope/missing/table_id/features replace corresponding
            discovery entries while preserving other discovery settings.

        Returns
        -------
        Result
            Result of the named operation. Discovery operations return
            Result; paths returns Result. The concrete type depends
            on the recipe's runtime operation and, for explore, its dimensions.

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
        Runtime controls are never saved into evidence or recipe parameters. Nested
        analyses share this call's cancellation/progress context. Invalid or unsupported
        operation arguments are rejected by the selected operation.
        """
        return self.operations()[self.operation](df, **{**self.parameters, **overrides})

    def to_dict(self) -> dict[str, Any]:
        """Export the recipe configuration as ordinary JSON-compatible fields.

        Returns
        -------
        dict[str, Any]
            Version, operation, parameters, and notes. The top-level mapping is new;
            the parameters dictionary remains shared with the recipe.
        """
        return {
            "version": self.version,
            "operation": self.operation,
            "parameters": self.parameters,
            "notes": self.notes,
        }

    def save(self, path: str | PathLike[str]) -> None:
        """Write the recipe as indented strict JSON.

        Parameters
        ----------
        path : str or os.PathLike[str]
            Destination file. Existing contents are overwritten; parent directories
            are not created.

        Returns
        -------
        None
            Writes the complete recipe followed by a newline.

        Raises
        ------
        OSError
            The destination cannot be written.
        TypeError or ValueError
            Mutated parameters are no longer strict-JSON serializable.
        """
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Recipe:
        """Read and validate a saved recipe JSON file.

        Parameters
        ----------
        path : str or os.PathLike[str]
            Existing recipe JSON file to read.

        Returns
        -------
        Recipe
            Restored recipe with validated version, operation, and strict JSON options.

        Raises
        ------
        OSError
            The file cannot be read.
        ValueError
            JSON is malformed or the recipe version/operation/parameters are invalid.
        TypeError
            Required constructor fields are absent or extra fields are supplied.
        """
        return cls(**json.loads(Path(path).read_text()))


def compare(before: Result, after: Result) -> Result:
    """Compare populated fractions by feature across two availability results.

    Parameters
    ----------
    before : Result
        Earlier missingness result. Features align by name, not position.
    after : Result
        Later missingness result with compatible counting unit, entity keys, and
        aggregation. Source deliveries and scopes may differ.

    Returns
    -------
    Result
        Kind 'comparison', with changes, findings, and both source identities,
        conventions, scopes, and analysis units. populated_fraction_delta is
        after minus before; absent features or empty denominators yield None.

    Raises
    ------
    ValueError
        Either result is not missingness, or counting units/entity aggregation
        differ.

    Notes
    -----
    Delta is a fraction (0.25 means 25 percentage points), not relative percent
    change. A comparison preserves evidence about both populations; it does not
    establish that their selection or missing conventions are equivalent.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> before = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> after = fw.missingness(pd.DataFrame({"x": [1, 2]}))
    >>> fw.compare(before, after)["changes"][0]["populated_fraction_delta"]
    0.5
    """
    if before.kind != "missingness" or after.kind != "missingness":
        raise ValueError("compare accepts two missingness results")

    def counting(analysis):
        unit = analysis["analysis_unit"]
        if unit["counting_unit"] == "rows":
            return ("rows",)
        return ("entities", unit["entity_keys"], unit["presence_aggregation"])

    if counting(before) != counting(after):
        raise ValueError("Comparison requires the same analysis unit, entity keys and aggregation")
    left = {r["feature"]: r for r in before["availability"]}
    right = {r["feature"]: r for r in after["availability"]}
    records = []
    for c in sorted(left.keys() | right.keys()):
        a, b = left.get(c), right.get(c)
        delta = (
            b["populated_fraction"] - a["populated_fraction"]
            if a and b and a["denominator"] and b["denominator"]
            else None
        )
        records.append({"feature": c, "before": a, "after": b, "populated_fraction_delta": delta})
    base = {
        "source": after["source"],
        "scope": after["scope"],
        "before_source": before["source"],
        "after_source": after["source"],
        "before_scope": before["scope"],
        "after_scope": after["scope"],
        "before_analysis_unit": before["analysis_unit"],
        "after_analysis_unit": after["analysis_unit"],
        "before_convention": before["missing_convention"],
        "after_convention": after["missing_convention"],
        "changes": records,
        "analysis_unit": after["analysis_unit"],
        "findings": [],
    }
    for record in records:
        finding(
            base,
            "availability_change",
            f"{record['feature']}: availability before → after",
            [record["feature"]],
            record,
            [],
            example_limit=0,
            unit=counting(after)[0],
        )
    return result("comparison", base)
