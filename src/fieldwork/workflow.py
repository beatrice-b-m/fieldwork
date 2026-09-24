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
    """Saved parameters of one analysis, to reapply to new deliveries.

    Construction checks that parameters are strict JSON (tuples load back as
    lists); each operation validates its own arguments when the recipe runs.

    Parameters
    ----------
    operation : str
        One of the names in Recipe.operations(): 'missingness', 'dependencies'
        (discover_dependencies), 'paths' (suggest_paths), 'value_patterns',
        'explore', 'profile', 'census', 'grain', 'levels', 'pairs',
        'joint_counts', 'infer_schema' or 'relate' (whose right frame is a run
        override: ``run(left, right=right)``).
    parameters : dict[str, Any], optional
        The operation's keyword arguments; default empty. A scope and runtime
        controls are not saved: pass them to run.
    notes : str, optional
        Free-text notes; default empty.
    version : str, optional
        Recipe format version; '1.0' (default) is the only one.

    Raises
    ------
    ValueError
        The version or operation is unsupported, parameters include a scope or
        runtime control, or contain nonfinite numbers.
    TypeError
        Parameters are not JSON serializable (a KeySpec, Scope or timestamp).

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
        if {"progress", "cancel", "timeout", "safe_errors"} & self.parameters.keys():
            raise ValueError(
                "Runtime controls belong in Recipe.run overrides, not saved parameters"
            )
        if {"scope", "scopes"} & self.parameters.keys():
            raise ValueError(
                "Recipes reapply to deliveries; pass a scope when running, not in the recipe"
            )
        json.dumps(self.to_dict(), allow_nan=False)

    @staticmethod
    def operations() -> dict[str, Callable[..., Result]]:
        """A new mapping from recipe operation names to the public analyses."""
        from ._explore import census, grain, infer_schema, joint_counts, levels, pairs, profile
        from .overview import explore
        from .relate import relate

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
            "relate": relate,
        }

    @operation("recipe")
    def run(
        self,
        df: pd.DataFrame,
        **overrides: Any,
    ) -> Result:
        """Run the operation on a delivery with the saved parameters.

        Parameters
        ----------
        df : pandas.DataFrame
            Delivery to analyze; any Scope override must come from this frame.
        **overrides : Any
            Keyword arguments replacing saved parameters for this run only (such
            as ``scope``), and the runtime controls of fieldwork.typing.Runtime.

        Returns
        -------
        Result
            The operation's result. The operation rejects invalid arguments.
        """
        return self.operations()[self.operation](df, **{**self.parameters, **overrides})

    def to_dict(self) -> dict[str, Any]:
        """The recipe as JSON-compatible fields: version, operation, parameters, notes.

        The parameters dictionary is shared with the recipe, not copied.
        """
        return {
            "version": self.version,
            "operation": self.operation,
            "parameters": self.parameters,
            "notes": self.notes,
        }

    def save(self, path: str | PathLike[str]) -> None:
        """Write the recipe as indented strict JSON, overwriting ``path``."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Recipe:
        """Read and validate a recipe saved by save.

        Raises
        ------
        ValueError
            The JSON is malformed or its version, operation or parameters are invalid.
        """
        return cls(**json.loads(Path(path).read_text()))


def compare(before: Result, after: Result) -> Result:
    """Compare populated fractions by feature across two missingness results.

    Parameters
    ----------
    before, after : Result
        Missingness results with the same counting unit, entity keys and
        aggregation; their sources and scopes may differ. Features align by name.

    Returns
    -------
    Result
        Kind 'comparison': per-feature ``changes`` with
        ``populated_fraction_delta`` (after minus before, as a fraction; None for
        a feature missing from one side or an empty denominator), findings, and
        both results' sources, scopes and conventions.

    Raises
    ------
    ValueError
        Either result is not missingness, or their counting units differ.

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
