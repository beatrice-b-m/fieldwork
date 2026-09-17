"""Composition and portable recipes delegate to public analytical operations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._explore.orchestration import explore as explicit_explore
from .availability import missingness
from .discovery import discover_dependencies
from .evidence import finding, foundation_context, result
from .navigation import suggest_paths
from .patterns import value_patterns


def explore(df, dimensions=None, *, discovery=None, **options):
    """Explicit dimensions preserve the foundation API; omitted dimensions discover an overview."""
    if dimensions is not None:
        config = dict(discovery or {})
        incompatible = config.keys() - {"scope", "missing", "table_id", "features"}
        if incompatible:
            raise ValueError(
                f"Explicit dimensions do not accept discovery search options: {sorted(incompatible)}"
            )
        duplicate = config.keys() & options.keys()
        if duplicate:
            raise ValueError(f"Configuration supplied twice: {sorted(duplicate)}")
        options = {**config, **options}
        context = {k: options.pop(k) for k in ("scope", "missing", "table_id") if k in options}
        if context:
            return foundation_context(df, explicit_explore, dimensions, **context, **options)
        return explicit_explore(df, dimensions, **options)
    if options:
        raise TypeError("With omitted dimensions, configure search with discovery dictionary")
    paths = suggest_paths(df, **(discovery or {}))
    shared = {
        k: v
        for k, v in (discovery or {}).items()
        if k in {"scope", "missing", "table_id", "features"}
    }
    availability = missingness(df, **shared)
    dependencies = discover_dependencies(df, max_key_size=1, max_candidates=20, **shared)
    base = {
        **availability.payload,
        "sections": {
            "missingness": availability.to_dict(),
            "dependencies": dependencies.to_dict(),
            "paths": paths.to_dict(),
        },
    }
    if paths.best:
        base["sections"]["census"] = paths["paths"][0]["preview"]
    base["findings"] = []
    for section, analysis in [
        ("missingness", availability),
        ("dependencies", dependencies),
        ("paths", paths),
    ]:
        for record in analysis["findings"]:
            base["findings"].append(
                {
                    **record,
                    "id": f"f{len(base['findings'])}",
                    "selector": {
                        **record["selector"],
                        "analysis_section": section,
                        "scope_ref": f"sections.{section}.scope",
                        "parameters_ref": f"sections.{section}.parameters",
                        "missing_convention_ref": f"sections.{section}.missing_convention",
                    },
                }
            )
    return result("overview", base)


@dataclass(frozen=True)
class Recipe:
    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if self.version != "1.0" or self.operation not in self.operations():
            raise ValueError("Unsupported recipe version or operation")
        if "scope" in self.parameters:
            raise ValueError(
                "Recipes reapply to deliveries; pass a scope when running, not in the recipe"
            )
        json.dumps(self.to_dict(), allow_nan=False)

    @staticmethod
    def operations():
        from ._explore import census, grain, joint_counts, levels

        return {
            "missingness": missingness,
            "dependencies": discover_dependencies,
            "paths": suggest_paths,
            "value_patterns": value_patterns,
            "explore": explore,
            "census": census,
            "grain": grain,
            "levels": levels,
            "joint_counts": joint_counts,
        }

    def run(self, df, **overrides):
        return self.operations()[self.operation](df, **{**self.parameters, **overrides})

    def to_dict(self):
        return {
            "version": self.version,
            "operation": self.operation,
            "parameters": self.parameters,
            "notes": self.notes,
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def compare(before, after):
    """Compare per-feature populated fractions across scopes or deliveries."""
    if before.kind != "missingness" or after.kind != "missingness":
        raise ValueError("compare accepts two missingness results")

    def counting(analysis):
        unit = analysis.payload.get("analysis_unit", {})
        if unit.get("counting_unit", "rows") == "rows":
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
        "before_convention": before["missing_convention"],
        "after_convention": after["missing_convention"],
        "changes": records,
        "analysis_unit": after.payload.get(
            "analysis_unit",
            {
                "counting_unit": "rows",
                "denominator": after["scope"]["evaluated_rows"],
                "presence_aggregation": "per_row",
            },
        ),
        "findings": [],
    }
    for record in records:
        finding(
            base,
            "availability_change",
            f"{record['feature']}: availability across deliveries",
            [record["feature"]],
            record,
            [],
            example_limit=0,
            unit=counting(after)[0],
        )
    return result("comparison", base)
