"""Composition and portable recipes delegate to public analytical operations."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._explore.orchestration import explore as explicit_explore
from ._runtime import operation, phase
from .availability import missingness
from .discovery import discover_dependencies
from .evidence import finding, foundation_context, prepare, result
from .families import feature_network
from .navigation import suggest_paths
from .patterns import value_patterns


@operation("overview")
def explore(df, dimensions=None, *, discovery=None, sections=None, section_options=None, **options):
    """Explicit dimensions preserve the foundation API; omitted dimensions discover an overview."""
    if dimensions is not None:
        if sections is not None or section_options is not None:
            raise ValueError("sections and section_options apply to discovery overviews only")
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
    if options.keys() - {"scope", "missing", "table_id", "features"}:
        raise TypeError("With omitted dimensions, configure search with discovery dictionary")
    # Run-time population overrides retain all other configured search settings.
    config = {**(discovery or {}), **options}
    if {"progress", "cancel", "timeout"} & config.keys():
        raise ValueError("Pass runtime controls directly to explore, not in discovery")
    shared = {k: v for k, v in config.items() if k in {"scope", "missing", "table_id", "features"}}
    availability_options = {
        k: config.pop(k) for k in ("entity", "unit", "entity_presence") if k in config
    }
    contexts = config.pop("by", None)
    names = ("missingness", "dependencies", "paths", "value_patterns")
    requested = list(names if sections is None else sections)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(names):
        raise ValueError(f"sections must contain distinct names from {names}")
    section_options = dict(section_options or {})
    if section_options.keys() - set(requested):
        raise ValueError("section_options must refer to requested sections")
    protected = {"scope", "missing", "table_id", "progress", "cancel", "timeout"}
    functions = {
        "paths": suggest_paths,
        "missingness": missingness,
        "dependencies": discover_dependencies,
        "value_patterns": value_patterns,
    }
    configs = {
        "paths": dict(config),
        "missingness": {"by": contexts, **availability_options, **shared},
        "dependencies": {"by": contexts, "max_key_size": 1, "max_candidates": 20, **shared},
        "value_patterns": {"by": contexts, "max_pairs": 20, **shared},
    }
    # Validate all options before starting potentially expensive work.
    inspect.signature(suggest_paths).bind(df, **config)
    for name in requested:
        overrides = dict(section_options.get(name, {}))
        if protected & overrides.keys() or any(k.startswith("_") for k in overrides):
            raise ValueError("section_options cannot override source context or runtime controls")
        configs[name].update(overrides)
        inspect.signature(functions[name]).bind(df, **configs[name])
    analyses = {}
    for name in ("paths", "missingness", "dependencies", "value_patterns"):
        if name in requested:
            analyses[name] = functions[name](df, **configs[name])
    if "missingness" in analyses:
        base = dict(analyses["missingness"].payload)
    else:
        base = prepare(df, features=[], **{k: v for k, v in shared.items() if k != "features"})[-1]
    base["sections"] = {
        name: analyses[name].to_dict() if name in analyses else {"status": "not_requested"}
        for name in names
    }
    paths = analyses.get("paths")
    if paths is not None and paths.best:
        base["sections"]["census"] = paths["paths"][0]["preview"]
    if sections is not None or section_options:
        base["section_selection"] = {
            "requested": [n for n in names if n in requested],
            "omitted": [n for n in names if n not in requested],
        }
    base["findings"] = []
    for section in names:
        if section not in analyses:
            continue
        analysis = analyses[section]
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
    with phase("assembling overview"):
        base["feature_network"] = feature_network(base)
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

    @operation("recipe")
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

    def analysis_unit(analysis):
        return analysis.payload.get(
            "analysis_unit",
            {
                "counting_unit": "rows",
                "entity_keys": [],
                "denominator": analysis["scope"]["evaluated_rows"],
                "presence_aggregation": "per_row",
            },
        )

    def counting(analysis):
        unit = analysis_unit(analysis)
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
        "before_scope": before["scope"],
        "after_scope": after["scope"],
        "before_analysis_unit": analysis_unit(before),
        "after_analysis_unit": analysis_unit(after),
        "before_convention": before["missing_convention"],
        "after_convention": after["missing_convention"],
        "changes": records,
        "analysis_unit": analysis_unit(after),
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
