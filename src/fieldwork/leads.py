"""Rank overview findings by how promising they are as investigation leads.

The scores are a deliberately simple heuristic, not a statistical measure. They
favor evidence an analyst would want to explain (near-rules with exceptions,
exclusive or empty columns, mixed formats) over trivially true or purely
descriptive findings (unique determinants, constant targets, uniform formats).
"""

from __future__ import annotations

from typing import Any

# Relationships that hold for any data of this shape; they carry no structure.
TRIVIAL_CONSTANT = "target is constant"
TRIVIAL_UNIQUE = "determinant is unique here"
TRIVIAL = {TRIVIAL_CONSTANT, TRIVIAL_UNIQUE}


def lead(record: dict[str, Any], constant_columns: set[str]) -> tuple[float, str]:
    """Return a (score, reason) pair for one finding; higher scores rank first."""
    pattern, m = record["pattern"], record["measurements"]
    if pattern == "approximate_dependency":
        if m.get("repeated_rows"):
            return 0.9, "near-rule with exceptions"
        return 0.2, "near-rule supported only by singleton groups"
    if pattern == "mutually_exclusive":
        return 0.85, "columns never populated together"
    if pattern == "availability":
        if not m["populated"]:
            return 0.75, "column is always missing"
        return 0.6, "column is partially populated"
    if pattern == "presence_implication":
        if m["exception_rate"]:
            return 0.7, "presence rule with exceptions"
        return 0.55, "presence rule"
    if pattern == "string_patterns":
        if m.get("format_count", 0) > 1:
            return 0.7, "mixed string formats"
        return 0.15, "uniform string format"
    if pattern == "value_alias":
        return 0.65, "equivalent encodings"
    if pattern == "numeric_range":
        if m.get("nonfinite"):
            return 0.6, "non-finite numbers"
        return 0.15, "numeric range"
    if pattern in {"numeric_offset", "numeric_ratio"}:
        return 0.6, "derived numeric column"
    if pattern == "similar_availability":
        return 0.6, "columns usually populated together"
    if pattern == "context_constancy":
        fraction = m.get("constant_group_fraction")
        if fraction is not None and 0 < fraction < 1:
            return 0.6, "constant in some contexts only"
        return 0.3, "context constancy"
    if pattern == "exact_dependency":
        if m["target"] in constant_columns:
            return 0.05, TRIVIAL_CONSTANT
        if m.get("repeated_rows"):
            return 0.5, "rule with repeated support"
        return 0.1, TRIVIAL_UNIQUE
    if pattern == "availability_family":
        return 0.5, "columns populated together"
    if pattern == "indexed_family":
        return 0.45, "indexed column family"
    if pattern == "availability_signature":
        if m.get("count") == m.get("denominator"):
            return 0.05, "single availability pattern"
        return 0.4, "availability pattern"
    if pattern == "census_path":
        return 0.2, "suggested census path"
    return 0.3, pattern.replace("_", " ")


def rank(findings: list[dict[str, Any]], constant_columns: set[str]) -> list[dict[str, Any]]:
    """Order findings by lead score (stable within ties) and annotate each.

    Repeats of the same pattern on the same leading column (for example one
    near-key determining many targets) are halved after the first, so the top
    of the list covers distinct leads. Equivalent encodings form chains across
    columns, so all but the first are halved regardless of column.
    """
    seen: dict[tuple[str, str], int] = {}
    scored = []
    for record in findings:
        score, reason = lead(record, constant_columns)
        leading = record["features"][0]["column"] if record["features"] else ""
        key = (record["pattern"], "" if record["pattern"] == "value_alias" else leading)
        if seen.get(key):
            score /= 2
        seen[key] = seen.get(key, 0) + 1
        scored.append((-score, len(scored), {**record, "lead": {"score": score, "reason": reason}}))
    return [record for *_, record in sorted(scored, key=lambda item: item[:2])]
