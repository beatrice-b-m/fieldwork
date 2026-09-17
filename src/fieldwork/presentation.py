"""Saved-evidence presentations with allowlisted topology projections."""

from __future__ import annotations

import html
import json

from ._explore.graphics import _SVG, _wrap
from ._explore.graphics import render_html as foundation_html
from ._explore.graphics import render_svg as foundation_svg
from ._explore.render import _clip, _safe
from ._explore.render import render_plaintext as foundation_text
from ._explore.visual_data import visualization_data as foundation_data
from .evidence import limit

KINDS = {"missingness", "dependencies", "paths", "value_patterns", "overview", "comparison"}


def _data(result, section=None):
    data = (
        {"kind": result.kind, "schema_version": result.schema_version, **result.payload}
        if hasattr(result, "payload")
        else result
    )
    if section and data.get("kind") == "overview":
        if section not in data["sections"]:
            raise ValueError(f"Unknown section: {section}")
        return data["sections"][section]
    return data


def visualization_data(result, *, section=None, detail="full"):
    data = _data(result, section)
    if data.get("kind") not in KINDS:
        return foundation_data(
            data,
            section=None if section and _data(result).get("kind") == "overview" else section,
            detail=detail,
        )
    if detail not in {"full", "topology"}:
        raise ValueError("detail must be 'full' or 'topology'")
    output = {"kind": data["kind"], "detail": detail, "findings": []}
    for record in data.get("findings", []):
        row = {
            "pattern": record["pattern"],
            "statement": record["statement"],
            "features": record["features"],
        }
        if detail == "full":
            row.update(
                measurements=record["measurements"],
                counting_unit=record["counting_unit"],
                examples=record["examples"],
                exceptions=record["exceptions"],
            )
        output["findings"].append(row)
    if detail == "full":
        for key in (
            "scope",
            "coverage",
            "availability",
            "signatures",
            "entities",
            "contexts",
            "changes",
        ):
            if key in data:
                output[key] = data[key]
    else:
        # Frequency-ranked evidence is canonicalized; no support or selectors are exported.
        output["findings"].sort(key=lambda r: (r["pattern"], r["statement"]))
    return output


def render_plaintext(
    result,
    *,
    width=100,
    max_lines=200,
    max_nodes=1000,
    detail="full",
    missing_label="<NA>",
    unicode_mode="safe",
):
    if _data(result).get("kind") not in KINDS:
        return foundation_text(
            result,
            width=width,
            max_lines=max_lines,
            max_nodes=max_nodes,
            detail=detail,
            missing_label=missing_label,
            unicode_mode=unicode_mode,
        )
    for name, value, minimum in [
        ("width", width, 1),
        ("max_lines", max_lines, 1),
        ("max_nodes", max_nodes, 0),
    ]:
        limit(name, value, minimum=minimum)
    data = visualization_data(result, detail=detail)
    lines = [f"Fieldwork · {data['kind']}"]
    if detail == "topology":
        lines.append("Topology only · quantitative evidence suppressed")
    else:
        lines.append(f"Population: {data.get('scope', {}).get('evaluated_rows', 0)} rows")
    for row in data["findings"][:max_nodes]:
        lines.append(row["statement"])
        if detail == "full":
            metrics = ", ".join(
                f"{k}={v}"
                for k, v in row["measurements"].items()
                if not isinstance(v, (dict, list))
            )
            lines.append("  " + metrics)
            lines.append(
                f"  Examples: {row['examples']['positions']}; exceptions: {row['exceptions']['positions']}"
            )
    if len(data["findings"]) > max_nodes:
        lines.append("... more findings not rendered (max_nodes)")
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["... more output not rendered (max_lines)"]
    return "\n".join(_clip(_safe(line, unicode_mode), width, unicode_mode) for line in lines)


def render_svg(
    result, *, section=None, detail="full", view=None, show_exceptions=False, max_findings=12
):
    data = _data(result, section)
    if data.get("kind") not in KINDS:
        return foundation_svg(
            data,
            section=None if _data(result).get("kind") == "overview" else section,
            detail=detail,
            view=view,
            show_exceptions=show_exceptions,
        )
    if view not in {None, "findings"}:
        raise ValueError("Discovery results support view='findings'")
    limit("max_findings", max_findings)
    data = visualization_data(data, detail=detail)
    svg = _SVG(f"Fieldwork {data['kind']}")
    svg.text(24, 36, f"Fieldwork / {data['kind'].replace('_', ' ').title()}", size=23, bold=True)
    svg.text(
        24,
        62,
        "Topology only"
        if detail == "topology"
        else f"{data.get('scope', {}).get('evaluated_rows', 0)} evaluated rows · saved evidence",
        size=13,
    )
    y = 92
    if detail == "full" and "availability" in data:
        for row in data["availability"][:max_findings]:
            for line in _wrap(row["feature"], 28):
                svg.text(24, y + 18, line, size=13)
                y += 18
            fraction = row["populated_fraction"] or 0
            svg.rect(270, y - 1, 400, 17, fill="#e2e8f0")
            if fraction:
                svg.rect(270, y - 1, round(400 * fraction, 2), 17, fill="#66a89b")
            svg.text(685, y + 13, f"{row['populated']}/{row['denominator']}", size=12)
            y += 28
    else:
        for row in data["findings"][:max_findings]:
            lines = _wrap(row["statement"], 90)
            metrics = ""
            if detail == "full":
                metrics = " · ".join(
                    f"{k}: {v:.3g}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in row["measurements"].items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                )
            detail_lines = _wrap(metrics, 102) if metrics else []
            height = 26 + len(lines) * 20 + len(detail_lines) * 17
            svg.rect(20, y, 820, height)
            cursor = y + 24
            for line in lines:
                svg.text(34, cursor, line, bold=True)
                cursor += 20
            for line in detail_lines:
                svg.text(34, cursor, line, size=12)
                cursor += 17
            y += height + 10
    total = (
        len(data.get("availability", data["findings"]))
        if detail == "full"
        else len(data["findings"])
    )
    if total > max_findings:
        svg.text(
            24,
            y + 14,
            f"More evidence available · display limit {max_findings}"
            if detail == "full"
            else "More evidence available",
            size=12,
        )
        y += 30
    return svg.finish(864, y + 20)


def render_html(result, *, section=None, detail="full", max_findings=100):
    data = _data(result, section)
    if data.get("kind") not in KINDS:
        return foundation_html(
            data,
            section=None if _data(result).get("kind") == "overview" else section,
            detail=detail,
        )
    limit("max_findings", max_findings)
    projected = visualization_data(data, detail=detail)
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        "<title>Fieldwork evidence</title><style>body{font:16px system-ui;background:#f6f8fb;color:#193345;max-width:1000px;margin:2rem auto;padding:1rem}details{background:white;border:1px solid #cbd5df;border-radius:8px;padding:1rem;margin:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{max-width:100%;height:auto}</style><body>",
        render_svg(data, detail=detail, max_findings=min(12, max_findings)),
        "<h1>Inspect findings</h1>",
    ]
    for row in projected["findings"][:max_findings]:
        parts.append(
            "<details><summary>"
            + html.escape(row["statement"])
            + "</summary><pre>"
            + html.escape(json.dumps(row, indent=2, ensure_ascii=False))
            + "</pre></details>"
        )
    if len(projected["findings"]) > max_findings:
        parts.append("<p>More findings available; display limit reached.</p>")
    parts.append("</body></html>")
    return "".join(parts)
