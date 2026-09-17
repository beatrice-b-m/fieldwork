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
from .evidence import limit, qualitative_analysis_unit


def structural_evidence(structure):
    return {
        k: structure[k]
        for k in ("context", "present", "absent", "entity_keys", "presence_pattern", "relation")
        if k in structure
    }


def _qualitative_unit(unit):
    return {
        k: unit[k] for k in ("counting_unit", "entity_keys", "presence_aggregation") if k in unit
    }


def candidate_role(candidate):
    if not candidate["evaluated_rows"] or not candidate["groups"]:
        return "no evaluated support"
    if candidate["groups"] == 1:
        return "constant"
    if candidate["unique"]:
        return "unique identifier"
    return "repeated grouping"


def candidate_priority(candidate):
    roles = {
        "repeated grouping": 0,
        "unique identifier": 1,
        "constant": 2,
        "no evaluated support": 3,
    }
    return (
        roles[candidate_role(candidate)],
        -len(candidate["determines"]),
        -candidate["repeated_rows"],
        len(candidate["columns"]),
        tuple(candidate["columns"]),
    )


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
        if data["sections"][section].get("status") == "not_requested":
            raise ValueError(f"Section {section!r} was not requested")
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
            "structure": structural_evidence(record.get("structure", {})),
            "counting_unit": record["counting_unit"],
            "analysis_unit": qualitative_analysis_unit(data, record),
        }
        if detail == "full":
            row.update(
                id=record["id"],
                measurements=record["measurements"],
                examples=record["examples"],
                exceptions=record["exceptions"],
            )
        output["findings"].append(row)
    if "analysis_unit" in data:
        output["analysis_unit"] = _qualitative_unit(data["analysis_unit"])
    for side in ("before", "after"):
        scope = data.get(f"{side}_scope")
        unit = data.get(f"{side}_analysis_unit")
        if scope is not None:
            output[f"{side}_scope"] = (
                scope
                if detail == "full"
                else {k: scope[k] for k in ("name", "parent") if k in scope}
            )
        if unit is not None:
            output[f"{side}_analysis_unit"] = unit if detail == "full" else _qualitative_unit(unit)
    if detail == "full":
        for key in (
            "analysis_unit",
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
    if "feature_network" in data:
        network = dict(data["feature_network"])
        findings = {record["id"]: record for record in data.get("findings", [])}
        network["relationships"] = []
        for edge in data["feature_network"]["relationships"]:
            record = findings.get(edge.get("evidence", {}).get("overview_finding_id"))
            if "analysis_unit" not in edge and record is not None:
                edge = {**edge, "analysis_unit": qualitative_analysis_unit(data, record)}
            network["relationships"].append(edge)
        output["feature_network"] = (
            network
            if detail == "full"
            else {
                "nodes": network["nodes"],
                "components": network["components"],
                "relationships": sorted(
                    [
                        {
                            k: structural_evidence(edge[k])
                            if k == "structure"
                            else _qualitative_unit(edge[k])
                            if k == "analysis_unit"
                            else edge[k]
                            for k in (
                                "kind",
                                "analysis_unit",
                                "features",
                                "structure",
                                "determinant",
                                "source",
                                "target",
                            )
                            if k in edge
                        }
                        for edge in network["relationships"]
                    ],
                    key=lambda edge: json.dumps(edge, sort_keys=True),
                ),
            }
        )
    if data["kind"] == "overview":
        sections = data["sections"]
        if "section_selection" in data:
            output["section_selection"] = data["section_selection"]
        output["overview"] = {
            "families": [
                family["features"] for family in sections["missingness"].get("families", [])
            ],
            "paths": [path["dimensions"] for path in sections["paths"].get("paths", [])],
            "grains": [
                {
                    "columns": c["columns"],
                    "role": candidate_role(c),
                    **(
                        {"groups": c["groups"], "repeated_groups": c["repeated_groups"]}
                        if detail == "full"
                        else {}
                    ),
                }
                for c in sorted(
                    sections["dependencies"].get("candidates", []), key=candidate_priority
                )
            ],
            "signatures": [
                {
                    "present": sig["present"],
                    "absent": sig["absent"],
                    **({"count": sig["count"]} if detail == "full" else {}),
                }
                for sig in sections["missingness"].get("signatures", [])
            ],
        }
        if detail == "topology":
            for name in output["overview"]:
                output["overview"][name].sort(key=lambda row: json.dumps(row, sort_keys=True))
    if detail == "topology":
        # Frequency-ranked evidence is canonicalized; no support or selectors are exported.
        output["findings"].sort(key=lambda r: (r["pattern"], r["statement"]))
    return output


def _unit_label(unit):
    label = unit["counting_unit"]
    if "denominator" in unit:
        label = f"{unit['denominator']} {label}"
    if unit.get("entity_keys"):
        label += "; keys=" + ", ".join(unit["entity_keys"])
    if "presence_aggregation" in unit:
        label += "; presence=" + unit["presence_aggregation"]
    return label


def _comparison_labels(data):
    labels = []
    for side in ("before", "after"):
        scope = data.get(f"{side}_scope")
        if scope is None:
            continue
        label = f"{side.title()}: {scope['name']}"
        if "evaluated_rows" in scope:
            label += f"; {scope['evaluated_rows']} evaluated rows"
        labels.append(label)
        unit = data.get(f"{side}_analysis_unit")
        if unit:
            labels.append(f"  {side.title()} analysis: " + _unit_label(unit))
    return labels


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
    elif data["kind"] != "comparison":
        lines.append(f"Population: {data.get('scope', {}).get('evaluated_rows', 0)} rows")
    if data.get("section_selection", {}).get("omitted"):
        lines.append("Not requested: " + ", ".join(data["section_selection"]["omitted"]))
    lines.extend(_comparison_labels(data))
    if "analysis_unit" in data and data["kind"] != "comparison":
        lines.append("Analysis: " + _unit_label(data["analysis_unit"]))
    if data["kind"] == "overview":
        overview = data["overview"]
        lines.append("Availability families")
        lines.extend("  " + ", ".join(group) for group in overview["families"][: min(5, max_nodes)])
        lines.append("Major availability signatures")
        for signature in overview["signatures"][: min(5, max_nodes)]:
            text = "  Present: " + (", ".join(signature["present"]) or "none")
            if detail == "full":
                text += f" ({signature['count']} {data.get('analysis_unit', {}).get('counting_unit', 'rows')})"
            lines.append(text)
        lines.append("Candidate grains")
        for candidate in overview["grains"][: min(5, max_nodes)]:
            text = "  " + ", ".join(candidate["columns"]) + ": " + candidate["role"]
            if detail == "full":
                text += f"; {candidate['groups']} groups, {candidate['repeated_groups']} repeated"
            lines.append(text)
        lines.append("Suggested census paths")
        lines.extend("  " + " > ".join(path) for path in overview["paths"][:max_nodes])
        if "feature_network" in data:
            lines.append("Connected feature evidence")
            for group in data["feature_network"]["components"][: min(5, max_nodes)]:
                lines.append("  " + ", ".join(group))
        lines.append(
            "Summary lists are limited; individual sections retain complete evidence and coverage."
        )
    for row in [] if data["kind"] == "overview" else data["findings"][:max_nodes]:
        lines.append((f"[{row['id']}] " if detail == "full" else "") + row["statement"])
        lines.append("  Analysis: " + _unit_label(row["analysis_unit"]))
        if detail == "full":
            if "explanation" in row["measurements"]:
                lines.extend(
                    "  " + reason for reason in row["measurements"]["explanation"].split("; ")
                )
            metrics = ", ".join(
                f"{k}={v}"
                for k, v in row["measurements"].items()
                if not isinstance(v, (dict, list))
            )
            lines.append(f"  Unit: {row['counting_unit']}; " + metrics)
            for feature in row["measurements"].get("availability", []):
                lines.append(
                    f"  {feature['feature']}: {feature['populated']}/{feature['denominator']} populated {row['counting_unit']}"
                )
            lines.append(
                f"  Examples: {row['examples']['positions']}; exceptions: {row['exceptions']['positions']}"
            )
    if data["kind"] != "overview" and len(data["findings"]) > max_nodes:
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
        else "Availability comparison · before → after"
        if data["kind"] == "comparison"
        else (
            f"{data.get('scope', {}).get('evaluated_rows', 0)} evaluated rows · "
            + (
                f"{data['analysis_unit']['denominator']} {data['analysis_unit']['counting_unit']} · {data['analysis_unit']['presence_aggregation']}"
                if "analysis_unit" in data
                else "saved evidence"
            )
        ),
        size=13,
    )
    y = 92
    context_labels = _comparison_labels(data)
    if data.get("section_selection", {}).get("omitted"):
        context_labels.append("Not requested: " + ", ".join(data["section_selection"]["omitted"]))
    if "analysis_unit" in data and data["kind"] != "comparison":
        context_labels.append("Analysis: " + _unit_label(data["analysis_unit"]))
    for label in context_labels:
        for line in _wrap(label, 102):
            svg.text(24, y, line, size=13)
            y += 20
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
            metrics = _unit_label(row["analysis_unit"])
            if detail == "full":
                metrics = (
                    row["counting_unit"]
                    + " · "
                    + " · ".join(
                        f"{k}: {v:.3g}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in row["measurements"].items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    )
                )
            if detail == "full" and row["measurements"].get("explanation"):
                metrics = row["measurements"]["explanation"]
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
    if projected["kind"] == "comparison":
        for side in ("before", "after"):
            if f"{side}_scope" in projected:
                parts.append(
                    f"<h2>{side.title()} population</h2>"
                    + _html_evidence(
                        {
                            "scope": projected[f"{side}_scope"],
                            "analysis_unit": projected.get(f"{side}_analysis_unit", {}),
                        }
                    )
                )
    elif "analysis_unit" in projected:
        parts.append("<h2>Analysis population</h2>" + _html_evidence(projected["analysis_unit"]))
    if projected.get("section_selection", {}).get("omitted"):
        parts.append(
            "<p>Not requested: "
            + html.escape(", ".join(projected["section_selection"]["omitted"]))
            + "</p>"
        )
    if "feature_network" in projected:
        parts.append("<h2>Browse feature connections</h2>")
        for node in projected["feature_network"]["nodes"]:
            parts.append("<details><summary>" + html.escape(node["column"]) + "</summary><ul>")
            for edge in projected["feature_network"]["relationships"]:
                if node not in edge["features"]:
                    continue
                label = (
                    edge["kind"].replace("_", " ")
                    + ": "
                    + ", ".join(f["column"] for f in edge["features"])
                )
                if edge.get("determinant"):
                    label += " (" + ", ".join(edge["determinant"]) + " → " + edge["target"] + ")"
                if edge.get("analysis_unit"):
                    label += " · " + _unit_label(edge["analysis_unit"])
                parts.append("<li>" + html.escape(label))
                if detail == "full":
                    finding_id = edge["evidence"]["overview_finding_id"]
                    if any(f["id"] == finding_id for f in projected["findings"][:max_findings]):
                        parts.append(
                            ' · <a href="#'
                            + html.escape(finding_id, quote=True)
                            + '">inspect evidence</a>'
                        )
                    else:
                        parts.append(" · evidence outside display limit")
                if edge.get("structure"):
                    parts.append(_html_evidence(edge["structure"]))
                parts.append("</li>")
            parts.append("</ul></details>")
    for row in projected["findings"][:max_findings]:
        anchor = f' id="{html.escape(row["id"], quote=True)}"' if detail == "full" else ""
        parts.append(
            "<details" + anchor + "><summary>" + html.escape(row["statement"]) + "</summary>"
        )
        parts.append("<p>Analysis: " + html.escape(_unit_label(row["analysis_unit"])) + "</p>")
        if detail == "full":
            parts.append(
                "<p>Finding "
                + html.escape(row["id"])
                + " · counting unit: "
                + html.escape(row["counting_unit"])
                + "</p>"
            )
            parts.append(_html_evidence(row["measurements"]))
            parts.append(
                "<h3>Representative source rows</h3>"
                + _html_evidence({"examples": row["examples"], "exceptions": row["exceptions"]})
            )
        if row.get("structure"):
            parts.append(_html_evidence(row["structure"]))
        parts.append("</details>")
    if len(projected["findings"]) > max_findings:
        parts.append("<p>More findings available; display limit reached.</p>")
    parts.append("</body></html>")
    return "".join(parts)


def _html_evidence(value):
    """Readable saved evidence without making readers interpret serialized dictionaries."""
    if isinstance(value, dict):
        return (
            "<table>"
            + "".join(
                "<tr><th style='text-align:left;vertical-align:top;padding-right:1rem'>"
                + html.escape(str(key).replace("_", " "))
                + "</th><td>"
                + _html_evidence(item)
                + "</td></tr>"
                for key, item in value.items()
            )
            + "</table>"
        )
    if isinstance(value, list):
        if any(isinstance(item, (dict, list)) for item in value):
            return (
                "<ul>"
                + "".join("<li>" + _html_evidence(item) + "</li>" for item in value)
                + "</ul>"
            )
        return html.escape(", ".join(str(item) for item in value) or "none")
    if isinstance(value, float):
        return f"{value:.4g}"
    return html.escape("undefined" if value is None else str(value))
