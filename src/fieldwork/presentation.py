"""Saved-evidence presentations with allowlisted topology projections."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from typing import Any, Literal

from ._explore.graphics import _SVG, _wrap
from ._explore.graphics import render_html as foundation_html
from ._explore.graphics import render_svg as foundation_svg
from ._explore.render import _clip, _safe
from ._explore.render import render_plaintext as foundation_text
from ._explore.result import ExplorerResult
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


def candidate_priority(candidate, *, legacy=False):
    roles = {
        "repeated grouping": 0,
        "unique identifier": 1,
        "constant": 2,
        "no evaluated support": 3,
    }
    return (
        roles[candidate_role(candidate)],
        -len(candidate["determines"] if legacy else candidate["determines_with_repeated_support"]),
        -candidate["repeated_rows"],
        len(candidate["columns"]),
        tuple(candidate["columns"]),
    )


def _dependency_measurements(record):
    """Recover only arithmetic supported by saved evidence; never mutate it."""
    row = dict(record)
    if "repeated_rows" not in row:
        fields = ("evaluated_rows", "evaluated_groups", "repeated_groups")
        row["repeated_rows"] = (
            row["evaluated_rows"] - (row["evaluated_groups"] - row["repeated_groups"])
            if all(row.get(k) is not None for k in fields)
            else None
        )
    repeated, evaluated = row["repeated_rows"], row.get("evaluated_rows")
    row.setdefault(
        "repeat_coverage", repeated / evaluated if evaluated and repeated is not None else None
    )
    row.setdefault(
        "repeat_modal_accuracy",
        1 - row["repair_rows"] / repeated
        if repeated and row.get("repair_rows") is not None
        else None,
    )
    for field in (
        "determinant_evaluated_rows",
        "target_observed_rows",
        "target_coverage",
        "target_missing_excluded_rows",
    ):
        row.setdefault(field, None)
    return row


def _candidate_summaries(data):
    """Use one comparable ranking for the whole saved candidate collection."""
    candidates = []
    for original in data.get("candidates", []):
        candidate = dict(original)
        records = [
            d
            for d in data.get("dependencies", [])
            if d.get("context") is None and d["determinant"] == candidate["columns"]
        ]
        by_target = {d["target"]: d for d in records}
        if "determines_with_repeated_support" not in candidate:
            exact_records = [by_target.get(target) for target in candidate["determines"]]
            candidate["determines_with_repeated_support"] = (
                [d["target"] for d in exact_records if d["repeated_groups"] > 0]
                if all(
                    d is not None and d.get("repeated_groups") is not None for d in exact_records
                )
                else None
            )
        candidate.setdefault(
            "global_targets_tested", len(records) if "dependencies" in data else None
        )
        features = data.get("parameters", {}).get("features")
        candidate.setdefault(
            "global_targets_possible",
            len(set(features) - set(candidate["columns"])) if features is not None else None,
        )
        candidate["role"] = candidate_role(candidate)
        candidates.append(candidate)
    legacy = any(c["determines_with_repeated_support"] is None for c in candidates)
    return sorted(candidates, key=lambda c: candidate_priority(c, legacy=legacy))


def _available(value):
    return "unavailable" if value is None else str(value)


def _candidate_explanation(candidate):
    supported = candidate["determines_with_repeated_support"]
    tested, possible = candidate["global_targets_tested"], candidate["global_targets_possible"]
    text = (
        f"{candidate['groups']} groups, {candidate['repeated_groups']} repeated; "
        f"{candidate['repeated_rows']} determinant repeated rows; "
        f"{len(candidate['determines'])} exact targets, "
        f"{_available(len(supported) if supported is not None else None)} with repeated support; "
        f"global targets tested {_available(tested)}/{_available(possible)}"
    )
    if tested is None or possible is None:
        text += " (test coverage unavailable)"
    elif tested < possible:
        text += " (incomplete; untested targets are unknown)"
    return text


def _dependency_explanation(record, dropna):
    row = _dependency_measurements(record)
    n, repeated = row.get("evaluated_rows"), row["repeated_rows"]
    text = (
        "Exact"
        if row.get("exact")
        else "Consistency unsupported"
        if row.get("exact") is None
        else "Approximate"
    ) + f" on {_available(n)} evaluated rows"
    q, observed = row["determinant_evaluated_rows"], row["target_observed_rows"]
    text += (
        f"; target observed on {observed}/{q} determinant-eligible rows"
        if q is not None and observed is not None
        else "; observed target coverage unavailable"
    )
    if row["target_coverage"] is not None:
        text += f" (observed target coverage {row['target_coverage']:.3g})"
    if repeated == 0:
        text += "; no repeated groups after exclusions; repeat-only consistency not assessable"
    elif repeated is None:
        text += "; repeated support unavailable; repeat-only consistency unavailable"
    else:
        text += f"; {repeated}/{n} rows in repeated groups"
        if row["repeat_coverage"] is not None:
            text += f" (repeat coverage {row['repeat_coverage']:.3g})"
        accuracy = row["repeat_modal_accuracy"]
        text += (
            f"; repeat-only consistency {accuracy:.3g}"
            if accuracy is not None
            else "; repeat-only consistency unavailable"
        )
    if dropna is False:
        text += "; missing values participated in consistency measurements"
    return text


def _dependency_label(row):
    from .evidence import context_statement

    label = ", ".join(row["determinant"]) + " → " + row["target"]
    if row.get("context"):
        label += " within " + context_statement(row["context"])
    return label


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


def visualization_data(
    result: ExplorerResult | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Literal["full", "topology"] = "full",
) -> dict[str, Any]:
    """Project saved evidence into presentation data with explicit disclosure.

    Parameters
    ----------
    result : ExplorerResult or mapping
        Saved analytical result or its ordinary dictionary export. Rendering does
        not need the source dataframe and does not recompute analyses. Restore a
        compact envelope with from_dict before rendering.
    section : str or None, optional
        Analytical section to render; default None. Foundation explore defaults
        to grain; discovery overviews default to the combined overview. Unrequested
        or incompatible sections raise ValueError.
    detail : {'full', 'topology'}, optional
        Default 'full' includes measurements and evidence. 'topology' allowlists
        structural labels and qualitative relationships, suppressing quantities,
        row positions, and distribution statistics. It does not anonymize labels.

    Returns
    -------
    dict[str, Any]
        Fresh allowlisted presentation projection; not an analytical round-trip
        export. Use to_dict for full serialization. Full dependency results include
        candidate summaries and all completed dependency tests, even below the
        finding threshold. Legacy repeat measurements are recovered when possible;
        unavailable target coverage remains None, without changing saved data.

    Raises
    ------
    ValueError
        Detail or requested section is invalid or unsupported.

    Notes
    -----
    Display limits do not change the saved analytical population or search
    coverage. Topology is a disclosure projection, not anonymization: feature
    names and structural value labels can remain identifying. HTML/SVG labels
    and terminal control characters are escaped before presentation.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> fw.visualization_data(result, detail="topology")["detail"]
    'topology'
    """
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
            if record["pattern"] in {"exact_dependency", "approximate_dependency"}:
                dependency_data = data.get("sections", {}).get("dependencies", data)
                row["measurements"] = _dependency_measurements(record["measurements"])
                row["explanation"] = _dependency_explanation(
                    record["measurements"], dependency_data.get("parameters", {}).get("dropna")
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
    if data["kind"] == "dependencies" and detail == "full":
        output["candidates"] = _candidate_summaries(data)
        output["dependencies"] = [
            {
                **_dependency_measurements(d),
                "explanation": _dependency_explanation(d, data.get("parameters", {}).get("dropna")),
            }
            for d in data.get("dependencies", [])
        ]
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
                        {
                            k: c[k]
                            for k in (
                                "groups",
                                "repeated_groups",
                                "repeated_rows",
                                "determines",
                                "determines_with_repeated_support",
                                "global_targets_tested",
                                "global_targets_possible",
                            )
                        }
                        if detail == "full"
                        else {}
                    ),
                }
                for c in (
                    _candidate_summaries(sections["dependencies"])
                    if detail == "full"
                    else sections["dependencies"].get("candidates", [])
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
    result: ExplorerResult | Mapping[str, Any],
    *,
    width: int = 100,
    max_lines: int = 200,
    max_nodes: int = 1000,
    detail: Literal["full", "topology"] = "full",
    missing_label: str = "<NA>",
    unicode_mode: Literal["safe", "display"] = "safe",
) -> str:
    """Render saved evidence as bounded, escaped plaintext.

    Parameters
    ----------
    result : ExplorerResult or mapping
        Saved analytical result or its ordinary dictionary export. Rendering does
        not need the source dataframe and does not recompute analyses. Restore a
        compact envelope with from_dict before rendering.
    width : int, optional
        Positive maximum plaintext line width; default 100. Long lines are clipped.
    max_lines : int, optional
        Positive maximum output lines; default 200. Truncation is marked in output.
    max_nodes : int, optional
        Nonnegative per-list display budget; default 1000. Applies to findings,
        candidates, and completed dependency tests. Zero omits these details,
        not analytical evidence in the saved result.
    detail : {'full', 'topology'}, optional
        Default 'full' includes measurements and evidence. 'topology' allowlists
        structural labels and qualitative relationships, suppressing quantities,
        row positions, and distribution statistics. It does not anonymize labels.
    missing_label : str, optional
        Displayed foundation missing-value label; default "<NA>". Discovery
        findings retain their producer wording.
    unicode_mode : {'safe', 'display'}, optional
        Default 'safe' escapes non-ASCII text. 'display' preserves Unicode and
        requires fieldwork[unicode] for width calculation. Control characters are
        escaped in either mode.

    Returns
    -------
    str
        Bounded, terminal-safe plaintext. Nothing is printed or written.

    Raises
    ------
    ValueError
        Detail, Unicode mode, or display limits are invalid.
    ImportError
        unicode_mode='display' requires the optional unicode extra.

    Notes
    -----
    Display limits do not change the saved analytical population or search
    coverage. Topology is a disclosure projection, not anonymization: feature
    names and structural value labels can remain identifying. HTML/SVG labels
    and terminal control characters are escaped before presentation.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> isinstance(fw.render_plaintext(result), str)
    True
    """
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
                lines.append(text)
                lines.extend(
                    "    " + part for part in _candidate_explanation(candidate).split("; ")
                )
            else:
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
    if "candidates" in data:
        lines.append("Candidate grains")
        for candidate in data["candidates"][:max_nodes]:
            lines.append("  " + ", ".join(candidate["columns"]) + ": " + candidate["role"])
            lines.extend("  " + part for part in _candidate_explanation(candidate).split("; "))
    if "dependencies" in data:
        lines.append("Completed dependency tests (including below finding threshold)")
        for row in data["dependencies"][:max_nodes]:
            lines.append(_dependency_label(row))
            lines.extend("  " + part for part in row["explanation"].split("; "))
        if len(data["dependencies"]) > max_nodes:
            lines.append("... more dependency tests not rendered (max_nodes)")
    for row in [] if data["kind"] == "overview" else data["findings"][:max_nodes]:
        lines.append((f"[{row['id']}] " if detail == "full" else "") + row["statement"])
        lines.append("  Analysis: " + _unit_label(row["analysis_unit"]))
        if detail == "full":
            if "explanation" in row:
                lines.extend("  " + part for part in row["explanation"].split("; "))
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
    result: ExplorerResult | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Literal["full", "topology"] = "full",
    view: Literal["map", "matrix", "mapping", "association", "bars", "tree", "heatmap", "findings"]
    | None = None,
    show_exceptions: bool = False,
    max_findings: int = 12,
) -> str:
    """Render saved evidence as a self-contained static SVG.

    Parameters
    ----------
    result : ExplorerResult or mapping
        Saved analytical result or its ordinary dictionary export. Rendering does
        not need the source dataframe and does not recompute analyses. Restore a
        compact envelope with from_dict before rendering.
    section : str or None, optional
        Analytical section to render; default None. Foundation explore defaults
        to grain; discovery overviews default to the combined overview. Unrequested
        or incompatible sections raise ValueError.
    detail : {'full', 'topology'}, optional
        Default 'full' includes measurements and evidence. 'topology' allowlists
        structural labels and qualitative relationships, suppressing quantities,
        row positions, and distribution statistics. It does not anonymize labels.
    view : str or None, optional
        Default None chooses the result's default view. Grain: 'map' or 'matrix';
        pairs: 'mapping' or 'association'; levels: 'bars'; census: 'tree';
        joint counts: 'heatmap'; discovery: 'findings'. Association requires full
        detail. Other result/view combinations raise ValueError.
    show_exceptions : bool, optional
        Default False. True includes supported exact-dependency exception edges
        in the foundation grain map; other result kinds do not use this option.
    max_findings : int, optional
        Nonnegative per-list discovery display limit; default 12. Full dependency
        views show up to five candidate summaries and this many completed tests,
        including tests below the finding threshold. Zero omits these lists.
        Foundation rendering uses its saved analytical bounds instead.

    Returns
    -------
    str
        Self-contained SVG markup. Nothing is written to disk.

    Raises
    ------
    ValueError
        Detail, view, section, or finding limit is invalid or unsupported.

    Notes
    -----
    Display limits do not change the saved analytical population or search
    coverage. Topology is a disclosure projection, not anonymization: feature
    names and structural value labels can remain identifying. HTML/SVG labels
    and terminal control characters are escaped before presentation.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> "<svg" in fw.render_svg(result)
    True
    """
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
        rows = data["findings"][:max_findings]
        if detail == "full" and data["kind"] in {"overview", "dependencies"}:
            candidates = data.get("candidates", data.get("overview", {}).get("grains", []))
            rows = [
                {
                    "statement": ", ".join(c["columns"]) + ": " + c["role"],
                    "analysis_unit": {"counting_unit": "rows"},
                    "counting_unit": "rows",
                    "measurements": {},
                    "explanation": _candidate_explanation(c),
                }
                for c in candidates[: min(5, max_findings)]
            ]
            if data["kind"] == "dependencies":
                rows += [
                    {
                        "statement": _dependency_label(d),
                        "analysis_unit": {"counting_unit": "rows"},
                        "counting_unit": "rows",
                        "measurements": {},
                        "explanation": d["explanation"],
                    }
                    for d in data["dependencies"][:max_findings]
                ]
            else:
                rows += data["findings"][:max_findings]
        for row in rows:
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
            if detail == "full" and "explanation" in row:
                metrics = row["explanation"]
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


def render_html(
    result: ExplorerResult | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Literal["full", "topology"] = "full",
    max_findings: int = 100,
) -> str:
    """Render a standalone interactive HTML evidence document.

    Parameters
    ----------
    result : ExplorerResult or mapping
        Saved analytical result or its ordinary dictionary export. Rendering does
        not need the source dataframe and does not recompute analyses. Restore a
        compact envelope with from_dict before rendering.
    section : str or None, optional
        Analytical section to render; default None. Foundation explore defaults
        to grain; discovery overviews default to the combined overview. Unrequested
        or incompatible sections raise ValueError.
    detail : {'full', 'topology'}, optional
        Default 'full' includes measurements and evidence. 'topology' allowlists
        structural labels and qualitative relationships, suppressing quantities,
        row positions, and distribution statistics. It does not anonymize labels.
    max_findings : int, optional
        Nonnegative per-list discovery display limit; default 100. Applies to
        findings, candidates, and completed dependency tests, including tests
        below the finding threshold. Zero omits these lists.
        Foundation rendering uses its saved analytical bounds instead.

    Returns
    -------
    str
        Standalone HTML with embedded styles/graphics and local controls.
        Nothing is written to disk; no server or remote assets are required.

    Raises
    ------
    ValueError
        Detail, section, or finding limit is invalid or unsupported.

    Notes
    -----
    Display limits do not change the saved analytical population or search
    coverage. Topology is a disclosure projection, not anonymization: feature
    names and structural value labels can remain identifying. HTML/SVG labels
    and terminal control characters are escaped before presentation.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> result = fw.missingness(pd.DataFrame({"x": [1, None]}))
    >>> "<!doctype html>" in fw.render_html(result)
    True
    """
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
    if detail == "full":
        candidates = projected.get("candidates", projected.get("overview", {}).get("grains", []))
        if candidates:
            parts.append("<h2>Candidate grains</h2>")
            for candidate in candidates[:max_findings]:
                parts.append(
                    "<p>"
                    + html.escape(
                        ", ".join(candidate["columns"])
                        + ": "
                        + candidate["role"]
                        + "; "
                        + _candidate_explanation(candidate)
                    )
                    + "</p>"
                )
        if "dependencies" in projected:
            parts.append("<h2>Completed dependency tests (including below finding threshold)</h2>")
            for row in projected["dependencies"][:max_findings]:
                parts.append(
                    "<details><summary>"
                    + html.escape(_dependency_label(row))
                    + "</summary><p>"
                    + html.escape(row["explanation"])
                    + "</p>"
                    + _html_evidence({k: v for k, v in row.items() if k != "explanation"})
                    + "</details>"
                )
            if len(projected["dependencies"]) > max_findings:
                parts.append("<p>More dependency tests available; display limit reached.</p>")
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
            if "explanation" in row:
                parts.append("<p>" + html.escape(row["explanation"]) + "</p>")
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
