"""Saved-evidence presentations with allowlisted topology projections."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import quote

from ._explore.graphics import _SVG, _esc, _wrap
from ._explore.graphics import render_html as foundation_html
from ._explore.graphics import render_svg as foundation_svg
from ._explore.render import _clip, _safe
from ._explore.render import render_plaintext as foundation_text
from ._explore.visual_data import visualization_data as foundation_data
from ._html import collection, document
from .evidence import limit, qualitative_analysis_unit
from .result import Result


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
        -len(candidate["determines_with_repeated_support"]),
        -candidate["repeated_rows"],
        len(candidate["columns"]),
        tuple(candidate["columns"]),
    )


def _candidate_summaries(data):
    """Use one comparable ranking for the whole saved candidate collection."""
    candidates = [{**c, "role": candidate_role(c)} for c in data.get("candidates", [])]
    return sorted(candidates, key=candidate_priority)


def _collapse_equivalent(candidates):
    """Merge single-column candidates that partition the same rows identically.

    Mutual determination only implies the same partition on the same evaluated
    population, so group counts and evaluated rows must also match.
    """
    kept = []
    for candidate in candidates:
        columns = candidate["columns"]
        twin = next(
            (
                k
                for k in kept
                if len(columns) == 1
                and len(k["columns"]) == 1
                and k["groups"] == candidate["groups"]
                and k["evaluated_rows"] == candidate["evaluated_rows"]
                and columns[0] in k.get("determines", [])
                and k["columns"][0] in candidate.get("determines", [])
            ),
            None,
        )
        if twin is None:
            kept.append({**candidate, "equivalent": []})
        else:
            twin["equivalent"].append(columns[0])
    return kept


def _grain_title(candidate):
    title = ", ".join(candidate["columns"])
    if candidate.get("equivalent"):
        title += " (equivalent: " + ", ".join(candidate["equivalent"]) + ")"
    return title


def _signature_label(signature):
    present, absent = signature["present"], signature["absent"]
    if not absent:
        return "All populated"
    if len(absent) <= len(present):
        return "Missing: " + ", ".join(absent)
    return "Only: " + (", ".join(present) or "none")


def _candidate_explanation(candidate):
    supported = candidate["determines_with_repeated_support"]
    tested, possible = candidate["global_targets_tested"], candidate["global_targets_possible"]
    text = (
        f"{candidate['groups']} groups, {candidate['repeated_groups']} repeated; "
        f"{candidate['repeated_rows']} determinant repeated rows; "
        f"{len(candidate['determines'])} exact targets, {len(supported)} with repeated support; "
        f"global targets tested {tested}/{possible}"
    )
    if tested < possible:
        text += " (incomplete; untested targets are unknown)"
    return text


def _dependency_explanation(row, dropna):
    n, repeated = row["evaluated_rows"], row["repeated_rows"]
    text = (
        "Exact"
        if row["exact"]
        else "Consistency unsupported"
        if row["exact"] is None
        else "Approximate"
    ) + f" on {n} evaluated rows"
    q, observed = row["determinant_evaluated_rows"], row["target_observed_rows"]
    text += f"; target observed on {observed}/{q} determinant-eligible rows"
    if row["target_coverage"] is not None:
        text += f" (observed target coverage {row['target_coverage']:.3g})"
    if repeated == 0:
        text += "; no repeated groups after exclusions; repeat-only consistency not assessable"
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
    result: Result | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Literal["full", "topology"] = "full",
) -> dict[str, Any]:
    """Project saved evidence into presentation data with explicit disclosure.

    Parameters
    ----------
    result : Result or mapping
        Saved analytical result or its dictionary export. Rendering does not need
        the source dataframe and does not recompute analyses.
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
        finding threshold.
        Full overviews include section_coverage keyed by analytical section;
        each section retains its own search and retention limits.

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
            if "lead" in record:
                row["lead"] = record["lead"]
            if record["pattern"] in {"exact_dependency", "approximate_dependency"}:
                dependency_data = data.get("sections", {}).get("dependencies", data)
                row["explanation"] = _dependency_explanation(
                    record["measurements"], dependency_data.get("parameters", {}).get("dropna")
                )
        output["findings"].append(row)
    if "analysis_unit" in data:
        output["analysis_unit"] = _qualitative_unit(data["analysis_unit"])
    if data.get("skipped_features"):
        output["skipped_features"] = data["skipped_features"]
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
                **d,
                "explanation": _dependency_explanation(d, data.get("parameters", {}).get("dropna")),
            }
            for d in data.get("dependencies", [])
        ]
    if data["kind"] == "overview":
        sections = data["sections"]
        if detail == "full":
            output["section_coverage"] = {
                name: section["coverage"]
                for name, section in sections.items()
                if "coverage" in section
            }
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
                    "equivalent": c["equivalent"],
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
                for c in _collapse_equivalent(
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


def _coverage_lines(data):
    """Describe saved work bounds without implying exhaustive discovery."""
    coverages = data.get("section_coverage", {data["kind"]: data.get("coverage", {})})
    lines = []
    pairs = (
        ("candidates_evaluated", "candidate_space", "candidate keys evaluated"),
        ("dependency_tests", "dependency_tests_possible", "dependency tests completed"),
        ("pairs_evaluated", "pair_candidates", "pairs evaluated"),
        ("features_evaluated", "features_requested", "features evaluated"),
        ("contexts_evaluated", "contexts_total", "contexts evaluated"),
        ("contexts_shown", "contexts_total", "contexts retained"),
        ("signatures_shown", "signatures_total", "signatures retained"),
    )
    for section, coverage in coverages.items():
        for done, possible, label in pairs:
            if done in coverage and possible in coverage:
                n, total = coverage[done], coverage[possible]
                if total or n:
                    suffix = " (limited)" if n < total else ""
                    lines.append(f"{section}: {n}/{total} {label}{suffix}")
        if coverage.get("search_exhausted_budget"):
            lines.append(f"{section}: path search reached its candidate budget")
    return lines


def _skipped_label(skipped):
    return "Skipped columns with unsupported values: " + ", ".join(
        f"{record['feature']} ({record['value_type']})" for record in skipped
    )


def _sample_label(label, sample):
    positions = sample["positions"]
    total = sample.get("total")
    count = f"{len(positions)}/{total}" if total is not None else str(len(positions))
    return f"{label}: {count} saved source positions {positions}"


def render_plaintext(
    result: Result | Mapping[str, Any],
    *,
    width: int = 100,
    max_lines: int = 200,
    max_nodes: int = 1000,
    detail: Literal["full", "topology"] = "full",
    missing_label: str = "<NA>",
    unicode_mode: Literal["safe", "display"] = "display",
) -> str:
    """Render saved evidence as bounded, escaped plaintext.

    Parameters
    ----------
    result : Result or mapping
        Saved analytical result or its dictionary export. Rendering does not need
        the source dataframe and does not recompute analyses.
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
        Default 'display' preserves Unicode; width uses wcwidth when the optional
        fieldwork[unicode] extra is installed, otherwise East Asian width rules.
        'safe' escapes non-ASCII text. Control and bidirectional override
        characters are escaped in either mode.

    Returns
    -------
    str
        Bounded, terminal-safe plaintext. Nothing is printed or written.

    Raises
    ------
    ValueError
        Detail, Unicode mode, or display limits are invalid.

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
    if data.get("skipped_features"):
        lines.append(_skipped_label(data["skipped_features"]))
    lines.extend(_comparison_labels(data))
    if detail == "full":
        coverage = _coverage_lines(data)
        if data["kind"] == "overview":
            limited = any("(limited)" in line or "budget" in line for line in coverage)
            lines.append(
                "Search limits reached; review section coverage."
                if limited
                else "Search coverage is saved per section; untested work is unknown."
            )
        elif coverage:
            lines.append("Search coverage (untested work is unknown)")
            lines.extend("  " + line for line in coverage)
    if not data["findings"]:
        lines.append("No findings saved; review coverage and analysis settings.")
    if "analysis_unit" in data and data["kind"] != "comparison":
        lines.append("Analysis: " + _unit_label(data["analysis_unit"]))
    if data["kind"] == "overview":
        overview = data["overview"]
        unit = data.get("analysis_unit", {}).get("counting_unit", "rows")
        if detail == "full":
            leads = [row for row in data["findings"] if row.get("lead", {}).get("score", 0) >= 0.3]
            lines.append("Leads (inspect with result.inspect(df, id))")
            lines.extend(
                f"  [{row['id']}] {row['statement']} ({row['lead']['reason']})"
                for row in leads[: min(8, max_nodes)]
            )
            if not leads:
                lines.append("  none stood out; browse the sections below")
        lines.append("Candidate grains")
        for candidate in overview["grains"][: min(5, max_nodes)]:
            text = "  " + _grain_title(candidate) + ": " + candidate["role"]
            if detail == "full":
                text += (
                    f"; {candidate['groups']} groups, {len(candidate['determines'])} exact"
                    f" targets, {len(candidate['determines_with_repeated_support'])}"
                    " with repeated support"
                )
            lines.append(text)
        lines.append("Suggested census paths")
        lines.extend("  " + " > ".join(path) for path in overview["paths"][:max_nodes])
        lines.append("Availability families")
        lines.extend("  " + ", ".join(group) for group in overview["families"][: min(5, max_nodes)])
        if not overview["families"]:
            lines.append("  none")
        lines.append("Major availability signatures")
        for signature in overview["signatures"][: min(5, max_nodes)]:
            text = "  " + _signature_label(signature)
            if detail == "full":
                text += f" ({signature['count']} {unit})"
            lines.append(text)
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
        if len(data["candidates"]) > max_nodes:
            lines.append("... more candidate grains not rendered (max_nodes)")
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
            lines.append("  " + _sample_label("Examples", row["examples"]))
            lines.append("  " + _sample_label("Exceptions", row["exceptions"]))
    if data["kind"] != "overview" and len(data["findings"]) > max_nodes:
        lines.append("... more findings not rendered (max_nodes)")
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["... more output not rendered (max_lines)"]
    return "\n".join(_clip(_safe(line, unicode_mode), width, unicode_mode) for line in lines)


def render_svg(
    result: Result | Mapping[str, Any],
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
    result : Result or mapping
        Saved analytical result or its dictionary export. Rendering does not need
        the source dataframe and does not recompute analyses.
    section : str or None, optional
        Analytical section to render; default None. Foundation explore defaults
        to grain; discovery overviews default to the combined overview. Unrequested
        or incompatible sections raise ValueError.
    detail : {'full', 'topology'}, optional
        Default 'full' includes measurements and evidence. 'topology' allowlists
        structural labels and qualitative relationships, suppressing quantities,
        row positions, and distribution statistics. It does not anonymize labels.
    view : str or None, optional
        Default None chooses the result's default view. Grain: 'map' or 'matrix'
        (feature rows, candidate-key columns);
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
    if data.get("skipped_features"):
        context_labels.append(_skipped_label(data["skipped_features"]))
    if "analysis_unit" in data and data["kind"] != "comparison":
        context_labels.append("Analysis: " + _unit_label(data["analysis_unit"]))
    for label in context_labels:
        for line in _wrap(label, 102):
            svg.text(24, y, line, size=13)
            y += 20
    displayed_lists = []
    if detail == "full" and "availability" in data:
        displayed_lists.append(("availability features", len(data["availability"]), max_findings))
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
        displayed_lists.append(("findings", len(data["findings"]), max_findings))
        if detail == "full" and data["kind"] in {"overview", "dependencies"}:
            candidates = data.get("candidates", data.get("overview", {}).get("grains", []))
            displayed_lists.append(("candidate grains", len(candidates), min(5, max_findings)))
            rows = [
                {
                    "statement": _grain_title(c) + ": " + c["role"],
                    "analysis_unit": {"counting_unit": "rows"},
                    "counting_unit": "rows",
                    "measurements": {},
                    "explanation": _candidate_explanation(c),
                }
                for c in candidates[: min(5, max_findings)]
            ]
            if data["kind"] == "dependencies":
                displayed_lists = [displayed_lists[-1]]
                displayed_lists.append(
                    ("dependency tests", len(data["dependencies"]), max_findings)
                )
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
    if not any(total for _, total, _ in displayed_lists):
        svg.text(
            24, y + 14, "No records saved · review search coverage and analysis settings", size=13
        )
        y += 30
    for label, total, shown in displayed_lists:
        if total > shown:
            svg.text(
                24,
                y + 14,
                f"{label.title()}: {shown}/{total} shown · display limit reached"
                if detail == "full"
                else "More evidence available · display limit reached",
                size=12,
            )
            y += 30
    return svg.finish(864, y + 20)


def render_html(
    result: Result | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Literal["full", "topology"] = "full",
    max_findings: int = 100,
) -> str:
    """Render a standalone interactive HTML evidence document.

    Parameters
    ----------
    result : Result or mapping
        Saved analytical result or its dictionary export. Rendering does not need
        the source dataframe and does not recompute analyses.
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
        Discovery lists support text search, pattern/exception filters for findings,
        visible counts, and expand/collapse controls. Search covers only records
        included by max_findings. Evidence links reveal and focus their target.
        Foundation figures support fit-width and fixed scaling with local scrolling.
        The grain HTML matrix lists features as rows and candidate keys as columns,
        with sticky headers, feature search, and a candidate-key filter. Its native
        text follows browser scaling; figure scaling applies to SVG views.
        Controls require JavaScript; native evidence disclosures remain readable
        without it. Reports do not retrieve source rows or rerun analysis.

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
    full = detail == "full"
    title = "Fieldwork / " + projected["kind"].replace("_", " ").title()
    parts = [f'<header><p class="eyebrow">Saved evidence report</p><h1>{_esc(title)}</h1>']
    parts.append(
        "<p>Explore the saved analysis below. Controls filter this report; "
        "they do not rerun the analysis or retrieve source data.</p></header>"
    )
    if not full:
        parts.append(
            '<p class="notice">Topology only. Quantitative evidence and source positions '
            "were removed before export. Structural labels remain.</p>"
        )
    if projected["kind"] == "comparison":
        for label in _comparison_labels(projected):
            parts.append("<p>" + _esc(label) + "</p>")
    else:
        if full and "scope" in projected:
            scope = projected["scope"]
            parts.append(
                "<p><strong>Population:</strong> "
                + _esc(scope.get("name", "input"))
                + " · "
                + _esc(scope.get("evaluated_rows", "unavailable"))
                + " evaluated source rows</p>"
            )
        if "analysis_unit" in projected:
            parts.append(
                "<p><strong>Analysis:</strong> "
                + _esc(_unit_label(projected["analysis_unit"]))
                + "</p>"
            )
    if projected.get("section_selection", {}).get("omitted"):
        parts.append(
            '<p class="notice">Not requested: '
            + _esc(", ".join(projected["section_selection"]["omitted"]))
            + "</p>"
        )
    if projected.get("skipped_features"):
        parts.append(
            '<p class="notice">' + _esc(_skipped_label(projected["skipped_features"])) + "</p>"
        )
    if full:
        coverage = _coverage_lines(projected)
        parts.append(
            '<details><summary>Search coverage and limits</summary><div class="content">'
            "<p>Search limits restrict what was tested or retained. Untested work is unknown. "
            "Display limits below only restrict this report.</p>"
        )
        parts.extend("<p>" + _esc(line) + "</p>" for line in coverage)
        saved_coverage = projected.get("section_coverage", projected.get("coverage"))
        if saved_coverage is not None:
            parts.append(_html_evidence(saved_coverage))
        else:
            parts.append("<p>Search coverage is unavailable in this saved result.</p>")
        parts.append("</div></details>")
        if any("(limited)" in line or "budget" in line for line in coverage):
            parts.append(
                '<p class="notice">Some search or retention limits were reached. '
                "Open Search coverage and limits before interpreting absent findings.</p>"
            )
    parts.append(
        '<details><summary>Visual summary</summary><div class="figure">'
        + render_svg(data, detail=detail, max_findings=min(12, max_findings))
        + '</div><p class="content">The visual summary has its own display limit of '
        "up to 12 items per list. Browse the included evidence below.</p></details>"
    )

    if "feature_network" in projected:
        cards = []
        included_ids = {f["id"] for f in projected["findings"][:max_findings]} if full else set()
        for node in projected["feature_network"]["nodes"]:
            card = [
                "<details data-record><summary>"
                + _esc(node["column"])
                + '</summary><div class="content"><ul>'
            ]
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
                card.append("<li>" + _esc(label))
                if full:
                    finding_id = edge["evidence"]["overview_finding_id"]
                    if finding_id in included_ids:
                        card.append(
                            ' · <a href="#'
                            + _esc(quote(finding_id, safe=""))
                            + '">inspect evidence</a>'
                        )
                    else:
                        card.append(" · evidence outside display limit")
                if edge.get("structure"):
                    card.append(_html_evidence(edge["structure"]))
                card.append("</li>")
            cards.append("".join(card) + "</ul></div></details>")
        parts.append(
            '<details><summary>Browse feature connections</summary><div class="content">'
            "<p>Connections are leads to inspect, not proof of equivalence or causation.</p>"
            + collection("Feature connections", cards, len(cards), full=full)
            + "</div></details>"
        )
    if full:
        candidates = projected.get("candidates", projected.get("overview", {}).get("grains", []))
        if candidates:
            cards = [
                "<details data-record><summary>"
                + _esc(_grain_title(c))
                + ' <span class="badge">'
                + _esc(c["role"])
                + "</span></summary>"
                + '<div class="content"><p>'
                + _esc(_candidate_explanation(c))
                + "</p>"
                + _html_evidence(c)
                + "</div></details>"
                for c in candidates[:max_findings]
            ]
            parts.append(collection("Candidate grains", cards, len(candidates), full=True))
        if "dependencies" in projected:
            cards = [
                "<details data-record><summary>"
                + _esc(_dependency_label(row))
                + '</summary><div class="content"><p>'
                + _esc(row["explanation"])
                + "</p>"
                + _html_evidence({k: v for k, v in row.items() if k != "explanation"})
                + "</div></details>"
                for row in projected["dependencies"][:max_findings]
            ]
            parts.append(
                collection(
                    "Completed dependency tests (including below finding threshold)",
                    cards,
                    len(projected["dependencies"]),
                    full=True,
                )
            )
    cards = []
    for row in projected["findings"][:max_findings]:
        anchor = f' id="{_esc(row["id"])}"' if full else ""
        has_exceptions = full and bool(row["exceptions"].get("total", 0))
        card = [
            f'<details data-record data-pattern="{_esc(row["pattern"])}"'
            + (f' data-exceptions="{str(has_exceptions).lower()}"' if full else "")
            + anchor
            + '><summary><span class="badge">'
            + _esc(row["pattern"].replace("_", " "))
            + "</span>"
            + _esc(row["statement"])
            + (
                ' <span class="badge">' + _esc(row["lead"]["reason"]) + "</span>"
                if "lead" in row
                else ""
            )
            + '</summary><div class="content">'
        ]
        card.append("<p>Analysis: " + _esc(_unit_label(row["analysis_unit"])) + "</p>")
        if full:
            card.append(
                "<p>Finding "
                + _esc(row["id"])
                + " · counting unit: "
                + _esc(row["counting_unit"])
                + "</p>"
            )
            if "explanation" in row:
                card.append("<p>" + _esc(row["explanation"]) + "</p>")
            card.append(_html_evidence(row["measurements"]))
            if projected["kind"] == "comparison":
                card.append(
                    "<p>Change is after minus before. A fraction change of 0.25 means "
                    "25 percentage points. Compare the two populations and denominators "
                    "before interpreting a change as improvement. Inspect the original "
                    "before/after results for source rows.</p>"
                )
            else:
                card.append(
                    "<h3>Representative source rows</h3>"
                    "<p>Positions are zero-based offsets in the original ordered source, "
                    "not dataframe index labels. Saved samples are the first matches in source "
                    "order; their size is not the total support.</p>"
                )
                for key in ("examples", "exceptions"):
                    card.append("<p>" + _esc(_sample_label(key.title(), row[key])) + "</p>")
                card.append(
                    "<p>In Python, with this result named <code>result</code> and its identical "
                    "ordered source named <code>df</code>:</p><p><code>"
                    + _esc(f"result.inspect(df, {row['id']!r})")
                    + "</code></p><p>Use <code>all_matches=True</code> to retrieve all "
                    "matching rows and <code>exceptions=True</code> for exception rows. "
                    "These operations require the original source and a Python session.</p>"
                )
        if row.get("structure"):
            card.append(_html_evidence(row["structure"]))
        cards.append("".join(card) + "</div></details>")
    parts.append(
        collection(
            "Inspect findings",
            cards,
            len(projected["findings"]),
            full=full,
            patterns=sorted({r["pattern"] for r in projected["findings"][:max_findings]}),
            exceptions=full and projected["kind"] != "comparison",
        )
    )
    return document(title, "".join(parts))


def _html_evidence(value):
    """Readable saved evidence without making readers interpret serialized dictionaries."""
    if isinstance(value, dict):
        return (
            '<div class="table-scroll"><table class="evidence-table">'
            + "".join(
                '<tr><th scope="row">'
                + _esc(str(key).replace("_", " "))
                + "</th><td>"
                + _html_evidence(item)
                + "</td></tr>"
                for key, item in value.items()
            )
            + "</table></div>"
        )
    if isinstance(value, list):
        if any(isinstance(item, (dict, list)) for item in value):
            return (
                "<ul>"
                + "".join("<li>" + _html_evidence(item) + "</li>" for item in value)
                + "</ul>"
            )
        return _esc(", ".join(str(item) for item in value) or "none")
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _esc("unavailable / not defined" if value is None else str(value))
