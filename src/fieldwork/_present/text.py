"""Plaintext rendering of projections: one body function per result kind."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from .common import quantity
from .project import candidate_explanation, dependency_label, grain_title, signature_label

Lines = Iterator[str]

_TITLES = {
    "levels": "Levels",
    "census": "Census",
    "grain": "Grain",
    "pairs": "Pairs",
    "joint_counts": "Joint counts",
    "schema_proposal": "Schema proposals (suggestions)",
}
_MEANINGS = {
    "1:1": "one-to-one",
    "1:n": "one A to many B",
    "n:1": "many A to one B",
    "n:m": "many-to-many",
}


def lines(projection: Mapping[str, Any], *, max_nodes: int) -> Lines:
    """Every line of a projection's text, before width and line limits."""
    full = projection["detail"] == "full"
    yield f"Fieldwork · {projection['kind']}"
    if not full:
        yield "Topology only · quantitative evidence suppressed"
    body = _BODIES.get(projection["kind"], _findings)
    yield from body(projection, full, max_nodes)


def _section(projection: Mapping[str, Any], full: bool, max_nodes: int) -> Lines:
    """A foundation kind: a status line, warnings, then its evidence."""
    kind = projection["kind"]
    status = projection.get("status", "computed")
    conditional = ", conditional" if projection.get("conditional") else ""
    yield f"{_TITLES[kind]} ({status}{conditional})"
    if status == "not_requested":
        return
    for warning in projection.get("warnings", []):
        named = f" (column={warning['column']})" if "column" in warning else ""
        detail = f" ({warning['detail']})" if warning.get("detail") else ""
        yield f"  Warning: {warning['code']}{named}{detail}"
    yield from _FOUNDATION[kind](projection, full, max_nodes)


def _levels(projection, full, max_nodes) -> Lines:
    for feature in projection["features"]:
        if full:
            s = feature["summary"]
            yield (
                f"  {feature['label']}: {s['levels_reported']}/{s['levels_total']} levels; "
                f"{s['reported_rows']}/{s['evaluated_rows']} evaluated rows reported"
            )
            if feature["excluded"]:
                yield f"    {feature['scope']}"
        else:
            yield f"  {feature['label']}"
        for row in feature["rows"]:
            if not row.get("omitted"):
                yield f"    {row['label']}" + (f": {quantity(row['count'], 'row')}" if full else "")
            elif full:
                yield (
                    f"    ... {quantity(row['levels'], 'level')} / "
                    f"{quantity(row['count'], 'row')} not reported"
                )
            else:
                yield "    ... additional levels not reported (analysis limits)"


def _census(projection, full, max_nodes) -> Lines:
    if full:
        yield f"  {projection['scope']}"
    yield "  path: " + " > ".join(projection["dimensions"])
    rows = projection["rows"]
    if full:
        yield f"  total: {quantity(rows[0]['count'], 'row')}"
    # The budget keeps nodes breadth first, like the census's own node budget, so
    # every top-level group shows before any subdivision; each kept node's
    # ancestors are shallower and therefore kept too. Rows print in tree order.
    nodes = [row for row in rows[1:] if not row.get("omitted")]
    breadth_first = sorted(range(len(nodes)), key=lambda i: (nodes[i]["depth"], i))
    shown = {nodes[i]["id"] for i in breadth_first[:max_nodes]} | {rows[0]["id"]}
    for row in rows[1:]:
        indent = "  " * (row["depth"] + 1)
        if row.get("omitted"):
            if row["parent"] not in shown:
                continue
            reasons = ", ".join(row["reasons"]) or "output limits"
            yield (
                f"{indent}... {quantity(row['count'], 'row')} in "
                f"{quantity(row['levels'], 'child level')} omitted ({reasons})"
                if full
                else f"{indent}... child branches omitted ({reasons})"
            )
        elif row["id"] in shown:
            yield f"{indent}{row['label']}" + (f": {quantity(row['count'], 'row')}" if full else "")
    if len(nodes) > max_nodes:
        hidden = len(nodes) - max_nodes
        yield (
            f"  ... {hidden} nodes not rendered (renderer max_nodes)"
            if full
            else "  ... additional nodes not rendered (renderer max_nodes)"
        )


def _grain(projection, full, max_nodes) -> Lines:
    yield "  Observed dependencies among tested keys."
    for test in projection["dependencies"]:
        yield f"  {test['key']} [{', '.join(test['columns'])}] -> {test['target']}"
        if test["state"] == "undefined":
            yield f"    observed dependency undefined ({test['reason']})"
        elif not full:
            yield f"    observed dependency {test['state']}"
        else:
            yield (
                f"    observed dependency {test['state']}; "
                f"{test['violating_groups']}/{test['evaluated_groups']} violating groups; "
                f"{test['affected_rows']} affected rows"
            )
        if full:
            yield (
                f"    support: {test['evaluated_rows']} rows; {test['singleton_groups']} "
                f"singleton, {test['repeated_groups']} repeated groups"
            )
            if test["excluded"]:
                yield f"    {test['scope']}"
    for target in projection["targets"]:
        if not target["comparable"]:
            yield (
                f"  {target['label']}: determinant comparison unavailable "
                "(different row populations)"
            )
        elif target["determined"]:
            yield f"  {target['label']}: coarsest observed candidates: {', '.join(target['coarsest'])}"
            for group in target["equivalent"]:
                yield f"    equivalent: {', '.join(group)}"
            for group in target["incomparable"]:
                yield f"    incomparable: {', '.join(group)}"
        else:
            yield f"  {target['label']}: grain undetermined by supplied keys"


def _pairs(projection, full, max_nodes) -> Lines:
    if projection["omitted"]:
        counts = projection.get("omitted_counts")
        yield (
            f"  omitted: {counts['pairs']} requested pairs, {counts['contexts']} requested "
            "contexts (output limits)"
            if full
            else "  ... requested pairs or contexts omitted (output limits)"
        )
    yield "  A / B is left / right; relations describe observed pairs only."
    yield "  Pair support does not rule out higher-order constraints."
    for record in projection["records"]:
        a, b = record["columns"]
        yield f"  {a} / {b} [{record['context'] if record['context'] != 'Global' else 'global'}]"
        if full:
            yield f"    {record['scope']}"
        relation = record["relation"]
        yield (
            f"    observed relation: {relation} ({_MEANINGS[relation]})"
            if relation
            else f"    relation undefined ({record['relation_reason']})"
        )
        if full:
            value = record["association"]
            text = (
                f"{value:.6g}"
                if value is not None
                else f"undefined ({record['association_reason']})"
            )
            yield f"    Cramer's V: {text}"
        if "absence" in record:
            yield from _absence(record["absence"], (a, b), full)


def _absence(absence, names, full) -> Lines:
    if full:
        yield (
            f"    unobserved: {absence['absent_cells']}/{absence['total_cells']} domain cells "
            "(not evidence of impossibility)"
        )
    else:
        yield "    unobserved combinations (not evidence of impossibility)"
    for index, name in enumerate(names):
        source = absence["sources"][index]
        yield (
            f"    domain {name}: {absence['sizes'][index]} levels ({source})"
            if full
            else f"    domain {name}: {source}"
        )
    if full:
        classes = absence["classes"]
        yield f"    zero support in cohort: {classes['unobserved_zero_support']} cells"
        yield f"    level absent in context: {classes['level_absent_under_parent']} cells"
        yield (
            f"    within supported margins: {classes['unobserved_within_supported_margins']} cells"
        )
    for a, b in absence["examples"]:
        yield f"      unobserved example: {a} / {b}"
    if absence["examples_omitted"]:
        yield (
            f"    ... {absence['examples_omitted_count']} unobserved examples not reported"
            if full
            else "    ... additional unobserved examples not reported"
        )


def _joint_counts(projection, full, max_nodes) -> Lines:
    if full:
        yield f"  {projection['scope']}"
    a, b = projection["columns"]
    yield f"  {a} / {b} [{projection['context']}]"
    yield "  Observed cells and unobserved combinations."
    for cell in projection["cells"]:
        text = f"    {a}={cell['a_label']}, {b}={cell['b_label']}"
        yield text + (f": {quantity(cell['count'], 'row')}" if full else "")


def _schema_proposal(projection, full, max_nodes) -> Lines:
    for proposal in projection["proposals"]:
        yield f"  {proposal['label']}: suggested {proposal['role']}"
        for reason in proposal["reasons"]:
            yield f"    {reason['code'].lower().replace('_', ' ')}: {reason['value']}"
        yield f"    dependency evidence: {proposal['fd_evidence']}"
        for key in proposal.get("keys", []):
            suffix = f", {key['groups']} groups" if full else ""
            yield f"      {key['name']}: holds={key['holds']}{suffix}"


def _profile(projection, full, max_nodes) -> Lines:
    for section in projection["sections"].values():
        yield from _section(section, full, max_nodes)


_FOUNDATION: dict[str, Callable[..., Lines]] = {
    "levels": _levels,
    "census": _census,
    "grain": _grain,
    "pairs": _pairs,
    "joint_counts": _joint_counts,
    "schema_proposal": _schema_proposal,
}
_BODIES: dict[str, Callable[..., Lines]] = {
    **dict.fromkeys(_FOUNDATION, _section),
    "profile": _profile,
}


# Findings-based kinds.


def unit_label(unit: Mapping[str, Any]) -> str:
    text = unit["counting_unit"]
    if "denominator" in unit:
        text = f"{unit['denominator']} {text}"
    if unit.get("entity_keys"):
        text += "; keys=" + ", ".join(unit["entity_keys"])
    if "presence_aggregation" in unit:
        text += "; presence=" + unit["presence_aggregation"]
    return text


def comparison_labels(projection: Mapping[str, Any]) -> list[str]:
    labels = []
    for side in ("before", "after"):
        scope = projection.get(f"{side}_scope")
        if scope is None:
            continue
        text = f"{side.title()}: {scope['name']}"
        if "evaluated_rows" in scope:
            text += f"; {scope['evaluated_rows']} evaluated rows"
        labels.append(text)
        unit = projection.get(f"{side}_analysis_unit")
        if unit:
            labels.append(f"  {side.title()} analysis: " + unit_label(unit))
    return labels


_COVERAGE = (
    ("candidates_evaluated", "candidate_space", "candidate keys evaluated"),
    ("dependency_tests", "dependency_tests_possible", "dependency tests completed"),
    ("pairs_evaluated", "pair_candidates", "pairs evaluated"),
    ("features_evaluated", "features_requested", "features evaluated"),
    ("contexts_evaluated", "contexts_total", "contexts evaluated"),
    ("contexts_shown", "contexts_total", "contexts retained"),
    ("signatures_shown", "signatures_total", "signatures retained"),
)


def coverage_lines(projection: Mapping[str, Any]) -> list[str]:
    """Saved work bounds, without implying exhaustive discovery."""
    coverages = projection.get(
        "section_coverage", {projection["kind"]: projection.get("coverage", {})}
    )
    output = []
    for section, coverage in coverages.items():
        for done, possible, text in _COVERAGE:
            if done in coverage and possible in coverage:
                n, total = coverage[done], coverage[possible]
                if total or n:
                    output.append(
                        f"{section}: {n}/{total} {text}{' (limited)' if n < total else ''}"
                    )
        if coverage.get("search_exhausted_budget"):
            output.append(f"{section}: path search reached its candidate budget")
    return output


def skipped_label(skipped: list[Mapping[str, Any]]) -> str:
    return "Skipped columns with unsupported values: " + ", ".join(
        f"{record['feature']} ({record['value_type']})" for record in skipped
    )


def sample_label(name: str, sample: Mapping[str, Any]) -> str:
    positions = sample["positions"]
    total = sample.get("total")
    count = f"{len(positions)}/{total}" if total is not None else str(len(positions))
    return f"{name}: {count} saved source positions {positions}"


def _findings(projection, full, max_nodes) -> Lines:
    kind = projection["kind"]
    if full and kind != "comparison":
        yield f"Population: {projection.get('scope', {}).get('evaluated_rows', 0)} rows"
    if projection.get("section_selection", {}).get("omitted"):
        yield "Not requested: " + ", ".join(projection["section_selection"]["omitted"])
    if projection.get("skipped_features"):
        yield skipped_label(projection["skipped_features"])
    yield from comparison_labels(projection)
    if full:
        yield from _coverage_summary(projection)
    if not projection["findings"]:
        yield "No findings saved; review coverage and analysis settings."
    if "analysis_unit" in projection and kind != "comparison":
        yield "Analysis: " + unit_label(projection["analysis_unit"])
    if kind == "overview":
        yield from _overview(projection, full, max_nodes)
    yield from _candidates_and_tests(projection, max_nodes)
    if kind != "overview":
        yield from _finding_cards(projection, full, max_nodes)


def _coverage_summary(projection) -> Lines:
    coverage = coverage_lines(projection)
    if projection["kind"] == "overview":
        limited = any("(limited)" in line or "budget" in line for line in coverage)
        yield (
            "Search limits reached; review section coverage."
            if limited
            else "Search coverage is saved per section; untested work is unknown."
        )
    elif coverage:
        yield "Search coverage (untested work is unknown)"
        yield from ("  " + line for line in coverage)


def _overview(projection, full, max_nodes) -> Lines:
    overview = projection["overview"]
    unit = projection.get("analysis_unit", {}).get("counting_unit", "rows")
    if full:
        leads = [
            row for row in projection["findings"] if row.get("lead", {}).get("score", 0) >= 0.3
        ]
        yield "Leads (inspect with result.inspect(df, id))"
        yield from (
            f"  [{row['id']}] {row['statement']} ({row['lead']['reason']})"
            for row in leads[: min(8, max_nodes)]
        )
        if not leads:
            yield "  none stood out; browse the sections below"
    yield "Candidate grains"
    for candidate in overview["grains"][: min(5, max_nodes)]:
        text = "  " + grain_title(candidate) + ": " + candidate["role"]
        if full:
            text += (
                f"; {candidate['groups']} groups, {len(candidate['determines'])} exact targets, "
                f"{len(candidate['determines_with_repeated_support'])} with repeated support"
            )
        yield text
    yield "Suggested census paths"
    yield from ("  " + " > ".join(path) for path in overview["paths"][:max_nodes])
    yield "Availability families"
    yield from ("  " + ", ".join(group) for group in overview["families"][: min(5, max_nodes)])
    if not overview["families"]:
        yield "  none"
    yield "Major availability signatures"
    for signature in overview["signatures"][: min(5, max_nodes)]:
        yield (
            "  " + signature_label(signature) + (f" ({signature['count']} {unit})" if full else "")
        )
    if "feature_network" in projection:
        yield "Connected feature evidence"
        for group in projection["feature_network"]["components"][: min(5, max_nodes)]:
            yield "  " + ", ".join(group)
    yield "Summary lists are limited; individual sections retain complete evidence and coverage."


def _candidates_and_tests(projection, max_nodes) -> Lines:
    if "candidates" in projection:
        yield "Candidate grains"
        for candidate in projection["candidates"][:max_nodes]:
            yield "  " + ", ".join(candidate["columns"]) + ": " + candidate["role"]
            yield from ("  " + part for part in candidate_explanation(candidate).split("; "))
        if len(projection["candidates"]) > max_nodes:
            yield "... more candidate grains not rendered (max_nodes)"
    if "dependencies" in projection:
        yield "Completed dependency tests (including below finding threshold)"
        for row in projection["dependencies"][:max_nodes]:
            yield dependency_label(row)
            yield from ("  " + part for part in row["explanation"].split("; "))
        if len(projection["dependencies"]) > max_nodes:
            yield "... more dependency tests not rendered (max_nodes)"


def _finding_cards(projection, full, max_nodes) -> Lines:
    for row in projection["findings"][:max_nodes]:
        yield (f"[{row['id']}] " if full else "") + row["statement"]
        yield "  Analysis: " + unit_label(row["analysis_unit"])
        if not full:
            continue
        if "explanation" in row:
            yield from ("  " + part for part in row["explanation"].split("; "))
        measurements = row["measurements"]
        if "explanation" in measurements:
            yield from ("  " + reason for reason in measurements["explanation"].split("; "))
        metrics = ", ".join(
            f"{k}={v}" for k, v in measurements.items() if not isinstance(v, (dict, list))
        )
        yield f"  Unit: {row['counting_unit']}; " + metrics
        for feature in measurements.get("availability", []):
            yield (
                f"  {feature['feature']}: {feature['populated']}/{feature['denominator']} "
                f"populated {row['counting_unit']}"
            )
        yield "  " + sample_label("Examples", row["examples"])
        yield "  " + sample_label("Exceptions", row["exceptions"])
    if len(projection["findings"]) > max_nodes:
        yield "... more findings not rendered (max_nodes)"
