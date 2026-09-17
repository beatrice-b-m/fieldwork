"""Bounded plaintext rendering from explorer results only."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any, Literal

from .encoding import ScalarIdentity, display_scalar, validate_limit
from .result import ExplorerResult

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def _identity(record: Mapping[str, Any]) -> ScalarIdentity:
    kind = record["type"]
    if kind == "tuple":
        value = tuple(_identity(item) for item in record["value"])
    else:
        value = record.get("value")
    metadata = tuple(
        sorted((key, str(value)) for key, value in record.items() if key not in {"type", "value"})
    )
    return ScalarIdentity(kind, value, metadata)


def _safe(text: str, unicode_mode: str) -> str:
    text = _CONTROL.sub(lambda match: f"\\u{ord(match.group()):04x}", text)
    if unicode_mode == "safe":
        return text.encode("ascii", "backslashreplace").decode("ascii")
    if unicode_mode != "display":
        raise ValueError("unicode_mode must be 'safe' or 'display'")
    try:
        import wcwidth  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Native Unicode rendering requires the 'unicode' extra: "
            "pip install 'fieldwork[unicode]'"
        ) from exc
    return text


def _clip(text: str, width: int, unicode_mode: str) -> str:
    if unicode_mode == "display":
        from wcwidth import wcswidth, wcwidth

        if wcswidth(text) <= width:
            return text
        marker = "." * min(3, width)
        target = max(0, width - len(marker))
        output = []
        used = 0
        for character in text:
            cells = max(0, wcwidth(character))
            if used + cells > target:
                break
            output.append(character)
            used += cells
        return "".join(output) + marker
    if len(text) <= width:
        return text
    if width <= 3:
        return "." * width
    return text[: width - 3] + "..."


def _quantity(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _scope_text(scope: Mapping[str, Any]) -> str:
    return (
        f"rows: {scope['evaluated_rows']} evaluated / {scope['input_rows']} input; "
        f"excluded: {scope['missing_excluded_rows']} missing, "
        f"{scope['restriction_excluded_rows']} restricted"
    )


def _section_lines(
    kind: str,
    data: Mapping[str, Any],
    *,
    max_nodes: int,
    missing_label: str,
    detail: Literal["full", "topology"],
) -> Iterator[str]:
    show_quantities = detail == "full"

    def label(record: Mapping[str, Any], *, column: bool = False) -> str:
        if column and record["type"] == "string" and record["value"].isidentifier():
            return record["value"]
        return display_scalar(_identity(record), missing_label)

    def omission(node: Mapping[str, Any], indent: str) -> Iterator[str]:
        if node.get("omitted_child_rows"):
            reasons = ", ".join(node.get("stop_reasons", [])) or "output limits"
            if show_quantities:
                yield (
                    f"{indent}... {_quantity(node['omitted_child_rows'], 'row')} in "
                    f"{_quantity(node['omitted_child_levels'], 'child level')} omitted ({reasons})"
                )
            else:
                yield f"{indent}... child branches omitted ({reasons})"

    scopes = {scope["scope_id"]: scope for scope in data.get("scopes", [])}
    metadata = data.get("scope_metadata") or {}
    conditional = metadata.get("conditional") or any(s.get("conditional") for s in scopes.values())
    titles = {
        "levels": "Levels",
        "census": "Census",
        "grain": "Grain",
        "pairs": "Pairs",
        "schema_proposal": "Schema proposals (suggestions)",
        "joint_counts": "Joint counts",
    }
    if kind not in titles:
        yield f"Unsupported result kind: {kind!r}"
        return
    state = data.get("status", "unknown")
    yield f"{titles[kind]} ({state}{', conditional' if conditional else ''})"
    if state == "not_requested":
        return
    if metadata.get("scope"):
        yield f"  cohort from {metadata['source_scope']}"
        if show_quantities:
            yield f"    {_scope_text(metadata['scope'])}"
    warning_columns = {
        f["feature_id"]: f["column"]
        for f in (data.get("per_feature", []) if kind == "levels" else data.get("features", []))
        if isinstance(f, Mapping) and "feature_id" in f
    }
    for warning in data.get("warnings", []):
        if not show_quantities and warning["code"] == "LOW_RETAINED_FRACTION":
            continue
        column = warning.get("column", warning_columns.get(warning.get("feature_id")))
        named = f" (column={label(column, column=True)})" if column is not None else ""
        warning_detail = ", ".join(
            f"{key}={value!r}"
            for key, value in warning.items()
            if key not in {"code", "column", "column_label"}
            and not (key == "feature_id" and column is not None)
        )
        yield (
            f"  Warning: {warning['code']}"
            + named
            + (f" ({warning_detail})" if warning_detail and show_quantities else "")
        )

    if kind == "levels":
        for feature in data.get("per_feature", []):
            scope = scopes[feature["scope_id"]]
            column = label(feature["column"], column=True)
            if show_quantities:
                yield (
                    f"  {column}: {feature['levels_reported']}/{feature['levels_total']} levels; "
                    f"{feature['reported_rows']}/{scope['evaluated_rows']} evaluated rows reported"
                )
            else:
                yield f"  {column}"
            if show_quantities and (
                scope["missing_excluded_rows"] or scope["restriction_excluded_rows"]
            ):
                yield f"    {_scope_text(scope)}"
            levels = feature.get("levels", [])
            if not show_quantities:
                levels = sorted(levels, key=lambda level: _identity(level["value"]).sort_key())
            for level in levels:
                rendered = label(level["value"])
                yield (
                    f"    {rendered}: {_quantity(level['count'], 'row')}"
                    if show_quantities
                    else f"    {rendered}"
                )
            if feature["omitted_levels"]:
                if show_quantities:
                    yield (
                        f"    ... {_quantity(feature['omitted_levels'], 'level')} / "
                        f"{_quantity(feature['unreported_rows'], 'row')} not reported"
                    )
                else:
                    yield "    ... additional levels not reported (analysis limits)"
    elif kind == "census":
        tree = data.get("tree", {})
        scope = scopes.get(tree.get("scope_id"))
        if scope and show_quantities:
            yield f"  {_scope_text(scope)}"
        features = {f["feature_id"]: label(f["column"], column=True) for f in data["features"]}
        yield "  path: " + " > ".join(features[f] for f in tree["dimensions"])
        root = tree["root"]
        if show_quantities:
            yield f"  total: {_quantity(root['count'], 'row')}"
        yield from omission(root, "    ")
        nodes = tree.get("nodes", [])
        # Keep the producer's ancestor-closed budget allocation, but print each
        # subtree contiguously. Never infer parentage from depth or storage order.
        children: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        retained = nodes[:max_nodes]
        needed_levels = {node["level_id"] for node in retained}
        values = {
            entry["level_id"]: entry["value"]
            for entry in data["level_dictionary"]
            if entry["level_id"] in needed_levels
        }
        for node in retained:
            children[node["parent_id"]].append(node)
        if not show_quantities:
            for siblings in children.values():
                siblings.sort(key=lambda node: _identity(values[node["level_id"]]).sort_key())
        stack = [iter(children[root["node_id"]])]
        while stack:
            node = next(stack[-1], None)
            if node is None:
                stack.pop()
                continue
            indent = "  " * len(stack)
            rendered = f"{indent}{features[node['feature_id']]}={label(values[node['level_id']])}"
            if show_quantities:
                rendered += f": {_quantity(node['count'], 'row')}"
            yield rendered
            yield from omission(node, indent + "  ")
            if children.get(node["node_id"]):
                stack.append(iter(children[node["node_id"]]))
        if len(nodes) > len(retained):
            if show_quantities:
                yield f"  ... {len(nodes) - len(retained)} nodes not rendered (renderer max_nodes)"
            else:
                yield "  ... additional nodes not rendered (renderer max_nodes)"
    elif kind == "grain":
        yield "  Observed dependencies among tested keys."
        for dependency in data.get("dependencies", []):
            columns = ", ".join(label(c, column=True) for c in dependency["key_columns"])
            yield (
                f"  {dependency['key_name']} [{columns}] -> "
                f"{label(dependency['target'], column=True)}"
            )
            state = dependency["holds"]
            if state is None:
                reason = dependency["undefined_reason"].replace("_", " ")
                yield f"    observed dependency undefined ({reason})"
            elif not show_quantities:
                yield f"    observed dependency {'holds' if state else 'fails'}"
            else:
                yield (
                    f"    observed dependency {'holds' if state else 'fails'}; "
                    f"{dependency['violating_groups']}/{dependency['evaluated_groups']} "
                    f"violating groups; {dependency['affected_rows']} affected rows"
                )
            if show_quantities:
                yield (
                    f"    support: {dependency['evaluated_rows']} rows; "
                    f"{dependency['singleton_groups']} singleton, "
                    f"{dependency['repeated_groups']} repeated groups"
                )
            scope = scopes.get(dependency["scope_id"])
            if (
                show_quantities
                and scope
                and (scope["missing_excluded_rows"] or scope["restriction_excluded_rows"])
            ):
                yield f"    {_scope_text(scope)}"
        for target in data.get("targets", []):
            name = label(target["target"], column=True)
            if target["cross_key_comparison"] == "not_comparable":
                yield f"  {name}: determinant comparison unavailable (different row populations)"
            elif target["determining_keys"]:
                yield f"  {name}: coarsest observed candidates: {', '.join(target['coarsest_candidates'])}"
                for group in target["equivalent_determinants"]:
                    yield f"    equivalent: {', '.join(group)}"
                for group in target["incomparable_candidates"]:
                    yield f"    incomparable: {', '.join(group)}"
            else:
                yield f"  {name}: grain undetermined by supplied keys"
    elif kind == "pairs":
        omitted_contexts = data.get(
            "omitted_contexts",
            data.get("requested_contexts", 0) - data.get("processed_contexts", 0),
        )
        if data.get("omitted_pairs") or omitted_contexts:
            if show_quantities:
                yield (
                    f"  omitted: {data.get('omitted_pairs', 0)} requested pairs, "
                    f"{omitted_contexts} requested contexts (output limits)"
                )
            else:
                yield "  ... requested pairs or contexts omitted (output limits)"
        yield "  A / B is left / right; relations describe observed pairs only."
        yield "  Pair support does not rule out higher-order constraints."
        for pair in data.get("pairs", []):
            names = [label(c, column=True) for c in pair["columns"]]
            context = (
                ", ".join(
                    f"{label(c['column'], column=True)}={label(c['value'])}"
                    for c in pair["context"]
                )
                or "global"
            )
            yield f"  {names[0]} / {names[1]} [{context}]"
            if show_quantities:
                yield f"    {_scope_text(pair['scope'])}"
            relation = pair["relation"]
            meanings = {
                "1:1": "one-to-one",
                "1:n": "one A to many B",
                "n:1": "many A to one B",
                "n:m": "many-to-many",
            }
            if relation is None:
                yield f"    relation undefined ({pair['relation_reason'].replace('_', ' ')})"
            else:
                yield f"    observed relation: {relation} ({meanings[relation]})"
            if show_quantities:
                association = pair["cramers_v"]
                v_text = (
                    f"{association:.6g}"
                    if association is not None
                    else f"undefined ({pair['cramers_v_reason'].replace('_', ' ')})"
                )
                yield f"    Cramer's V: {v_text}"
            absence = pair.get("absence")
            if absence:
                domains = pair["domains"]
                if show_quantities:
                    yield (
                        f"    unobserved: {absence['absent_cells']}/{absence['total_cells']} "
                        "domain cells (not evidence of impossibility)"
                    )
                else:
                    yield "    unobserved combinations (not evidence of impossibility)"
                for side, name in zip(("a", "b"), names):
                    if show_quantities:
                        yield (
                            f"    domain {name}: {domains[side + '_size']} levels "
                            f"({domains[side + '_source'].replace('_', ' ')})"
                        )
                    else:
                        yield f"    domain {name}: {domains[side + '_source'].replace('_', ' ')}"
                if show_quantities:
                    classes = absence["classes"]
                    yield f"    zero support in cohort: {classes['unobserved_zero_support']} cells"
                    yield f"    level absent in context: {classes['level_absent_under_parent']} cells"
                    yield (
                        "    within supported margins: "
                        f"{classes['unobserved_within_supported_margins']} cells"
                    )
                for example in absence["examples"]:
                    yield f"      unobserved example: {label(example['a'])} / {label(example['b'])}"
                if absence["examples_omitted"]:
                    if show_quantities:
                        yield f"    ... {absence['examples_omitted']} unobserved examples not reported"
                    else:
                        yield "    ... additional unobserved examples not reported"
    elif kind == "joint_counts":
        if show_quantities:
            for scope in scopes.values():
                yield f"  {_scope_text(scope)}"
        names = [label(c, column=True) for c in data["columns"]]
        context = (
            ", ".join(
                f"{label(p['column'], column=True)}={label(p['value'])}"
                for p in data.get("context", [])
            )
            or "global"
        )
        yield f"  {names[0]} / {names[1]} [{context}]"
        yield "  Observed cells and unobserved combinations."
        for cell in data["cells"]:
            text = (
                f"    {names[0]}={label(data['a'][cell['a']])}, "
                f"{names[1]}={label(data['b'][cell['b']])}"
            )
            yield text + (f": {_quantity(cell['count'], 'row')}" if show_quantities else "")
    elif kind == "schema_proposal":
        for proposal in data.get("proposals", []):
            yield f"  {label(proposal['column'], column=True)}: suggested {proposal['proposed_role']}"
            for reason in proposal["reasons"]:
                if show_quantities or reason["code"] in {"DTYPE", "NAME_HINT_ID"}:
                    yield f"    {reason['code'].lower().replace('_', ' ')}: {reason['value']}"
            evidence = proposal["fd_evidence"]
            if isinstance(evidence, Mapping):
                yield f"    dependency evidence: {evidence['status']}"
                for key in evidence["keys"]:
                    suffix = f", {key['evaluated_groups']} groups" if show_quantities else ""
                    yield f"      {key['name']}: holds={key['holds']}{suffix}"
            else:
                yield f"    dependency evidence: {evidence.replace('_', ' ')}"


def render_plaintext(
    result: ExplorerResult | Mapping[str, Any],
    *,
    width: int = 100,
    max_lines: int = 200,
    max_nodes: int = 1000,
    missing_label: str = "<NA>",
    unicode_mode: str = "safe",
    detail: Literal["full", "topology"] = "full",
) -> str:
    """Render bounded, terminal-safe evidence, with explicit populations and omissions.

    Strings are quoted. Census nodes retain the producer's budget order but are
    displayed parent-first with contiguous subtrees. ``detail="topology"`` keeps
    labels and qualitative relationships while suppressing quantities and
    distribution statistics. It also canonicalizes frequency-ranked siblings.
    A final truncation marker replaces the last line only when further content
    actually exists.
    """

    for name, value, zero in (
        ("width", width, False),
        ("max_lines", max_lines, False),
        ("max_nodes", max_nodes, True),
    ):
        if value is None:
            raise ValueError(f"{name} must be an integer")
        validate_limit(name, value, zero=zero)
    _safe("", unicode_mode)  # Validate even when the line budget is one.
    if detail not in {"full", "topology"}:
        raise ValueError("detail must be 'full' or 'topology'")
    if isinstance(result, ExplorerResult):
        data, kind, version = result.payload, result.kind, result.schema_version
    else:
        data, kind, version = result, result.get("kind", "?"), result.get("schema_version", "?")

    def lines() -> Iterator[str]:
        yield f"Fieldwork feature explorer v{version}"
        if detail == "topology":
            yield "Topology display (quantitative evidence suppressed)"
        sections = data.get("sections")
        if sections is not None:
            for section_kind, section in sections.items():
                yield from _section_lines(
                    section_kind,
                    section,
                    max_nodes=max_nodes,
                    missing_label=missing_label,
                    detail=detail,
                )
        else:
            yield from _section_lines(
                kind, data, max_nodes=max_nodes, missing_label=missing_label, detail=detail
            )

    output: list[str] = []
    for line in lines():
        if len(output) == max_lines:
            output[-1] = _clip("... more output not rendered (max_lines)", width, unicode_mode)
            break
        output.append(_clip(_safe(line, unicode_mode), width, unicode_mode))
    return "\n".join(output)
