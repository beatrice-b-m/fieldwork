"""Self-contained SVG figures of projections: one function per figure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

from .common import esc, wrap
from .project import candidate_explanation, dependency_label, grain_title
from .text import comparison_labels, skipped_label, unit_label

COLORS = {
    "constant": "#dcf3e7",
    "varying": "#ffe4d4",
    "undefined": "#eceef2",
    "untested": "#ffffff",
    "1:1": "#dcf3e7",
    "1:n": "#dcecff",
    "n:1": "#eee2ff",
    "n:m": "#ffe4d4",
}


class SVG:
    """An SVG document assembled from escaped text and rectangles."""

    def __init__(self, title: str, *, role: str = "img", marker: str = "arrow"):
        self.title, self.role, self.marker = title, role, marker
        self.parts: list[str] = []

    def text(self, x, y, value, *, size=14, bold=False, attrs=""):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" '
            f'font-weight="{600 if bold else 400}" {attrs}>{esc(value)}</text>'
        )

    def rect(self, x, y, width, height, *, fill="#ffffff", attrs=""):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="7" fill="{fill}" stroke="#cbd5df" {attrs}/>'
        )

    def finish(self, width, height) -> str:
        marker = (
            f'<defs><marker id="{self.marker}" markerWidth="8" markerHeight="8" '
            'refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" '
            'fill="#647b8c"/></marker></defs>'
            if any("marker-end=" in part for part in self.parts)
            else ""
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" role="{self.role}" '
            f'aria-label="{esc(self.title)}" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" font-family="Arial, sans-serif" fill="#193345">'
            f"<title>{esc(self.title)}</title>"
            + marker
            + '<rect width="100%" height="100%" fill="#f6f8fb"/>'
            + "".join(self.parts)
            + "</svg>"
        )


def _heading(svg: SVG, projection: Mapping[str, Any], title: str) -> int:
    svg.text(24, 32, title, size=22, bold=True)
    y = 55
    lines = [projection.get("caption", ""), projection.get("scope", "")]
    if projection["detail"] == "topology":
        lines.append("Topology only · quantitative evidence suppressed")
    if projection.get("missingness"):
        lines.append("Missingness: " + projection["missingness"])
    for line in lines:
        for wrapped in wrap(line, 105) if line else []:
            svg.text(24, y, wrapped, size=12)
            y += 18
    return y + 20


def _ranks(projection) -> dict[str, int]:
    parents = defaultdict(list)
    for edge in projection["edges"]:
        parents[edge["target"]].append(edge["source"])
    ranks: dict[str, int] = {}
    pending = [n["id"] for n in projection["nodes"]]
    while pending:
        ready = [n for n in pending if all(p in ranks for p in parents[n])]
        if not ready:
            raise ValueError("Grain graph must be acyclic")
        for node in ready:
            ranks[node] = max((ranks[p] + 1 for p in parents[node]), default=0)
            pending.remove(node)
    return ranks


def _node_lines(node, projection, features, exceptions, collapsed) -> list[tuple[str, Any, bool]]:
    lines: list[tuple[str, Any, bool]] = [(title, None, True) for title in node["titles"]]
    if len(node["titles"]) > 1:
        lines.append(("Observationally equivalent keys", None, False))
    support = node.get("support")
    if support:
        lines.append(
            (
                f"{support['evaluated_groups']} groups · {support['evaluated_rows']} rows",
                None,
                False,
            )
        )
        lines.append(
            (
                f"{support['singleton_groups']} singleton · {support['repeated_groups']} repeated groups",
                None,
                False,
            )
        )
    lines.append(("Attributes collapsed" if collapsed else "Attributes", None, True))
    for feature_id in [] if collapsed else node["attributes"]:
        feature = features[feature_id]
        shared = " (shared)" if len(feature["nodes"]) > 1 else ""
        lines.append((feature["label"] + shared, feature_id, False))
    if not node["attributes"]:
        lines.append(("No assigned attributes", None, False))
    for record in projection["evidence"] if exceptions else []:
        if (
            record["key"] == node["key_names"][0]
            and record["state"] == "varying"
            and record["compatible"]
        ):
            text = "Varies: " + features[record["feature"]]["label"]
            if projection["detail"] == "full":
                text += f" · {record['violating_groups']}/{record['evaluated_groups']} groups"
            lines.append((text, record["feature"], False))
    return [(part, feature, bold) for text, feature, bold in lines for part in wrap(text)]


def grain_map(
    projection: Mapping[str, Any],
    *,
    exceptions: bool = False,
    collapsed: bool = False,
    role: str = "img",
    marker: str = "arrow",
) -> str:
    """Keys as nodes from coarse to fine, with the columns each one determines.

    An HTML page embeds several variants, so each gets its own marker ID and an
    interactive role.
    """
    svg = SVG("Observed grain map", role=role, marker=marker)
    y = _heading(svg, projection, "Observed grain map")
    svg.text(24, y, "Arrows mean finer grouping ↓", size=13, bold=True)
    y += 30
    features = {f["id"]: f for f in projection["features"]}
    nodes = projection["nodes"]
    ranks = _ranks(projection)
    layers = defaultdict(list)
    for node in nodes:
        layers[ranks[node["id"]]].append(node)
    contents = {n["id"]: _node_lines(n, projection, features, exceptions, collapsed) for n in nodes}
    heights = {node_id: 30 + 20 * len(lines) for node_id, lines in contents.items()}
    width = max(900, max((len(layer) for layer in layers.values()), default=1) * 400 + 24)
    positions = {}
    for rank in sorted(layers):
        start = (width - len(layers[rank]) * 400 + 20) / 2
        for index, node in enumerate(layers[rank]):
            positions[node["id"]] = (start + index * 400, y)
        y += max(heights[n["id"]] for n in layers[rank]) + 70
    for edge in projection["edges"]:
        (ax, ay), (bx, by) = positions[edge["source"]], positions[edge["target"]]
        ay += heights[edge["source"]]
        svg.parts.append(
            f'<path data-source="{edge["source"]}" data-target="{edge["target"]}" '
            f'd="M{ax + 190},{ay} C{ax + 190},{(ay + by) / 2} {bx + 190},{(ay + by) / 2} '
            f'{bx + 190},{by - 4}" fill="none" stroke="#647b8c" stroke-width="1.5" '
            f'marker-end="url(#{marker})"/>'
        )
    for node in nodes:
        x, top = positions[node["id"]]
        assigned = " ".join(f["id"] for f in projection["features"] if node["id"] in f["nodes"])
        svg.parts.append(f'<g data-node="{node["id"]}" data-features="{assigned}">')
        svg.rect(x, top, 380, heights[node["id"]])
        for index, (text, feature, bold) in enumerate(contents[node["id"]]):
            attrs = f'data-feature="{feature}" class="feature" tabindex="0"' if feature else ""
            svg.text(x + 16, top + 25 + 20 * index, text, bold=bold, attrs=attrs)
        svg.parts.append("</g>")
    unplaced = [f for f in projection["features"] if not f["nodes"] and not f["key_component"]]
    if unplaced:
        svg.text(24, y, "Not placed by tested keys", bold=True)
        y += 24
        for feature in unplaced:
            for line in wrap(feature["label"] + " · " + feature["reason"].replace("_", " "), 100):
                svg.text(24, y, line, attrs=f'data-feature="{feature["id"]}" class="feature"')
                y += 20
    return svg.finish(width, y + 20)


def matrix_keys(projection: Mapping[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            [key for node in projection["nodes"] for key in node["key_names"]]
            + [row["key"] for row in projection["evidence"]]
        )
    )


def grain_matrix(projection: Mapping[str, Any]) -> str:
    """Each feature's behavior (constant, varying...) under each candidate key."""
    svg = SVG("Candidate key × target feature")
    y = _heading(svg, projection, "Candidate key × target feature")
    svg.text(24, y, "Constant · varying · undefined · untested | * different target population")
    y += 32
    keys = matrix_keys(projection)
    header = [wrap(key, 16) for key in keys]
    for index, lines in enumerate(header):
        for line_index, line in enumerate(lines):
            svg.text(300 + index * 140 + 6, y + line_index * 18, line, size=12)
    y += max((len(lines) for lines in header), default=1) * 18 + 10
    lookup = {(r["key"], r["feature"]): r for r in projection["evidence"]}
    for feature in projection["features"]:
        lines = wrap(feature["label"], 30)
        height = max(44, 20 * len(lines) + 10)
        for i, line in enumerate(lines):
            svg.text(24, y + 24 + i * 20, line)
        for index, key in enumerate(keys):
            record = lookup.get((key, feature["id"]))
            state = record["state"] if record else "untested"
            x = 300 + index * 140
            svg.rect(x, y, 134, height - 6, fill=COLORS[state])
            marker = " *" if record and not record["compatible"] else ""
            attrs = f'data-feature="{feature["id"]}" class="feature" tabindex="0"'
            svg.text(x + 8, y + 24, state + marker, size=12, attrs=attrs)
        y += height
    return svg.finish(max(900, 300 + 140 * len(keys) + 24), y + 20)


def bars(projection: Mapping[str, Any]) -> str:
    """Levels per feature, or the census tree, as aligned share-of-total bars."""
    is_tree = projection["kind"] == "census"
    svg = SVG("Observed census" if is_tree else "Feature levels")
    y = _heading(svg, projection, svg.title)
    full = projection["detail"] == "full"
    groups = (
        [{"label": "Bars show share of all evaluated rows", "rows": projection["rows"]}]
        if is_tree
        else projection["features"]
    )
    bar_x = 380 + max((r.get("depth", 0) for g in groups for r in g["rows"]), default=0) * 20
    width = max(980, bar_x + 550)
    for group in groups:
        for line in wrap(group["label"], 100):
            svg.text(24, y, line if full or not is_tree else "Observed paths", bold=True)
            y += 22
        for line in wrap(group.get("scope", ""), 100) if group.get("scope") else []:
            svg.text(24, y, line, size=12)
            y += 18
        for row in group["rows"]:
            y = _bar(svg, row, y, bar_x, is_tree, full)
        y += 22
    return svg.finish(width, y + 10)


def _bar(svg: SVG, row, y, bar_x, is_tree, full) -> int:
    x = 24 + row.get("depth", 0) * 20
    attrs = f'data-row="{row["id"]}" data-parent="{row.get("parent") or ""}"' if is_tree else ""
    svg.parts.append(f"<g {attrs}>")
    lines = wrap(row["label"], 42)
    if is_tree:
        svg.parts.append(
            f'<path d="M{x},{y - 10} v8 h8 m-3,-3 l3,3 -3,3" fill="none" stroke="#647b8c"/>'
        )
    for i, line in enumerate(lines):
        svg.text(x + (14 if is_tree else 0), y + i * 18, line, size=13)
    if full:
        share = row.get("share")
        svg.rect(bar_x, y - 13, 300, 16, fill="#edf1f5")
        fill = "#a5aeb8" if row.get("omitted") else "#65a5b3"
        svg.rect(bar_x, y - 13, 300 * (share or 0), 16, fill=fill)
        value = f"{row['count']} rows · {share:.1%}" if share is not None else "0 rows · undefined"
        svg.text(bar_x + 312, y, value, size=12)
    svg.parts.append("</g>")
    return y + max(28, len(lines) * 18 + 8)


_REVERSE = {"1:n": "n:1", "n:1": "1:n", "1:1": "1:1", "n:m": "n:m", "undefined": "undefined"}


def pairs(projection: Mapping[str, Any], *, association: bool = False) -> str:
    """A feature x feature matrix per context: relations, or Cramér's V."""
    svg = SVG("Pair association" if association else "Directional pair mappings")
    y = _heading(svg, projection, svg.title)
    features = projection["features"]
    if projection["omitted"]:
        svg.text(24, y, "Additional pairs or contexts omitted by analysis budgets.")
        y += 26
    for context in projection["contexts"]:
        for line in wrap(context["label"], 100):
            svg.text(24, y, line, bold=True)
            y += 24
        header = [wrap(feature, 14) for feature in features]
        for i, lines in enumerate(header):
            for j, line in enumerate(lines):
                svg.text(240 + i * 125, y + 18 * j, line, size=12)
        y += max((len(lines) for lines in header), default=1) * 18 + 12
        lookup = {}
        for cell in context["cells"]:
            lookup[(cell["a"], cell["b"])] = cell
            lookup[(cell["b"], cell["a"])] = {**cell, "relation": _REVERSE[cell["relation"]]}
        for i, feature in enumerate(features):
            y = _pair_row(svg, feature, i, len(features), lookup, y, association)
        y += 32
    return svg.finish(max(900, 240 + len(features) * 125 + 24), y + 20)


def _pair_row(svg, feature, i, count, lookup, y, association) -> int:
    lines = wrap(feature, 25)
    height = max(48, len(lines) * 18 + 10)
    for k, line in enumerate(lines):
        svg.text(24, y + 24 + 18 * k, line, size=12)
    for j in range(count):
        cell = lookup.get((i, j))
        value, fill = ("—" if i == j else "untested"), "#ffffff"
        if cell and association:
            v = cell["association"]
            value = f"{v:.3f}" if v is not None else "undefined"
            fill = f"rgb({int(245 - 120 * (v or 0))}, {int(248 - 75 * (v or 0))}, 230)"
        elif cell:
            value, fill = cell["relation"], COLORS[cell["relation"]]
        x = 240 + j * 125
        svg.parts.append("<g>")
        if cell:
            title = cell["scope"]
            if association and cell["association"] is None:
                title += " · " + str(cell["association_reason"])
            svg.parts.append(f"<title>{esc(title)}</title>")
        svg.rect(x, y, 119, height - 6, fill=fill)
        svg.text(x + 12, y + 24, value, size=12)
        svg.parts.append("</g>")
    return y + height


def heatmap(projection: Mapping[str, Any]) -> str:
    """Observed cells of one pair; blank cells were not observed."""
    svg = SVG("Selected-pair joint cells")
    y = _heading(svg, projection, "Selected-pair joint cells")
    columns = projection["columns"]
    for line in wrap("Rows: " + columns[0] + " · Columns: " + columns[1], 100):
        svg.text(24, y, line)
        y += 20
    y += 8
    headers = [wrap(v, 13) for v in projection["b"]]
    for j, lines in enumerate(headers):
        for k, line in enumerate(lines):
            svg.text(240 + j * 120, y + k * 18, line, size=12)
    y += max((len(lines) for lines in headers), default=1) * 18 + 10
    lookup = {(c["a"], c["b"]): c for c in projection["cells"]}
    full = projection["detail"] == "full"
    maximum = max((c.get("count", 1) for c in projection["cells"]), default=1)
    for i, value in enumerate(projection["a"]):
        lines = wrap(value, 25)
        height = max(44, len(lines) * 18 + 10)
        for k, line in enumerate(lines):
            svg.text(24, y + 24 + k * 18, line, size=12)
        for j in range(len(projection["b"])):
            cell = lookup.get((i, j))
            fraction = cell["count"] / maximum if cell and full else 0
            fill = "#ffffff" if not cell else "#dcf3e7"
            if cell and full:
                fill = f"rgb({int(235 - 140 * fraction)}, {int(245 - 85 * fraction)}, 210)"
            svg.rect(240 + j * 120, y, 114, height - 6, fill=fill)
            text = cell["count"] if cell and full else "observed" if cell else "—"
            svg.text(250 + j * 120, y + 24, text, size=12)
        y += height
    return svg.finish(max(900, 264 + 120 * len(projection["b"])), y + 20)


def findings(projection: Mapping[str, Any], max_findings: int) -> str:
    """Evidence cards: findings, candidates and tests, or availability bars."""
    kind, full = projection["kind"], projection["detail"] == "full"
    svg = SVG(f"Fieldwork {kind}")
    svg.text(24, 36, f"Fieldwork / {kind.replace('_', ' ').title()}", size=23, bold=True)
    svg.text(24, 62, _subtitle(projection), size=13)
    y = 92
    for text in _context_lines(projection):
        for line in wrap(text, 102):
            svg.text(24, y, line, size=13)
            y += 20
    if full and "availability" in projection:
        rows = projection["availability"]
        displayed = [("availability features", len(rows), max_findings)]
        for row in rows[:max_findings]:
            y = _availability_bar(svg, row, y)
    else:
        displayed, cards = _cards(projection, max_findings)
        for card in cards:
            y = _card(svg, card, y, full)
    if not any(total for _, total, _ in displayed):
        svg.text(
            24, y + 14, "No records saved · review search coverage and analysis settings", size=13
        )
        y += 30
    for name, total, shown in displayed:
        if total > shown:
            text = f"{name.title()}: {shown}/{total} shown · display limit reached"
            svg.text(
                24,
                y + 14,
                text if full else "More evidence available · display limit reached",
                size=12,
            )
            y += 30
    return svg.finish(864, y + 20)


def _subtitle(projection) -> str:
    if projection["detail"] == "topology":
        return "Topology only"
    if projection["kind"] == "comparison":
        return "Availability comparison · before → after"
    unit = projection.get("analysis_unit")
    described = (
        f"{unit['denominator']} {unit['counting_unit']} · {unit['presence_aggregation']}"
        if unit and "denominator" in unit
        else "saved evidence"
    )
    return f"{projection.get('scope', {}).get('evaluated_rows', 0)} evaluated rows · {described}"


def _context_lines(projection) -> list[str]:
    lines = comparison_labels(projection)
    if projection.get("section_selection", {}).get("omitted"):
        lines.append("Not requested: " + ", ".join(projection["section_selection"]["omitted"]))
    if projection.get("skipped_features"):
        lines.append(skipped_label(projection["skipped_features"]))
    if "analysis_unit" in projection and projection["kind"] != "comparison":
        lines.append("Analysis: " + unit_label(projection["analysis_unit"]))
    return lines


def _availability_bar(svg: SVG, row, y) -> int:
    for line in wrap(row["feature"], 28):
        svg.text(24, y + 18, line, size=13)
        y += 18
    fraction = row["populated_fraction"] or 0
    svg.rect(270, y - 1, 400, 17, fill="#e2e8f0")
    if fraction:
        svg.rect(270, y - 1, round(400 * fraction, 2), 17, fill="#66a89b")
    svg.text(685, y + 13, f"{row['populated']}/{row['denominator']}", size=12)
    return y + 28


def _cards(projection, max_findings) -> tuple[list[tuple[str, int, int]], list[dict[str, Any]]]:
    """Cards to draw, and each list's total and shown count."""
    if projection["kind"] == "schema_proposal":
        proposals = projection["proposals"]
        rows = [
            {
                "statement": f"{p['label']}: suggested {p['role']}",
                "explanation": "; ".join(
                    f"{r['code'].lower()}: {r['value']}" for r in p["reasons"]
                ),
            }
            for p in proposals[:max_findings]
        ]
        return [("proposals", len(proposals), max_findings)], rows
    rows = list(projection["findings"][:max_findings])
    displayed = [("findings", len(projection["findings"]), max_findings)]
    full = projection["detail"] == "full"
    if full and projection["kind"] in {"overview", "dependencies"}:
        candidates = projection.get("candidates", projection.get("overview", {}).get("grains", []))
        shown = min(5, max_findings)
        cards = [
            {
                "statement": grain_title(c) + ": " + c["role"],
                "explanation": candidate_explanation(c),
            }
            for c in candidates[:shown]
        ]
        displayed.append(("candidate grains", len(candidates), shown))
        if projection["kind"] == "dependencies":
            tests = projection["dependencies"]
            displayed = [displayed[-1], ("dependency tests", len(tests), max_findings)]
            rows = [
                {"statement": dependency_label(d), "explanation": d["explanation"]}
                for d in tests[:max_findings]
            ]
        rows = cards + rows
    return displayed, rows


def _card(svg: SVG, row, y, full) -> int:
    lines = wrap(row["statement"], 90)
    metrics = unit_label(row["analysis_unit"]) if "analysis_unit" in row else ""
    measurements = row.get("measurements", {})
    if full and measurements:
        metrics = f"{row['counting_unit']} · " + " · ".join(
            f"{k}: {v:.3g}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in measurements.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        if measurements.get("explanation"):
            metrics = measurements["explanation"]
    if full and "explanation" in row:
        metrics = row["explanation"]
    if not full and "explanation" in row and "analysis_unit" not in row:
        metrics = row["explanation"]
    detail = wrap(metrics, 102) if metrics else []
    height = 26 + len(lines) * 20 + len(detail) * 17
    svg.rect(20, y, 820, height)
    cursor = y + 24
    for line in lines:
        svg.text(34, cursor, line, bold=True)
        cursor += 20
    for line in detail:
        svg.text(34, cursor, line, size=12)
        cursor += 17
    return y + height + 10


VIEWS: dict[str, tuple[str, ...]] = {
    "grain": ("map", "matrix"),
    "pairs": ("mapping", "association"),
    "levels": ("bars",),
    "census": ("tree",),
    "joint_counts": ("heatmap",),
}


def figure(
    projection: Mapping[str, Any],
    view: str | None = None,
    *,
    show_exceptions: bool = False,
    max_findings: int = 12,
) -> str:
    """The SVG for a projection, in its default or a requested view."""
    kind = projection["kind"]
    valid = VIEWS.get(kind, ("findings",))
    view = view or valid[0]
    if view not in valid:
        raise ValueError(f"Invalid view {view!r} for {kind}; choose from {valid}")
    if view == "association" and projection["detail"] != "full":
        raise ValueError("Association view requires detail='full'")
    drawers: dict[str, Callable[[], str]] = {
        "map": lambda: grain_map(projection, exceptions=show_exceptions),
        "matrix": lambda: grain_matrix(projection),
        "mapping": lambda: pairs(projection),
        "association": lambda: pairs(projection, association=True),
        "bars": lambda: bars(projection),
        "tree": lambda: bars(projection),
        "heatmap": lambda: heatmap(projection),
        "findings": lambda: findings(projection, max_findings),
    }
    return drawers[view]()
