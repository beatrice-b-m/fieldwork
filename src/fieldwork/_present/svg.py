"""Self-contained SVG figures of projections: one function per figure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

from . import style as st
from .common import esc, wrap
from .project import candidate_explanation, dependency_label, grain_title, structure_notes
from .text import comparison_labels, skipped_label, unit_label


class SVG:
    """An SVG document assembled from escaped text and shapes, always on white."""

    def __init__(self, title: str, *, role: str = "img", marker: str = "arrow"):
        self.title, self.role, self.marker = title, role, marker
        self.parts: list[str] = []

    def text(
        self, x, y, value, *, size=13, bold=False, mono=False, color=None, anchor=None, attrs=""
    ):
        extra = f' font-family="{st.MONO}"' if mono else ""
        extra += f' fill="{color}"' if color else ""
        extra += f' text-anchor="{anchor}"' if anchor else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" '
            f'font-weight="{700 if bold else 400}"{extra} {attrs}>{esc(value)}</text>'
        )

    def rect(
        self, x, y, width, height, *, fill=st.PAPER, stroke=st.INK, dashed=False, rx=4, attrs=""
    ):
        outline = f' stroke="{stroke}" stroke-width="1.2"' if stroke else ""
        outline += ' stroke-dasharray="5 4"' if dashed and stroke else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{rx}" fill="{fill}"{outline} {attrs}/>'
        )

    def line(self, x1, y1, x2, y2, *, stroke=st.LINE, width=1):
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{width}"/>'
        )

    def finish(self, width, height) -> str:
        marker = (
            f'<defs><marker id="{self.marker}" viewBox="0 0 8 8" markerWidth="8" '
            'markerHeight="8" markerUnits="userSpaceOnUse" refX="7" refY="4" orient="auto">'
            f'<path d="M0 0L8 4L0 8z" fill="{st.INK}"/></marker></defs>'
            if any("marker-end=" in part for part in self.parts)
            else ""
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" role="{self.role}" '
            f'aria-label="{esc(self.title)}" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" font-family="{st.SANS}" fill="{st.TEXT}">'
            f"<title>{esc(self.title)}</title>"
            + marker
            + f'<rect width="100%" height="100%" fill="{st.PAPER}"/>'
            + "".join(self.parts)
            + "</svg>"
        )


def _heading(svg: SVG, projection: Mapping[str, Any], title: str) -> int:
    svg.text(24, 40, title, size=24, bold=True)
    y = 64
    lines = [projection.get("caption", ""), projection.get("scope", "")]
    if projection["detail"] == "topology":
        lines.append("Topology only · quantitative evidence suppressed")
    if projection.get("missingness"):
        lines.append("Missingness: " + projection["missingness"])
    for line in lines:
        for wrapped in wrap(line, 105) if line else []:
            svg.text(24, y, wrapped, color=st.INK_MUTED)
            y += 18
    return y + 20


def _legend(svg: SVG, y, items, *, columns=2, width=300) -> int:
    """A legend at a fixed place below a grid: (fill, stroke, dashed, icon, label, text)."""
    svg.line(24, y, 24 + columns * width - 24, y)
    y += 12
    for index, (fill, stroke, dashed, kind, name, text) in enumerate(items):
        x, top = 24 + (index % columns) * width, y + (index // columns) * 26
        svg.rect(x, top, 22, 18, fill=fill, stroke=stroke, dashed=dashed)
        if kind:
            svg.parts.append(st.icon(kind, stroke, x + 4, top + 2))
        svg.parts.append(
            f'<text x="{x + 32}" y="{top + 14}" font-size="13"><tspan font-weight="700">'
            f"{esc(name)}</tspan>  {esc(text)}</text>"
        )
    return y + ((len(items) + columns - 1) // columns) * 26


def _range_legend(svg: SVG, y, cuts, *, counts, empty) -> int:
    """The four shading ranges with the cut-offs used, then the empty-cell swatch."""
    labels = st.range_labels(cuts, counts=counts)
    for index, (fill, text) in enumerate(zip(st.AMOUNT_SCALE, labels, strict=True)):
        svg.rect(24 + index * 100, y, 94, 16, fill=fill, stroke=None, rx=0)
        svg.text(24 + index * 100, y + 32, text, size=11)
    fill, stroke, dashed, text = empty
    svg.rect(24 + 4 * 100 + 16, y, 40, 16, fill=fill, stroke=stroke, dashed=dashed, rx=0)
    svg.text(24 + 4 * 100 + 16, y + 32, text, size=11)
    return y + 48


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


def _node_lines(node, projection, features, exceptions, collapsed) -> list[tuple[str, Any, str]]:
    """Card lines as (text, feature id, kind); kind is title, plain, label or feature."""
    lines: list[tuple[str, Any, str]] = [(title, None, "title") for title in node["titles"]]
    if len(node["titles"]) > 1:
        lines.append(("Observationally equivalent keys", None, "plain"))
    lines.append((node["role"].capitalize(), None, "plain"))
    support = node.get("support")
    if support:
        lines.append(
            (
                f"{support['evaluated_groups']} groups · {support['evaluated_rows']} rows",
                None,
                "plain",
            )
        )
        lines.append(
            (
                f"{support['singleton_groups']} singleton · {support['repeated_groups']} repeated groups",
                None,
                "plain",
            )
        )
    lines.append(("Attributes collapsed" if collapsed else "Attributes", None, "label"))
    for feature_id in [] if collapsed else node["attributes"]:
        feature = features[feature_id]
        shared = " (shared)" if len(feature["nodes"]) > 1 else ""
        lines.append((feature["label"] + shared, feature_id, "feature"))
    if not node["attributes"]:
        lines.append(("No assigned attributes", None, "plain"))
    for record in projection["evidence"] if exceptions else []:
        if (
            record["key"] == node["key_names"][0]
            and record["state"] == "varying"
            and record["compatible"]
        ):
            text = "Varies: " + features[record["feature"]]["label"]
            if projection["detail"] == "full":
                text += f" · {record['violating_groups']}/{record['evaluated_groups']} groups"
            lines.append((text, record["feature"], "plain"))
    # Key names are set larger in monospace, so they wrap sooner.
    return [
        (part, feature, kind)
        for text, feature, kind in lines
        for part in wrap(text, 36 if kind == "title" else 42)
    ]


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
    y = _heading(svg, {**projection, "caption": _join(projection.get("caption"))}, svg.title)
    features = {f["id"]: f for f in projection["features"]}
    nodes = projection["nodes"]
    ranks = _ranks(projection)
    layers = defaultdict(list)
    for node in nodes:
        layers[ranks[node["id"]]].append(node)
    contents = {n["id"]: _node_lines(n, projection, features, exceptions, collapsed) for n in nodes}
    heights = {node_id: 38 + 20 * len(lines) for node_id, lines in contents.items()}
    width = max(900, max((len(layer) for layer in layers.values()), default=1) * 400 + 24)
    positions = {}
    for rank in sorted(layers):
        start = (width - len(layers[rank]) * 400 + 20) / 2
        for index, node in enumerate(layers[rank]):
            positions[node["id"]] = (start + index * 400, y)
        y += max(heights[n["id"]] for n in layers[rank]) + 70
    # Orthogonal edges: every edge into a target turns on the same row above it,
    # so several sources join one trunk that enters the target once.
    for edge in projection["edges"]:
        (ax, ay), (bx, by) = positions[edge["source"]], positions[edge["target"]]
        ay += heights[edge["source"]]
        svg.parts.append(
            f'<path data-source="{edge["source"]}" data-target="{edge["target"]}" '
            f'd="M{ax + 190} {ay}V{by - 30}H{bx + 190}V{by - 2}" fill="none" '
            f'stroke="{st.INK}" stroke-width="1.2" marker-end="url(#{marker})"/>'
        )
    for node in nodes:
        x, top = positions[node["id"]]
        assigned = " ".join(f["id"] for f in projection["features"] if node["id"] in f["nodes"])
        svg.parts.append(f'<g data-node="{node["id"]}" data-features="{assigned}">')
        svg.rect(x, top, 380, heights[node["id"]])
        cursor, previous = top + 26, "title"
        for text, feature, kind in contents[node["id"]]:
            if previous == "title" and kind != "title":  # a rule under the key names
                svg.line(x + 12, cursor - 12, x + 368, cursor - 12)
                cursor += 8
            previous = kind
            attrs = f'data-feature="{feature}" class="feature" tabindex="0"' if feature else ""
            svg.text(
                x + 16,
                cursor,
                text,
                size={"title": 15, "label": 11, "feature": 13}.get(kind, 13),
                bold=kind in {"title", "label"},
                mono=kind in {"title", "feature"},
                color=st.INK_MUTED if kind == "label" else None,
                attrs=attrs,
            )
            cursor += 20
        svg.parts.append("</g>")
    unplaced = [f for f in projection["features"] if not f["nodes"] and not f["key_component"]]
    if unplaced:
        lines = [
            (part, feature["id"])
            for feature in unplaced
            for part in wrap(feature["label"] + " · " + feature["reason"].replace("_", " "), 90)
        ]
        # Dashed: these columns are outside what the tested keys explain.
        svg.rect(24, y, width - 48, 54 + 20 * len(lines), stroke=st.MUTED, dashed=True)
        svg.text(40, y + 26, "Not placed by tested keys", bold=True)
        svg.line(36, y + 38, width - 36, y + 38)
        for index, (text, feature_id) in enumerate(lines):
            svg.text(
                40,
                y + 60 + 20 * index,
                text,
                mono=True,
                attrs=f'data-feature="{feature_id}" class="feature"',
            )
        y += 54 + 20 * len(lines) + 20
    return svg.finish(width, y + 20)


def _join(caption) -> str:
    """The map's caption with the arrow reading, so no annotation needs placing."""
    note = "Arrows point to finer groupings."
    return f"{caption} {note}" if caption else note


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
    keys = matrix_keys(projection)
    width = max(900, 300 + 140 * len(keys) + 24)
    header = [wrap(key, 16) for key in keys]
    for index, lines in enumerate(header):
        for line_index, line in enumerate(lines):
            svg.text(
                300 + index * 140 + 6, y + line_index * 18, line, size=12, bold=True, mono=True
            )
    y += max((len(lines) for lines in header), default=1) * 18 - 8
    svg.line(24, y, width - 24, y, stroke=st.INK)
    y += 10
    lookup = {(r["key"], r["feature"]): r for r in projection["evidence"]}
    for feature in projection["features"]:
        lines = wrap(feature["label"], 30)
        height = max(44, 20 * len(lines) + 10)
        for i, line in enumerate(lines):
            svg.text(24, y + 24 + i * 20, line, mono=True)
        for index, key in enumerate(keys):
            record = lookup.get((key, feature["id"]))
            state = record["state"] if record else "untested"
            fill, stroke, dashed, kind = st.STATES[state]
            x = 300 + index * 140
            svg.rect(x, y, 134, height - 6, fill=fill, stroke=stroke, dashed=dashed)
            if kind:
                svg.parts.append(st.icon(kind, stroke, x + 10, y + 12))
            marker = " *" if record and not record["compatible"] else ""
            attrs = f'data-feature="{feature["id"]}" class="feature" tabindex="0"'
            svg.text(
                x + 30,
                y + 24,
                state + marker,
                color=st.INK_MUTED if state == "untested" else None,
                attrs=attrs,
            )
        y += height
    items = [
        (*st.STATES["constant"], "Constant", "one value per group"),
        (*st.STATES["varying"], "Varying", "conflicting values"),
        (*st.STATES["undefined"], "Undefined", "no evaluated support"),
        (*st.STATES["untested"], "Untested", "no saved test"),
    ]
    y = _legend(svg, y + 16, items)
    svg.text(24, y + 16, "* the target's evaluated population differs from the map population")
    return svg.finish(width, y + 36)


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
            svg.text(24, y, line if full or not is_tree else "Observed paths", size=15, bold=True)
            y += 22
        for line in wrap(group.get("scope", ""), 100) if group.get("scope") else []:
            svg.text(24, y, line, size=12, color=st.INK_MUTED)
            y += 18
        y += 8
        rows_at: dict[str, tuple[int, int]] = {}
        for row in group["rows"]:
            y = _bar(svg, row, y, bar_x, is_tree, full, rows_at)
        y += 22
    if full and any(row.get("omitted") for g in groups for row in g["rows"]):
        svg.rect(24, y - 4, 22, 14, fill=st.PAPER, stroke=st.MUTED, dashed=True, rx=0)
        svg.text(54, y + 8, "Omitted branches: outside the shown paths, mass kept")
        y += 30
    return svg.finish(width, y + 10)


def _bar(svg: SVG, row, y, bar_x, is_tree, full, rows_at) -> int:
    depth = row.get("depth", 0)
    x = 24 + depth * 20
    attrs = f'data-row="{row["id"]}" data-parent="{row.get("parent") or ""}"' if is_tree else ""
    svg.parts.append(f"<g {attrs}>")
    lines = wrap(row["label"], 42)
    if is_tree and row.get("parent") in rows_at:
        # A thin elbow guide from the parent row, not an arrow per row.
        parent_y, parent_depth = rows_at[row["parent"]]
        guide_x = 24 + parent_depth * 20 + 4
        svg.parts.append(
            f'<path d="M{guide_x} {parent_y + 5}V{y - 4}H{x + 8}" fill="none" '
            f'stroke="{st.MUTED}" stroke-width="0.6"/>'
        )
    if is_tree:
        rows_at[row["id"]] = (y, depth)
    for i, line in enumerate(lines):
        svg.text(x + (12 if is_tree and depth else 0), y + i * 18, line, bold=is_tree and not depth)
    if full:
        share = row.get("share")
        svg.rect(bar_x, y - 12, 300, 14, fill=st.SURFACE, stroke=st.LINE, rx=0)
        if row.get("omitted"):
            svg.rect(bar_x, y - 12, 300 * (share or 0), 14, stroke=st.MUTED, dashed=True, rx=0)
        else:
            svg.rect(bar_x, y - 12, 300 * (share or 0), 14, fill=st.AMOUNT, stroke=None, rx=0)
        value = f"{row['count']} rows · {share:.1%}" if share is not None else "0 rows · undefined"
        svg.text(bar_x + 316, y, value, attrs='style="font-variant-numeric:tabular-nums"')
    svg.parts.append("</g>")
    return y + max(30, len(lines) * 18 + 10)


_REVERSE = {"1:n": "n:1", "n:1": "1:n", "1:1": "1:1", "n:m": "n:m", "undefined": "undefined"}


def pairs(projection: Mapping[str, Any], *, association: bool = False) -> str:
    """A feature x feature matrix per context: relations, or Cramér's V."""
    svg = SVG("Pair association" if association else "Directional pair mappings")
    cuts = projection.get("shading", {}).get("association") if association else None
    y = _heading(svg, projection, svg.title)
    features = projection["features"]
    if projection["omitted"]:
        svg.text(24, y, "Additional pairs or contexts omitted by analysis budgets.")
        y += 26
    for context in projection["contexts"]:
        for line in wrap(context["label"], 100):
            svg.text(24, y, line, bold=True)
            y += 24
        header = [wrap(feature, 16) for feature in features]
        for i, lines in enumerate(header):
            for j, line in enumerate(lines):
                svg.text(240 + i * 125, y + 18 * j, line, size=12, bold=True, mono=True)
        y += max((len(lines) for lines in header), default=1) * 18 - 6
        svg.line(24, y, 240 + len(features) * 125 - 6, y, stroke=st.INK)
        y += 12
        lookup = {}
        for cell in context["cells"]:
            lookup[(cell["a"], cell["b"])] = cell
            lookup[(cell["b"], cell["a"])] = {**cell, "relation": _REVERSE[cell["relation"]]}
        for i, feature in enumerate(features):
            y = _pair_row(svg, feature, i, len(features), lookup, y, association, cuts)
        y += 24
    if association and cuts:
        y = _range_legend(svg, y, cuts, counts=False, empty=(st.PAPER, st.MUTED, True, "undefined"))
    elif not association:
        items = [
            (*st.RELATIONS["n:1"], None, "n:1", "row determines column"),
            (*st.RELATIONS["1:n"], None, "1:n", "column determines row"),
            (*st.RELATIONS["1:1"], None, "1:1", "determined both ways"),
            (*st.RELATIONS["n:m"], None, "n:m", "neither determines the other"),
        ]
        y = _legend(svg, y, items, width=360)
    return svg.finish(max(900, 240 + len(features) * 125 + 24), y + 20)


def _pair_row(svg, feature, i, count, lookup, y, association, cuts) -> int:
    lines = wrap(feature, 25)
    height = max(48, len(lines) * 18 + 10)
    for k, line in enumerate(lines):
        svg.text(24, y + 24 + 18 * k, line, mono=True)
    for j in range(count):
        cell = lookup.get((i, j))
        x = 240 + j * 125
        svg.parts.append("<g>")
        if cell:
            title = cell["scope"]
            if association and cell["association"] is None:
                title += " · " + str(cell["association_reason"])
            svg.parts.append(f"<title>{esc(title)}</title>")
        color = None
        if i == j:
            value = "—"
            svg.rect(x, y, 119, height - 6, fill=st.SURFACE, stroke=None)
        elif not cell:
            value, color = "untested", st.INK_MUTED
            svg.rect(x, y, 119, height - 6, stroke=st.SUBTLE, dashed=True)
        elif association and cell["association"] is None:
            value, color = "undefined", st.INK_MUTED
            svg.rect(x, y, 119, height - 6, stroke=st.MUTED, dashed=True)
        elif association:
            v = cell["association"]
            index = st.shade(v, cuts)
            value = f"{v:.3f}"
            color = st.ON_DARKEST if index == 3 else None
            svg.rect(x, y, 119, height - 6, fill=st.AMOUNT_SCALE[index], stroke=None)
        else:
            value = cell["relation"]
            fill, stroke, dashed = st.RELATIONS[value]
            color = st.ON_DARKEST if value == "1:1" else None
            svg.rect(x, y, 119, height - 6, fill=fill, stroke=stroke, dashed=dashed)
        svg.text(x + 60, y + 24, value, bold=not association, color=color, anchor="middle")
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
            svg.text(240 + j * 120 + 57, y + k * 18, line, size=12, bold=True, anchor="middle")
    y += max((len(lines) for lines in headers), default=1) * 18 - 6
    svg.line(24, y, 240 + 120 * len(projection["b"]) - 6, y, stroke=st.INK)
    y += 12
    lookup = {(c["a"], c["b"]): c for c in projection["cells"]}
    full = projection["detail"] == "full"
    cuts = projection.get("shading", {}).get("counts")
    for i, value in enumerate(projection["a"]):
        lines = wrap(value, 25)
        height = max(44, len(lines) * 18 + 10)
        for k, line in enumerate(lines):
            svg.text(24, y + 24 + k * 18, line, size=12)
        for j in range(len(projection["b"])):
            cell = lookup.get((i, j))
            x, color = 240 + j * 120, None
            if not cell:
                svg.rect(x, y, 114, height - 6, stroke=st.LINE)
            elif full and cuts:
                index = st.shade(cell["count"], cuts)
                color = st.ON_DARKEST if index == 3 else None
                svg.rect(x, y, 114, height - 6, fill=st.AMOUNT_SCALE[index], stroke=None)
            else:
                svg.rect(x, y, 114, height - 6, fill=st.PANEL, stroke=None)
            text = cell["count"] if cell and full else "observed" if cell else "—"
            svg.text(
                x + 57,
                y + 24,
                text,
                color=color or (None if cell else st.INK_MUTED),
                anchor="middle",
            )
        y += height
    if full and cuts:
        y = _range_legend(svg, y + 20, cuts, counts=True, empty=(st.PAPER, st.LINE, False, "none"))
    return svg.finish(max(900, 264 + 120 * len(projection["b"])), y + 20)


def findings(projection: Mapping[str, Any], max_findings: int) -> str:
    """Evidence cards: findings, candidates and tests, or availability bars."""
    kind, full = projection["kind"], projection["detail"] == "full"
    svg = SVG(f"Fieldwork {kind}")
    svg.text(24, 40, f"Fieldwork / {kind.replace('_', ' ').title()}", size=24, bold=True)
    svg.text(24, 64, _subtitle(projection), color=st.INK_MUTED)
    y = 94
    for text in _context_lines(projection):
        for line in wrap(text, 102):
            svg.text(24, y, line, color=st.INK_MUTED)
            y += 20
    if full and "availability" in projection:
        rows = projection["availability"]
        displayed = [("availability features", len(rows), max_findings)]
        y += 8
        for row in rows[:max_findings]:
            y = _availability_bar(svg, row, y)
        if rows[:max_findings]:
            svg.rect(24, y + 4, 22, 14, fill=st.AMOUNT, stroke=None, rx=0)
            svg.text(54, y + 16, "Populated")
            svg.rect(150, y + 4, 22, 14, stroke=st.MUTED, dashed=True, rx=0)
            svg.text(180, y + 16, "Missing")
            y += 34
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
                color=st.INK_MUTED,
            )
            y += 30
    return svg.finish(864, y + 20)


def _subtitle(projection) -> str:
    if projection["detail"] == "topology":
        return "Topology only"
    if projection["kind"] == "comparison":
        return "Availability comparison · before → after"
    if projection["kind"] == "relation":
        return "Relation · left and right tables through a key"
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
    if "analysis_unit" in projection and projection["kind"] not in {"comparison", "relation"}:
        lines.append("Analysis: " + unit_label(projection["analysis_unit"]))
    return lines


def _availability_bar(svg: SVG, row, y) -> int:
    lines = wrap(row["feature"], 28)
    for index, line in enumerate(lines):
        svg.text(24, y + 13 + 18 * index, line, mono=True)
    fraction = row["populated_fraction"] or 0
    width = round(400 * fraction, 2)
    # Populated share solid; the missing remainder dashed. Labels sit in a fixed
    # column after the bar, whatever its length.
    if width:
        svg.rect(270, y, width, 18, fill=st.AMOUNT, stroke=None, rx=0)
    if width < 400:
        svg.rect(270 + width, y, 400 - width, 18, stroke=st.MUTED, dashed=True, rx=0)
    populated, total = row["populated"], row["denominator"]
    share = f" · {row['populated_fraction']:.0%}" if row["populated_fraction"] is not None else ""
    svg.parts.append(
        f'<text x="684" y="{y + 13}" font-size="13" style="font-variant-numeric:tabular-nums">'
        f"{populated} / {total}{share}"
        f'\u00a0<tspan fill="{st.INK_MUTED}">·\u00a0{total - populated} missing</tspan></text>'
    )
    return y + max(34, 18 * len(lines) + 12)


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
    if not full and projection["kind"] == "dependencies":
        candidates = projection["candidates"]
        shown = min(5, max_findings)
        cards = [{"statement": grain_title(c) + ": " + c["role"]} for c in candidates[:shown]]
        displayed.append(("candidate grains", len(candidates), shown))
        rows = cards + rows
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
    metrics = "; ".join(part for part in (metrics, *structure_notes(row)) if part)
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
    height = 26 + len(lines) * 20 + (len(detail) * 18 + 12 if detail else 0)
    svg.rect(24, y, 816, height)
    cursor = y + 25
    for line in lines:
        svg.text(38, cursor, line, size=14, bold=True)
        cursor += 20
    if detail:
        svg.line(36, cursor - 10, 828, cursor - 10)
        cursor += 8
    for line in detail:
        svg.text(38, cursor, line, size=12.5)
        cursor += 18
    return y + height + 12


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
