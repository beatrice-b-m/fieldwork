"""Dependency-free SVG figures and standalone HTML explorer presentations."""

from __future__ import annotations

import html
import textwrap
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Literal

from .render import _CONTROL
from .result import ExplorerResult
from .visual_data import visualization_data

_COLORS = {
    "constant": "#dcf3e7",
    "varying": "#ffe4d4",
    "undefined": "#eceef2",
    "untested": "#ffffff",
    "1:1": "#dcf3e7",
    "1:n": "#dcecff",
    "n:1": "#eee2ff",
    "n:m": "#ffe4d4",
}


def _esc(value: Any) -> str:
    return html.escape(_CONTROL.sub(lambda m: f"\\u{ord(m.group()):04x}", str(value)), quote=True)


def _wrap(text: str, width: int = 42) -> list[str]:
    # Account conservatively for wide glyphs without a font/runtime dependency.
    lines = []
    for line in textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False):
        part, used = "", 0.0
        for character in line:
            units = (
                2
                if unicodedata.east_asian_width(character) in {"W", "F"}
                else 1.9
                if character in "MW@%"
                else 1
            )
            if part and used + units > width:
                lines.append(part)
                part, used = "", 0.0
            part += character
            used += units
        if part:
            lines.append(part)
    return lines or [""]


class _SVG:
    def __init__(self, title: str):
        self.title = title
        self.parts: list[str] = []

    def text(self, x, y, value, *, size=14, bold=False, attrs=""):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" '
            f'font-weight="{600 if bold else 400}" {attrs}>{_esc(value)}</text>'
        )

    def rect(self, x, y, width, height, *, fill="#ffffff", attrs=""):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="7" fill="{fill}" stroke="#cbd5df" {attrs}/>'
        )

    def finish(self, width, height):
        marker = (
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" '
            'refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" '
            'fill="#647b8c"/></marker></defs>'
            if any("marker-end=" in part for part in self.parts)
            else ""
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="{_esc(self.title)}" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" '
            'font-family="Arial, sans-serif" fill="#193345">'
            f"<title>{_esc(self.title)}</title>"
            + marker
            + '<rect width="100%" height="100%" fill="#f6f8fb"/>'
            + "".join(self.parts)
            + "</svg>"
        )


def _heading(svg, data, title):
    svg.text(24, 32, title, size=22, bold=True)
    y = 55
    lines = [data.get("caption", ""), data.get("scope", "")]
    if data["detail"] == "topology":
        lines.append("Topology only · quantitative evidence suppressed")
    if data.get("missingness"):
        lines.append("Missingness: " + data["missingness"])
    for line in lines:
        if line:
            for wrapped in _wrap(line, 105):
                svg.text(24, y, wrapped, size=12)
                y += 18
    return y + 20


def _grain_svg(data, *, exceptions=False, collapsed=False):
    svg = _SVG("Observed grain map")
    y = _heading(svg, data, "Observed grain map")
    svg.text(24, y, "Arrows mean finer grouping ↓", size=13, bold=True)
    y += 30
    features = {f["id"]: f for f in data["features"]}
    nodes = data["nodes"]
    parents = defaultdict(list)
    for edge in data["edges"]:
        parents[edge["target"]].append(edge["source"])
    ranks = {}
    pending = {n["id"] for n in nodes}
    while pending:
        ready = [
            n["id"]
            for n in nodes
            if n["id"] in pending and all(p in ranks for p in parents[n["id"]])
        ]
        if not ready:
            raise ValueError("Grain graph must be acyclic")
        for node_id in ready:
            ranks[node_id] = max((ranks[p] + 1 for p in parents[node_id]), default=0)
            pending.remove(node_id)
    layers = defaultdict(list)
    for node in nodes:
        layers[ranks[node["id"]]].append(node)
    contents = {}
    heights = {}
    for node in nodes:
        lines = [(title, None, True) for title in node["titles"]]
        if len(node["titles"]) > 1:
            lines.append(("Observationally equivalent keys", None, False))
        support = node.get("support")
        if support:
            lines.extend(
                [
                    (
                        f"{support['evaluated_groups']} groups · {support['evaluated_rows']} rows",
                        None,
                        False,
                    ),
                    (
                        (
                            f"{support['singleton_groups']} singleton · "
                            f"{support['repeated_groups']} repeated groups"
                        ),
                        None,
                        False,
                    ),
                ]
            )
        lines.append(("Attributes collapsed" if collapsed else "Attributes", None, True))
        for feature_id in [] if collapsed else node["attributes"]:
            feature = features[feature_id]
            shared = " (shared)" if len(feature["nodes"]) > 1 else ""
            lines.append((feature["label"] + shared, feature_id, False))
        if not node["attributes"]:
            lines.append(("No assigned attributes", None, False))
        if exceptions:
            for record in data["evidence"]:
                if (
                    record["key"] == node["key_names"][0]
                    and record["state"] == "varying"
                    and record["compatible"]
                ):
                    text = "Varies: " + features[record["feature"]]["label"]
                    if data["detail"] == "full":
                        text += (
                            f" · {record['violating_groups']}/{record['evaluated_groups']} groups"
                        )
                    lines.append((text, record["feature"], False))
        wrapped = [(part, feature, bold) for text, feature, bold in lines for part in _wrap(text)]
        contents[node["id"]] = wrapped
        heights[node["id"]] = 30 + 20 * len(wrapped)
    width = max(900, max((len(layer) for layer in layers.values()), default=1) * 400 + 24)
    positions = {}
    for rank in sorted(layers):
        layer = layers[rank]
        start = (width - len(layer) * 400 + 20) / 2
        for index, node in enumerate(layer):
            positions[node["id"]] = (start + index * 400, y)
        y += max(heights[n["id"]] for n in layer) + 70
    for edge in data["edges"]:
        a, b = edge["source"], edge["target"]
        ax, ay = positions[a]
        bx, by = positions[b]
        ay += heights[a]
        svg.parts.append(
            f'<path data-source="{a}" data-target="{b}" '
            f'd="M{ax + 190},{ay} C{ax + 190},{(ay + by) / 2} '
            f'{bx + 190},{(ay + by) / 2} {bx + 190},{by - 4}" '
            'fill="none" stroke="#647b8c" stroke-width="1.5" '
            'marker-end="url(#arrow)"/>'
        )
    for node in nodes:
        node_id = node["id"]
        x, top = positions[node_id]
        svg.parts.append(f'<g data-node="{node_id}">')
        svg.rect(x, top, 380, heights[node_id])
        for index, (text, feature, bold) in enumerate(contents[node_id]):
            attrs = f'data-feature="{feature}" class="feature" tabindex="0"' if feature else ""
            svg.text(x + 16, top + 25 + 20 * index, text, bold=bold, attrs=attrs)
        svg.parts.append("</g>")
    unplaced = [f for f in data["features"] if not f["nodes"] and not f["key_component"]]
    if unplaced:
        svg.text(24, y, "Not placed by tested keys", bold=True)
        y += 24
        for feature in unplaced:
            text = feature["label"] + " · " + feature["reason"].replace("_", " ")
            for line in _wrap(text, 100):
                svg.text(24, y, line, attrs=f'data-feature="{feature["id"]}" class="feature"')
                y += 20
    return svg.finish(width, y + 20)


def _matrix_keys(data):
    return list(
        dict.fromkeys(
            [key for node in data["nodes"] for key in node["key_names"]]
            + [row["key"] for row in data["evidence"]]
        )
    )


def _matrix_svg(data):
    svg = _SVG("Candidate key × target feature")
    y = _heading(svg, data, "Candidate key × target feature")
    svg.text(24, y, "Constant · varying · undefined · untested | * different target population")
    y += 32
    features = data["features"]
    keys = _matrix_keys(data)
    label_width = 300
    header_lines = [_wrap(key, 16) for key in keys]
    header_height = max((len(lines) for lines in header_lines), default=1) * 18 + 10
    for index, lines in enumerate(header_lines):
        for line_index, line in enumerate(lines):
            svg.text(
                label_width + index * 140 + 6,
                y + line_index * 18,
                line,
                size=12,
            )
    y += header_height
    lookup = {(r["key"], r["feature"]): r for r in data["evidence"]}
    for feature in features:
        lines = _wrap(feature["label"], 30)
        height = max(44, 20 * len(lines) + 10)
        for i, line in enumerate(lines):
            svg.text(24, y + 24 + i * 20, line)
        for index, key in enumerate(keys):
            record = lookup.get((key, feature["id"]))
            state = record["state"] if record else "untested"
            x = label_width + index * 140
            svg.rect(x, y, 134, height - 6, fill=_COLORS[state])
            svg.text(
                x + 8,
                y + 24,
                state + (" *" if record and not record["compatible"] else ""),
                size=12,
                attrs=f'data-feature="{feature["id"]}" class="feature" tabindex="0"',
            )
        y += height
    return svg.finish(max(900, label_width + 140 * len(keys) + 24), y + 20)


def _matrix_html(data):
    """Keep wide grain evidence readable without scaling text down to fit."""
    keys = _matrix_keys(data)
    lookup = {(r["key"], r["feature"]): r for r in data["evidence"]}
    parts = [
        (
            "<h2>Feature behavior by candidate key</h2>"
            "<p>Each row is a feature; each column is a tested grouping. "
            "Constant means one observed value per group; varying means conflicting values. "
            "Undefined means no evaluated support; untested means no saved test. "
            "* marks a different target population. Select a cell, then View selected evidence. "
            "Long labels are shortened visually; full names appear in placement evidence.</p>"
        ),
        (
            '<div class="toolbar" data-enhance hidden><label>Find matrix features'
            '<input type="search" id="matrix-search" placeholder="Feature name"></label>'
            '<label>Candidate key<select id="matrix-key"><option value="">All candidate keys</option>'
        ),
    ]
    for index, key in enumerate(keys):
        parts.append(f'<option value="{index}">{_esc(key)}</option>')
    parts.append(
        '</select></label><button type="button" id="matrix-reset">Clear matrix filters</button>'
        '</div><p id="matrix-status" role="status" aria-live="polite"></p>'
        '<p id="matrix-empty" class="empty" hidden>No features match this search.</p>'
        '<div class="matrix-scroll" tabindex="0" role="region" aria-label="Grain evidence matrix">'
        '<table class="grain-matrix"><caption>Feature behavior by candidate key</caption>'
        '<thead><tr><th scope="col">Feature</th>'
    )
    for index, key in enumerate(keys):
        parts.append(
            f'<th scope="col" data-matrix-key="{index}" title="{_esc(key)}">'
            f'<span class="matrix-label">{_esc(key)}</span></th>'
        )
    parts.append("</tr></thead><tbody>")
    for feature in data["features"]:
        label = _esc(feature["label"])
        parts.append(
            f'<tr data-matrix-feature="{label}"><th scope="row">'
            f'<button type="button" data-feature="{feature["id"]}" class="feature" '
            f'title="{label}"><span class="matrix-label">{label}</span></button></th>'
        )
        for index, key in enumerate(keys):
            record = lookup.get((key, feature["id"]))
            state = record["state"] if record else "untested"
            different = bool(record and not record["compatible"])
            description = f"{feature['label']} by {key}: {state}"
            if different:
                description += "; different target population"
            parts.append(
                f'<td data-matrix-key="{index}" style="background:{_COLORS[state]}">'
                f'<button type="button" data-feature="{feature["id"]}" class="feature" '
                f'aria-label="{_esc(description)}">{state}{" *" if different else ""}'
                "</button></td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    if not keys or not data["features"]:
        parts.append(
            '<p class="empty">No candidate/feature cells are available in this saved result.</p>'
        )
    return "".join(parts)


def _bars_svg(data):
    is_tree = data["kind"] == "census"
    svg = _SVG("Observed census" if is_tree else "Feature levels")
    y = _heading(svg, data, svg.title)
    full = data["detail"] == "full"
    groups = (
        [{"label": "Bars show share of all evaluated rows", "rows": data["rows"]}]
        if is_tree
        else data["features"]
    )
    max_depth = max((r.get("depth", 0) for group in groups for r in group["rows"]), default=0)
    bar_x = 380 + max_depth * 20
    width = max(980, bar_x + 550)
    for group in groups:
        for line in _wrap(group["label"], 100):
            svg.text(24, y, line if full or not is_tree else "Observed paths", bold=True)
            y += 22
        if group.get("scope"):
            for line in _wrap(group["scope"], 100):
                svg.text(24, y, line, size=12)
                y += 18
        for row in group["rows"]:
            x = 24 + row.get("depth", 0) * 20
            attrs = (
                f'data-row="{row["id"]}" data-parent="{row.get("parent") or ""}"' if is_tree else ""
            )
            svg.parts.append(f"<g {attrs}>")
            lines = _wrap(row["label"], 42)
            if is_tree:
                svg.parts.append(
                    f'<path d="M{x},{y - 10} v8 h8 m-3,-3 l3,3 -3,3" fill="none" stroke="#647b8c"/>'
                )
            for i, line in enumerate(lines):
                svg.text(x + (14 if is_tree else 0), y + i * 18, line, size=13)
            if full:
                share = row.get("share")
                svg.rect(bar_x, y - 13, 300, 16, fill="#edf1f5")
                svg.rect(
                    bar_x,
                    y - 13,
                    300 * (share or 0),
                    16,
                    fill="#a5aeb8" if row.get("omitted") else "#65a5b3",
                )
                value = (
                    f"{row['count']} rows · {share:.1%}"
                    if share is not None
                    else "0 rows · undefined"
                )
                svg.text(bar_x + 312, y, value, size=12)
            svg.parts.append("</g>")
            y += max(28, len(lines) * 18 + 8)
        y += 22
    return svg.finish(width, y + 10)


def _pairs_svg(data, *, association=False):
    svg = _SVG("Pair association" if association else "Directional pair mappings")
    y = _heading(svg, data, svg.title)
    features = data["features"]
    label_width = 240
    cell_width = 125
    reverse = {"1:n": "n:1", "n:1": "1:n", "1:1": "1:1", "n:m": "n:m", "undefined": "undefined"}
    if data["omitted"]:
        svg.text(24, y, "Additional pairs or contexts omitted by analysis budgets.")
        y += 26
    for context in data["contexts"]:
        for line in _wrap(context["label"], 100):
            svg.text(24, y, line, bold=True)
            y += 24
        header = [_wrap(feature, 14) for feature in features]
        for i, lines in enumerate(header):
            for j, line in enumerate(lines):
                svg.text(label_width + i * cell_width, y + 18 * j, line, size=12)
        y += max((len(lines) for lines in header), default=1) * 18 + 12
        lookup = {}
        for cell in context["cells"]:
            lookup[(cell["a"], cell["b"])] = cell
            lookup[(cell["b"], cell["a"])] = {**cell, "relation": reverse[cell["relation"]]}
        for i, feature in enumerate(features):
            lines = _wrap(feature, 25)
            height = max(48, len(lines) * 18 + 10)
            for k, line in enumerate(lines):
                svg.text(24, y + 24 + 18 * k, line, size=12)
            for j in range(len(features)):
                cell = lookup.get((i, j))
                value = "—" if i == j else "untested"
                fill = "#ffffff"
                if cell:
                    if association:
                        v = cell["association"]
                        value = f"{v:.3f}" if v is not None else "undefined"
                        fill = f"rgb({int(245 - 120 * (v or 0))}, {int(248 - 75 * (v or 0))}, 230)"
                    else:
                        value = cell["relation"]
                        fill = _COLORS[value]
                x = label_width + j * cell_width
                svg.parts.append("<g>")
                if cell:
                    title = cell["scope"]
                    if association and cell["association"] is None:
                        title += " · " + str(cell["association_reason"])
                    svg.parts.append(f"<title>{_esc(title)}</title>")
                svg.rect(x, y, cell_width - 6, height - 6, fill=fill)
                svg.text(x + 12, y + 24, value, size=12)
                svg.parts.append("</g>")
            y += height
        y += 32
    return svg.finish(max(900, label_width + len(features) * cell_width + 24), y + 20)


def _joint_svg(data):
    svg = _SVG("Selected-pair joint cells")
    y = _heading(svg, data, "Selected-pair joint cells")
    for line in _wrap("Rows: " + data["columns"][0] + " · Columns: " + data["columns"][1], 100):
        svg.text(24, y, line)
        y += 20
    y += 8
    headers = [_wrap(v, 13) for v in data["b"]]
    for j, lines in enumerate(headers):
        for k, line in enumerate(lines):
            svg.text(240 + j * 120, y + k * 18, line, size=12)
    y += max((len(lines) for lines in headers), default=1) * 18 + 10
    lookup = {(c["a"], c["b"]): c for c in data["cells"]}
    maximum = max((c.get("count", 1) for c in data["cells"]), default=1)
    for i, value in enumerate(data["a"]):
        lines = _wrap(value, 25)
        height = max(44, len(lines) * 18 + 10)
        for k, line in enumerate(lines):
            svg.text(24, y + 24 + k * 18, line, size=12)
        for j in range(len(data["b"])):
            cell = lookup.get((i, j))
            full = data["detail"] == "full"
            fraction = cell["count"] / maximum if cell and full else 0
            fill = (
                f"rgb({int(235 - 140 * fraction)}, {int(245 - 85 * fraction)}, 210)"
                if cell and full
                else "#dcf3e7"
                if cell
                else "#ffffff"
            )
            svg.rect(240 + j * 120, y, 114, height - 6, fill=fill)
            svg.text(
                250 + j * 120,
                y + 24,
                cell["count"] if cell and full else "observed" if cell else "—",
                size=12,
            )
        y += height
    return svg.finish(max(900, 264 + 120 * len(data["b"])), y + 20)


def _figure(data, view, exceptions):
    kind = data["kind"]
    valid = {
        "grain": {"map", "matrix"},
        "pairs": {"mapping", "association"},
        "levels": {"bars"},
        "census": {"tree"},
        "joint_counts": {"heatmap"},
    }
    defaults = {
        "grain": "map",
        "pairs": "mapping",
        "levels": "bars",
        "census": "tree",
        "joint_counts": "heatmap",
    }
    view = view or defaults[kind]
    if view not in valid[kind]:
        raise ValueError(f"Invalid view {view!r} for {kind}")
    if view == "association" and data["detail"] != "full":
        raise ValueError("Association view requires detail='full'")
    if kind == "grain":
        return _matrix_svg(data) if view == "matrix" else _grain_svg(data, exceptions=exceptions)
    if kind in {"levels", "census"}:
        return _bars_svg(data)
    if kind == "pairs":
        return _pairs_svg(data, association=view == "association")
    return _joint_svg(data)


def render_svg(
    result: ExplorerResult | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Literal["full", "topology"] = "full",
    view: str | None = None,
    show_exceptions: bool = False,
) -> str:
    """Render a self-contained static SVG. Combined results default to grain.

    No optional packages, executables, network resources, or input frame required.
    ``view`` selects grain map/matrix, pair mapping/association, levels bars,
    census tree, or selected-pair heatmap. Quantities are filtered before drawing.
    """
    return _figure(
        visualization_data(result, section=section, detail=detail), view, show_exceptions
    )


def _evidence_html(data):
    chunks = [
        '<section class="evidence"><h2>Placement evidence</h2>',
        "<p>Select a feature in the map, matrix, or menu. Key components appear in key headings.</p>",
    ]
    nodes = {n["id"]: " / ".join(n["titles"]) for n in data["nodes"]}
    for feature in data["features"]:
        chunks.append(
            f'<details id="evidence-{feature["id"]}" data-evidence="{feature["id"]}"><summary>{_esc(feature["label"])}</summary>'
        )
        placement = "; ".join(nodes[n] for n in feature["nodes"])
        chunks.append(
            "<p>"
            + _esc(
                "Coarsest supported: " + placement
                if placement
                else "Not placed by tested keys: " + feature["reason"].replace("_", " ")
            )
            + "</p>"
        )
        chunks.append(
            '<div class="table-scroll"><table><thead><tr><th>Candidate grouping</th>'
            "<th>Behavior</th><th>Scope</th>"
        )
        if data["detail"] == "full":
            chunks.append(
                "<th>Violating / evaluated groups</th><th>Singleton / repeated groups</th>"
                "<th>Affected / evaluated rows</th>"
            )
        chunks.append("</tr></thead><tbody>")
        for row in data["evidence"]:
            if row["feature"] != feature["id"]:
                continue
            scope = "Graph population" if row["compatible"] else "Different target population"
            chunks.append(
                f"<tr><td>{_esc(row['key'])}</td><td>{row['state']}</td>"
                f"<td>{scope}<br>{_esc(row['scope'])}</td>"
            )
            if data["detail"] == "full":
                chunks.append(
                    f"<td>{row['violating_groups']} / {row['evaluated_groups']}</td>"
                    f"<td>{row['singleton_groups']} / {row['repeated_groups']}</td>"
                    f"<td>{row['affected_rows']} / {row['evaluated_rows']}</td>"
                )
            chunks.append("</tr>")
        chunks.append("</tbody></table></div></details>")
    return "".join(chunks) + "</section>"


_SCRIPT = """
(() => {
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
function selectFeature(id) {
  const select = $('#feature'); if (select) select.value = id;
  $$('[data-evidence]').forEach(e => { e.open = e.dataset.evidence === id; });
  $$('[data-feature]').forEach(e => {
    e.classList.toggle('active', e.dataset.feature === id);
    e.setAttribute('aria-pressed', String(e.dataset.feature === id));
  });
  const status = $('#selection-status'), link = $('#selection-link');
  if (status) status.textContent = id ? `Selected: ${select.selectedOptions[0].textContent}` : 'No feature selected.';
  if (link) { link.hidden = !id; link.href = '#evidence-' + id; }
  $$('[data-node]').forEach(e => e.classList.toggle('selected',
    e.dataset.features.split(' ').includes(id) && id !== ''));
}
$$('[data-feature]').forEach(e => {
  e.setAttribute('role', 'button');
  e.setAttribute('tabindex', '0');
  e.setAttribute('aria-pressed', 'false');
  e.addEventListener('click', () => selectFeature(e.dataset.feature));
  e.addEventListener('keydown', ev => {
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); selectFeature(e.dataset.feature); }
    if (ev.key === 'Escape') selectFeature('');
  });
});
$('#feature')?.addEventListener('change', e => selectFeature(e.target.value));
$('#collapse')?.addEventListener('change', e =>
  $$('[data-attributes]').forEach(s => s.hidden =
    s.dataset.attributes !== (e.target.checked ? 'collapsed' : 'expanded')));
$('#focus')?.addEventListener('change', e => {
  const id = e.target.value, keep = new Set([id]);
  $$('[data-source]').forEach(edge => {
    if (edge.dataset.source === id) keep.add(edge.dataset.target);
    if (edge.dataset.target === id) keep.add(edge.dataset.source);
  });
  $$('[data-node]').forEach(n => n.classList.toggle('dim', id !== '' && !keep.has(n.dataset.node)));
  $$('[data-source]').forEach(edge => edge.classList.toggle('dim', id !== '' &&
    !(keep.has(edge.dataset.source) && keep.has(edge.dataset.target))));
});
$('#view')?.addEventListener('change', e => {
  $$('[data-view]').forEach(v => v.hidden = v.dataset.view !== e.target.value);
  if ($('#matrix-search')) $('[data-scale]').disabled = e.target.value === 'matrix';
  ['collapse', 'exceptions', 'focus'].forEach(id => {
    const control = $('#' + id); if (control) control.disabled = e.target.value !== 'map';
  });
});
$('#context')?.addEventListener('change', e =>
  $$('[data-context]').forEach(v => v.hidden = e.target.value !== 'all' &&
    v.dataset.context !== e.target.value && v.dataset.context !== '0'));
$('#exceptions')?.addEventListener('change', e => {
  $('#map-normal').hidden = e.target.checked; $('#map-exceptions').hidden = !e.target.checked;
});
$$('[data-collapse-row]').forEach(button => button.addEventListener('click', () => {
  button.setAttribute('aria-expanded', button.getAttribute('aria-expanded') === 'true' ? 'false' : 'true');
  const closed = new Set($$('[data-collapse-row][aria-expanded="false"]').map(b => b.dataset.collapseRow));
  const hidden = new Set();
  $$('[data-tree-row]').forEach(row => {
    row.hidden = closed.has(row.dataset.parent) || hidden.has(row.dataset.parent);
    if (row.hidden) hidden.add(row.dataset.treeRow);
  });
}));
function filterMatrix() {
  const search = $('#matrix-search'); if (!search) return;
  const query = search.value.trim().toLocaleLowerCase();
  const rows = $$('[data-matrix-feature]');
  rows.forEach(row => row.hidden = !row.dataset.matrixFeature.toLocaleLowerCase().includes(query));
  const key = $('#matrix-key').value;
  $$('[data-matrix-key]').forEach(cell => cell.hidden = key !== '' && cell.dataset.matrixKey !== key);
  const shown = rows.filter(row => !row.hidden).length;
  $('#matrix-status').textContent = `${shown} of ${rows.length} features shown`;
  $('#matrix-empty').hidden = shown !== 0 || rows.length === 0;
}
$('#matrix-search')?.addEventListener('input', filterMatrix);
$('#matrix-key')?.addEventListener('change', filterMatrix);
$('#matrix-reset')?.addEventListener('click', () => {
  $('#matrix-search').value = ''; $('#matrix-key').value = ''; filterMatrix();
});
filterMatrix();
$('#reset-view')?.addEventListener('click', () => {
  $('#matrix-reset')?.click();
  $$('.toolbar select').forEach(select => {
    select.selectedIndex = 0; select.dispatchEvent(new Event('change'));
  });
  $$('.toolbar input[type=checkbox]').forEach(input => {
    input.checked = false; input.dispatchEvent(new Event('change'));
  });
  $$('[data-collapse-row]').forEach(button => button.setAttribute('aria-expanded', 'true'));
  $$('[data-tree-row]').forEach(row => row.hidden = false);
  selectFeature('');
});
})();
"""


def _tree_html(data):
    """An expandable HTML version of the same aligned-bar tree."""
    rows = data["rows"]
    parents = {r["parent"] for r in rows}
    output = ['<div class="table-scroll"><table><thead><tr><th>Observed path</th>']
    full = data["detail"] == "full"
    if full:
        output.append(
            "<th>Rows / share of total</th><th>Share of parent</th><th>Share of total</th>"
        )
    output.append("</tr></thead><tbody>")
    for row in rows:
        output.append(
            f'<tr data-tree-row="{row["id"]}" data-parent="{row["parent"] or ""}">'
            f'<td style="padding-left:{8 + row["depth"] * 20}px">'
        )
        if row["id"] in parents:
            output.append(
                f'<button data-collapse-row="{row["id"]}" aria-expanded="true" '
                f'aria-label="Toggle child branches of {_esc(row["label"])}">↕</button> '
            )
        output.append(_esc(row["label"]) + "</td>")
        if full:
            share = row["share"]
            fraction = f"{share:.1%}" if share is not None else "undefined"
            parent = row["parent_share"]
            parent_text = f"{parent:.1%}" if parent is not None else "—"
            output.append(
                f"<td>{row['count']} / {fraction}</td><td>{parent_text}</td>"
                f'<td><meter min="0" max="1" value="{share or 0}" '
                f'aria-label="Share of total: {fraction}"></meter></td>'
            )
        output.append("</tr>")
    return "".join(output) + "</tbody></table></div>"


def render_html(
    result: ExplorerResult | Mapping[str, Any],
    *,
    section: str | None = None,
    detail: Literal["full", "topology"] = "full",
) -> str:
    """Standalone interactive HTML using only filtered presentation evidence.

    Grain supports feature evidence, map/matrix switching, attribute collapse,
    key-neighborhood focus, and exact-dependency exceptions. Pair contexts remain
    separate; census branches can collapse. No remote assets or raw JSON payload.
    """
    data = visualization_data(result, section=section, detail=detail)
    kind = data["kind"]
    from .._html import document

    title = "Fieldwork / " + {
        "grain": "Observed grain",
        "pairs": "Pair relationships",
        "census": "Census",
        "levels": "Feature levels",
        "joint_counts": "Joint counts",
    }.get(kind, kind.replace("_", " ").title())
    parts = [
        (
            f'<header><p class="eyebrow">Saved evidence report</p><h1>{_esc(title)}</h1>'
            "<p>Explore saved relationships and populations. Controls change the view, "
            "not the analysis.</p></header>"
        ),
        (
            '<div class="toolbar" data-enhance hidden><label>Figure scale<select data-scale>'
            '<option value="actual">Actual size</option><option value="fit">Fit width</option>'
            '<option value="0.5">50%</option><option value="0.75">75%</option>'
            '<option value="1.5">150%</option><option value="2">200%</option></select></label>'
            '<button type="button" id="reset-view">Reset view</button></div>'
        ),
        (
            '<p class="muted">Large figures and tables scroll within their panels. '
            "Use Fit width for an overview, or Actual size to read labels.</p>"
        ),
    ]
    if detail == "topology":
        parts.append("<p>Topology only. Quantitative evidence was removed before export.</p>")
    if kind == "grain":
        parts.append(
            '<div class="toolbar" data-enhance hidden><label>Feature <select id="feature">'
            '<option value="">Select a feature</option>'
        )
        for feature in data["features"]:
            parts.append(f'<option value="{feature["id"]}">{_esc(feature["label"])}</option>')
        parts.append(
            '</select></label><label>Focus <select id="focus"><option value="">All keys</option>'
        )
        for node in data["nodes"]:
            parts.append(
                f'<option value="{node["id"]}">{_esc(" / ".join(node["titles"]))}</option>'
            )
        parts.append(
            '</select></label><label>View <select id="view">'
            '<option value="map">Map</option><option value="matrix">Matrix</option>'
            '</select></label><label><input type="checkbox" id="collapse">Collapse attributes</label>'
            '<label><input type="checkbox" id="exceptions">Show varying features</label></div>'
            "<p>Focus highlights a key and its neighbors; other connections remain visible.</p>"
            '<p id="selection-status" role="status" aria-live="polite">No feature selected.</p>'
            '<a id="selection-link" href="#" hidden>View selected evidence</a>'
        )
        for view in ("map", "matrix"):
            parts.append(f'<div data-view="{view}" {"hidden" if view == "matrix" else ""}>')
            if view == "matrix":
                parts.append(_matrix_html(data) + "</div>")
                continue
            for exceptions in (False, True):
                marker = (
                    ('id="map-exceptions" hidden' if exceptions else 'id="map-normal"')
                    if view == "map"
                    else ""
                )
                parts.append(f"<div {marker}>")
                for collapsed in (False, True) if view == "map" else (False,):
                    figure = (
                        _grain_svg(data, exceptions=exceptions, collapsed=collapsed)
                        if view == "map"
                        else _matrix_svg(data)
                    )
                    for node in data["nodes"]:
                        assigned = " ".join(
                            f["id"] for f in data["features"] if node["id"] in f["nodes"]
                        )
                        figure = figure.replace(
                            f'data-node="{node["id"]}"',
                            f'data-node="{node["id"]}" data-features="{assigned}"',
                        )
                    figure = figure.replace('role="img"', 'role="group"')
                    suffix = f"{view}-{exceptions}-{collapsed}"
                    figure = figure.replace('id="arrow"', f'id="arrow-{suffix}"').replace(
                        "url(#arrow)", f"url(#arrow-{suffix})"
                    )
                    state = "collapsed" if collapsed else "expanded"
                    attrs = f'data-attributes="{state}"' if view == "map" else ""
                    parts.append(
                        f'<div class="figure actual" {attrs} '
                        f"{'hidden' if collapsed else ''}>{figure}</div>"
                    )
                parts.append("</div>")
            parts.append("</div>")
        parts.append(_evidence_html(data))
    elif kind == "pairs":
        parts.append(
            '<div class="toolbar" data-enhance hidden><label>Context <select id="context">'
            '<option value="all">All contexts</option>'
        )
        for i, context in enumerate(data["contexts"]):
            parts.append(f'<option value="{i}">{_esc(context["label"])}</option>')
        parts.append("</select></label>")
        views = ["mapping"] + (["association"] if detail == "full" else [])
        if detail == "full":
            parts.append(
                '<label>Encoding <select id="view"><option value="mapping">Mapping</option>'
                '<option value="association">Cramér’s V</option></select></label>'
            )
        parts.append(
            "</div><p>Global and selected contexts use the same feature order. "
            "Hover over a cell to inspect its population.</p>"
        )
        for view in views:
            parts.append(f'<div data-view="{view}" {"hidden" if view == "association" else ""}>')
            for i, context in enumerate(data["contexts"]):
                parts.append(
                    f'<div class="figure actual" data-context="{i}">'
                    + _figure({**data, "contexts": [context]}, view, False)
                    + "</div>"
                )
            parts.append("</div>")
    elif kind == "census":
        parts.append("<p>" + _esc(data["scope"]) + "</p><p>" + _esc(data["caption"]) + "</p>")
        parts.append(_tree_html(data))
        parts.append(
            '<details><summary>Static figure</summary><div class="figure actual">'
            + _figure(data, None, False)
            + "</div></details>"
        )
    else:
        parts.append('<div class="figure actual">' + _figure(data, None, False) + "</div>")
    return document(title, "".join(parts), _SCRIPT)
