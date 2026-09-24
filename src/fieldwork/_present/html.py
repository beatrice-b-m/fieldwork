"""Standalone offline HTML documents of projections: one page builder per kind family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from .common import esc
from .common import esc as _esc
from .project import candidate_explanation, dependency_label, grain_title
from .svg import COLORS, figure, grain_map, matrix_keys
from .text import comparison_labels, coverage_lines, sample_label, skipped_label, unit_label

STYLE = """
:root{color-scheme:light;--ink:#193345;--muted:#4c6474;--line:#cbd5df;--accent:#006880}
*{box-sizing:border-box}body{margin:0;background:#f6f8fb;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:28px}h1{font-size:28px;margin:8px 0}h2{font-size:21px}h3{font-size:17px}
p{margin:10px 0}p,li,h1,h2,h3{overflow-wrap:anywhere}a{color:var(--accent)}.eyebrow,.muted{color:var(--muted)}
.eyebrow{font-size:13px;letter-spacing:.08em;text-transform:uppercase}
.toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:16px 0}
.toolbar label{min-width:0;max-width:100%;display:flex;flex-direction:column;gap:4px;font-size:14px}
.toolbar label:has(input[type=checkbox]){flex-direction:row;align-items:center}
input,select,button{font:inherit;color:inherit;max-width:100%}
input[type=search],select,button{padding:8px 12px;border:1px solid #8b9eac;border-radius:6px;background:white;min-height:42px}
button,summary{cursor:pointer}button:hover{background:#e8f3f5}
:focus-visible{outline:3px solid #087c9a;outline-offset:3px}
.notice,.empty{padding:12px 16px;border-left:4px solid #087c9a;background:#e9f3f6;margin:16px 0}
.badge{display:inline-block;padding:2px 8px;border:1px solid var(--line);border-radius:12px;font-size:13px;font-weight:400;margin-right:8px}
section{margin:28px 0;min-width:0}details{background:white;border:1px solid var(--line);border-radius:8px;margin:10px 0}
summary{padding:14px;font-weight:600;overflow-wrap:anywhere}details[open]>summary{border-bottom:1px solid var(--line)}
.content{padding:4px 16px 16px;overflow-wrap:anywhere}details:target{border-color:var(--accent)}
.figure,.table-scroll{overflow:auto;max-width:100%;margin:16px 0}
.figure{border:1px solid var(--line);border-radius:10px}
.figure svg{display:block;max-width:100%;height:auto}.figure.actual svg{max-width:none}
table{border-collapse:collapse;background:white;width:100%;font-size:14px}
.table-scroll>table:not(.evidence-table){min-width:680px}
th,td{border:1px solid var(--line);padding:8px 12px;text-align:left;vertical-align:top}
th{font-weight:600;background:#f1f5f8}td{overflow-wrap:anywhere}td table{margin:4px 0}
code{font:14px/1.6 ui-monospace,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.feature{cursor:pointer}.selected rect{stroke:#087c9a;stroke-width:3}.dim{opacity:.3}.feature.active{fill:#006880;font-weight:bold}
.matrix-scroll{overflow:auto;max-height:65vh;border:1px solid var(--line);border-radius:8px;isolation:isolate}
.grain-matrix{width:max-content;min-width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed}
.grain-matrix caption{text-align:left;padding:12px;background:#f1f5f8;font-weight:600}
.grain-matrix th,.grain-matrix td{width:160px;min-width:160px;max-width:160px;padding:8px;border-width:0 1px 1px 0}
.grain-matrix tr>th:first-child{width:240px;min-width:240px;max-width:240px;position:sticky;left:0;z-index:1}
.grain-matrix thead{position:sticky;top:0;z-index:3}
.grain-matrix thead th{position:sticky;top:0;z-index:2;background:#e8eff4}
.grain-matrix thead th:first-child{z-index:3}
.grain-matrix button{width:100%;border:0;background:transparent;text-align:left;padding:4px;font-weight:inherit}
.grain-matrix .active{outline:2px solid var(--accent);outline-offset:-2px}
.matrix-label{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;overflow-wrap:anywhere}
@media(max-width:600px){.grain-matrix tr>th:first-child{width:140px;min-width:140px;max-width:140px}}
[hidden]{display:none!important}.skip{position:absolute;left:-10000px}.skip:focus{left:16px;top:8px;background:white;padding:8px}
@media(max-width:600px){main{padding:16px}h1{font-size:24px}.toolbar label{width:100%}
.evidence-table,.evidence-table>tbody,.evidence-table>tbody>tr,.evidence-table>tbody>tr>th,.evidence-table>tbody>tr>td{display:block;width:100%}
.evidence-table>tbody>tr>td{border-top:0;margin-bottom:8px}.evidence-table .table-scroll{margin:0}}
@media print{body{background:white}main{max-width:none;padding:0}.toolbar,.skip{display:none}.figure{overflow:visible}}
"""


SCRIPT = """
(() => {
  const all = (s, root = document) => [...root.querySelectorAll(s)];
  all('[data-enhance]').forEach(e => e.hidden = false);
  all('[data-collection]').forEach(section => {
    const cards = all('[data-record]', section);
    const search = section.querySelector('[data-search]');
    if (!search) return;
    const pattern = section.querySelector('[data-pattern-filter]');
    const exceptions = section.querySelector('[data-exceptions-filter]');
    // Cache plain text once. Queries never become HTML or CSS selectors.
    const text = new Map(cards.map(c => [c, c.textContent.toLocaleLowerCase()]));
    const update = () => {
      const query = search.value.trim().toLocaleLowerCase();
      cards.forEach(card => card.hidden = !text.get(card).includes(query) ||
        (pattern && pattern.value !== '' && card.dataset.pattern !== pattern.value) ||
        (exceptions?.checked && card.dataset.exceptions !== 'true'));
      const visible = cards.filter(c => !c.hidden).length;
      section.querySelector('[data-status]').textContent =
        `${visible} of ${cards.length} included records shown`;
      section.querySelector('[data-no-matches]').hidden = visible !== 0 || cards.length === 0;
    };
    section.addEventListener('input', update);
    section.addEventListener('change', update);
    section.addEventListener('click', event => {
      const button = event.target.closest('button');
      if (!button) return;
      if (button.hasAttribute('data-reset')) {
        search.value = ''; if (pattern) pattern.value = '';
        if (exceptions) exceptions.checked = false;
        update();
      }
      if (button.hasAttribute('data-expand')) cards.filter(c => !c.hidden).forEach(c => c.open = true);
      if (button.hasAttribute('data-close')) cards.filter(c => !c.hidden).forEach(c => c.open = false);
    });
    update();
  });
  function revealHash() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target) return;
    const collection = target.closest('[data-collection]');
    collection?.querySelector('[data-reset]')?.click();
    for (let parent = target; parent; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS') parent.open = true;
    }
    target.querySelector('summary')?.focus({preventScroll: true});
    target.scrollIntoView({block: 'start'});
  }
  window.addEventListener('hashchange', revealHash);
  document.addEventListener('click', event => {
    const anchor = event.target.closest('a[href^="#"]');
    if (anchor && anchor.hash === location.hash) revealHash();
  });
  if (location.hash) revealHash();
  document.querySelector('[data-scale]')?.addEventListener('change', event => {
    const scale = event.target.value;
    all('.figure').forEach(figure => {
      figure.classList.toggle('actual', scale !== 'fit');
      const svg = figure.querySelector('svg');
      if (svg) svg.style.width = scale === 'fit' || scale === 'actual' ? '' :
        `${Number(svg.getAttribute('width')) * Number(scale)}px`;
    });
  });
})();
"""


def document(title: str, body: str, script: str = "") -> str:
    body = body.replace(
        'class="figure', 'tabindex="0" role="region" aria-label="Scrollable figure" class="figure'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{STYLE}</style></head><body>"
        '<a class="skip" href="#report">Skip to report</a><main id="report">'
        + body
        + "</main><script>"
        + SCRIPT
        + script
        + "</script></body></html>"
    )


def collection(
    title: str,
    cards: list[str],
    total: int,
    *,
    full: bool,
    patterns: list[str] | None = None,
    exceptions: bool = False,
) -> str:
    """Bounded record collection with progressive enhancement and honest limits."""
    shown = len(cards)
    parts = [f"<section data-collection><h2>{_esc(title)}</h2>"]
    if shown < total:
        label = f"{shown} of {total} records included. " if full else "More evidence available. "
        parts.append(
            '<p class="notice">' + label + "Display limit reached; "
            "increase max_findings when rendering to include more. "
            "Search only covers included records.</p>"
        )
    if cards:
        parts.append(
            '<div class="toolbar" data-enhance hidden>'
            '<label>Search included records<input type="search" data-search '
            'placeholder="Feature, finding, or evidence text"></label>'
        )
        if patterns:
            parts.append(
                '<label>Pattern<select data-pattern-filter><option value="">All patterns</option>'
            )
            parts.extend(
                f'<option value="{_esc(p)}">{_esc(p.replace("_", " "))}</option>' for p in patterns
            )
            parts.append("</select></label>")
        if exceptions:
            parts.append(
                '<label><input type="checkbox" data-exceptions-filter>With exception rows</label>'
            )
        parts.append(
            '<button type="button" data-reset>Clear filters</button>'
            '<button type="button" data-expand>Expand shown</button>'
            '<button type="button" data-close>Collapse shown</button></div>'
            '<p class="muted" data-status role="status" aria-live="polite"></p>'
            '<p class="empty" data-no-matches hidden>No included records match these filters. '
            "Clear filters to see the included evidence.</p>"
        )
    elif total == 0:
        parts.append(
            '<p class="empty">No records were saved for this section. '
            "Review search coverage and analysis settings before concluding there is no pattern.</p>"
        )
    return "".join(parts + cards) + "</section>"


FIGURE_SCRIPT = """
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

_FIGURE_TITLES = {
    "grain": "Observed grain",
    "pairs": "Pair relationships",
    "census": "Census",
    "levels": "Feature levels",
    "joint_counts": "Joint counts",
}


def page(projection: Mapping[str, Any], *, max_findings: int) -> str:
    """The standalone HTML document for a projection."""
    if projection["kind"] in _FIGURE_TITLES:
        return _figure_page(projection)
    return _evidence_page(projection, max_findings)


# Figure pages: grain, pairs, census, levels, joint counts.


def _figure_page(projection: Mapping[str, Any]) -> str:
    kind = projection["kind"]
    title = "Fieldwork / " + _FIGURE_TITLES[kind]
    parts = [
        (
            f'<header><p class="eyebrow">Saved evidence report</p><h1>{esc(title)}</h1>'
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
    if projection["detail"] == "topology":
        parts.append("<p>Topology only. Quantitative evidence was removed before export.</p>")
    body = {"grain": _grain_body, "pairs": _pairs_body, "census": _census_body}
    parts.append(body.get(kind, _plain_body)(projection))
    return document(title, "".join(parts), FIGURE_SCRIPT)


def _plain_body(projection: Mapping[str, Any]) -> str:
    return '<div class="figure actual">' + figure(projection) + "</div>"


def _grain_body(projection: Mapping[str, Any]) -> str:
    parts = [
        (
            '<div class="toolbar" data-enhance hidden><label>Feature <select id="feature">'
            '<option value="">Select a feature</option>'
        )
    ]
    parts += [
        f'<option value="{f["id"]}">{esc(f["label"])}</option>' for f in projection["features"]
    ]
    parts.append(
        '</select></label><label>Focus <select id="focus"><option value="">All keys</option>'
    )
    parts += [
        f'<option value="{n["id"]}">{esc(" / ".join(n["titles"]))}</option>'
        for n in projection["nodes"]
    ]
    parts.append(
        '</select></label><label>View <select id="view">'
        '<option value="map">Map</option><option value="matrix">Matrix</option>'
        '</select></label><label><input type="checkbox" id="collapse">Collapse attributes</label>'
        '<label><input type="checkbox" id="exceptions">Show varying features</label></div>'
        "<p>Focus highlights a key and its neighbors; other connections remain visible.</p>"
        '<p id="selection-status" role="status" aria-live="polite">No feature selected.</p>'
        '<a id="selection-link" href="#" hidden>View selected evidence</a><div data-view="map">'
    )
    for exceptions in (False, True):
        marker = 'id="map-exceptions" hidden' if exceptions else 'id="map-normal"'
        parts.append(f"<div {marker}>")
        for collapsed in (False, True):
            svg = grain_map(
                projection,
                exceptions=exceptions,
                collapsed=collapsed,
                role="group",
                marker=f"arrow-map-{exceptions}-{collapsed}",
            )
            state = "collapsed" if collapsed else "expanded"
            parts.append(
                f'<div class="figure actual" data-attributes="{state}" '
                f"{'hidden' if collapsed else ''}>{svg}</div>"
            )
        parts.append("</div>")
    parts.append('</div><div data-view="matrix" hidden>' + _matrix(projection) + "</div>")
    return "".join(parts) + _placements(projection)


def _matrix(projection: Mapping[str, Any]) -> str:
    """Wide grain evidence stays readable as a table instead of shrinking text."""
    keys = matrix_keys(projection)
    lookup = {(r["key"], r["feature"]): r for r in projection["evidence"]}
    parts = [
        (
            "<h2>Feature behavior by candidate key</h2>"
            "<p>Each row is a feature; each column is a tested grouping. "
            "Constant means one observed value per group; varying means conflicting values. "
            "Undefined means no evaluated support; untested means no saved test. "
            "* marks a different target population. Select a cell, then View selected evidence. "
            "Long labels are shortened visually; full names appear in placement evidence.</p>"
            '<div class="toolbar" data-enhance hidden><label>Find matrix features'
            '<input type="search" id="matrix-search" placeholder="Feature name"></label>'
            '<label>Candidate key<select id="matrix-key"><option value="">All candidate keys</option>'
        )
    ]
    parts += [f'<option value="{index}">{esc(key)}</option>' for index, key in enumerate(keys)]
    parts.append(
        '</select></label><button type="button" id="matrix-reset">Clear matrix filters</button>'
        '</div><p id="matrix-status" role="status" aria-live="polite"></p>'
        '<p id="matrix-empty" class="empty" hidden>No features match this search.</p>'
        '<div class="matrix-scroll" tabindex="0" role="region" aria-label="Grain evidence matrix">'
        '<table class="grain-matrix"><caption>Feature behavior by candidate key</caption>'
        '<thead><tr><th scope="col">Feature</th>'
    )
    parts += [
        f'<th scope="col" data-matrix-key="{index}" title="{esc(key)}">'
        f'<span class="matrix-label">{esc(key)}</span></th>'
        for index, key in enumerate(keys)
    ]
    parts.append("</tr></thead><tbody>")
    for feature in projection["features"]:
        name = esc(feature["label"])
        parts.append(
            f'<tr data-matrix-feature="{name}"><th scope="row">'
            f'<button type="button" data-feature="{feature["id"]}" class="feature" '
            f'title="{name}"><span class="matrix-label">{name}</span></button></th>'
        )
        for index, key in enumerate(keys):
            record = lookup.get((key, feature["id"]))
            state = record["state"] if record else "untested"
            different = bool(record and not record["compatible"])
            description = f"{feature['label']} by {key}: {state}"
            description += "; different target population" if different else ""
            parts.append(
                f'<td data-matrix-key="{index}" style="background:{COLORS[state]}">'
                f'<button type="button" data-feature="{feature["id"]}" class="feature" '
                f'aria-label="{esc(description)}">{state}{" *" if different else ""}</button></td>'
            )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    if not keys or not projection["features"]:
        parts.append(
            '<p class="empty">No candidate/feature cells are available in this saved result.</p>'
        )
    return "".join(parts)


def _placements(projection: Mapping[str, Any]) -> str:
    full = projection["detail"] == "full"
    nodes = {n["id"]: " / ".join(n["titles"]) for n in projection["nodes"]}
    chunks = [
        '<section class="evidence"><h2>Placement evidence</h2>',
        "<p>Select a feature in the map, matrix, or menu. Key components appear in key headings.</p>",
    ]
    for feature in projection["features"]:
        placement = "; ".join(nodes[n] for n in feature["nodes"])
        text = (
            "Coarsest supported: " + placement
            if placement
            else "Not placed by tested keys: " + feature["reason"].replace("_", " ")
        )
        chunks.append(
            f'<details id="evidence-{feature["id"]}" data-evidence="{feature["id"]}">'
            f"<summary>{esc(feature['label'])}</summary><p>{esc(text)}</p>"
            '<div class="table-scroll"><table><thead><tr><th>Candidate grouping</th>'
            "<th>Behavior</th><th>Scope</th>"
        )
        if full:
            chunks.append(
                "<th>Violating / evaluated groups</th><th>Singleton / repeated groups</th>"
                "<th>Affected / evaluated rows</th>"
            )
        chunks.append("</tr></thead><tbody>")
        for row in (r for r in projection["evidence"] if r["feature"] == feature["id"]):
            population = "Graph population" if row["compatible"] else "Different target population"
            chunks.append(
                f"<tr><td>{esc(row['key'])}</td><td>{row['state']}</td>"
                f"<td>{population}<br>{esc(row['scope'])}</td>"
            )
            if full:
                chunks.append(
                    f"<td>{row['violating_groups']} / {row['evaluated_groups']}</td>"
                    f"<td>{row['singleton_groups']} / {row['repeated_groups']}</td>"
                    f"<td>{row['affected_rows']} / {row['evaluated_rows']}</td>"
                )
            chunks.append("</tr>")
        chunks.append("</tbody></table></div></details>")
    return "".join(chunks) + "</section>"


def _pairs_body(projection: Mapping[str, Any]) -> str:
    full = projection["detail"] == "full"
    parts = [
        (
            '<div class="toolbar" data-enhance hidden><label>Context <select id="context">'
            '<option value="all">All contexts</option>'
        )
    ]
    parts += [
        f'<option value="{i}">{esc(context["label"])}</option>'
        for i, context in enumerate(projection["contexts"])
    ]
    parts.append("</select></label>")
    if full:
        parts.append(
            '<label>Encoding <select id="view"><option value="mapping">Mapping</option>'
            '<option value="association">Cramér’s V</option></select></label>'
        )
    parts.append(
        "</div><p>Global and selected contexts use the same feature order. "
        "Hover over a cell to inspect its population.</p>"
    )
    for view in ["mapping", *(["association"] if full else [])]:
        parts.append(f'<div data-view="{view}" {"hidden" if view == "association" else ""}>')
        for i, context in enumerate(projection["contexts"]):
            svg = figure({**projection, "contexts": [context]}, view)
            parts.append(f'<div class="figure actual" data-context="{i}">{svg}</div>')
        parts.append("</div>")
    return "".join(parts)


def _census_body(projection: Mapping[str, Any]) -> str:
    """An expandable table of the census tree, with the static figure."""
    full = projection["detail"] == "full"
    rows = projection["rows"]
    parents = {r["parent"] for r in rows}
    output = [
        f"<p>{esc(projection['scope'])}</p><p>{esc(projection['caption'])}</p>",
        '<div class="table-scroll"><table><thead><tr><th>Observed path</th>',
    ]
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
                f'aria-label="Toggle child branches of {esc(row["label"])}">↕</button> '
            )
        output.append(esc(row["label"]) + "</td>")
        if full:
            share, parent = row["share"], row["parent_share"]
            fraction = f"{share:.1%}" if share is not None else "undefined"
            output.append(
                f"<td>{row['count']} / {fraction}</td>"
                f"<td>{f'{parent:.1%}' if parent is not None else '—'}</td>"
                f'<td><meter min="0" max="1" value="{share or 0}" '
                f'aria-label="Share of total: {fraction}"></meter></td>'
            )
        output.append("</tr>")
    output.append("</tbody></table></div>")
    output.append(
        '<details><summary>Static figure</summary><div class="figure actual">'
        + figure(projection)
        + "</div></details>"
    )
    return "".join(output)


# Evidence pages: findings, candidates, tests and feature connections.


def _evidence_page(projection: Mapping[str, Any], max_findings: int) -> str:
    kind, full = projection["kind"], projection["detail"] == "full"
    title = "Fieldwork / " + kind.replace("_", " ").title()
    parts = [
        (
            f'<header><p class="eyebrow">Saved evidence report</p><h1>{esc(title)}</h1>'
            "<p>Explore the saved analysis below. Controls filter this report; "
            "they do not rerun the analysis or retrieve source data.</p></header>"
        )
    ]
    if not full:
        parts.append(
            '<p class="notice">Topology only. Quantitative evidence and source positions '
            "were removed before export. Structural labels remain.</p>"
        )
    parts.append(_context(projection, full))
    if full:
        parts.append(_coverage(projection))
    parts.append(
        '<details><summary>Visual summary</summary><div class="figure">'
        + figure(projection, max_findings=min(12, max_findings))
        + '</div><p class="content">The visual summary has its own display limit of '
        "up to 12 items per list. Browse the included evidence below.</p></details>"
    )
    if "feature_network" in projection:
        parts.append(_connections(projection, max_findings, full))
    if full or "candidates" in projection:
        parts.append(_candidates_and_tests(projection, max_findings, full))
    if kind == "schema_proposal":
        parts.append(_proposals(projection, max_findings))
    else:
        parts.append(_finding_cards(projection, max_findings, full))
    return document(title, "".join(parts))


def _context(projection: Mapping[str, Any], full: bool) -> str:
    parts = []
    if projection["kind"] == "comparison":
        parts += ["<p>" + esc(text) + "</p>" for text in comparison_labels(projection)]
    else:
        if full and "scope" in projection:
            scope = projection["scope"]
            parts.append(
                "<p><strong>Population:</strong> "
                + esc(scope.get("name", "input"))
                + " · "
                + esc(scope.get("evaluated_rows", "unavailable"))
                + " evaluated source rows</p>"
            )
        if "analysis_unit" in projection:
            parts.append(
                "<p><strong>Analysis:</strong> "
                + esc(unit_label(projection["analysis_unit"]))
                + "</p>"
            )
    if projection.get("section_selection", {}).get("omitted"):
        omitted = ", ".join(projection["section_selection"]["omitted"])
        parts.append('<p class="notice">Not requested: ' + esc(omitted) + "</p>")
    if projection.get("skipped_features"):
        parts.append(
            '<p class="notice">' + esc(skipped_label(projection["skipped_features"])) + "</p>"
        )
    return "".join(parts)


def _coverage(projection: Mapping[str, Any]) -> str:
    coverage = coverage_lines(projection)
    parts = [
        (
            '<details><summary>Search coverage and limits</summary><div class="content">'
            "<p>Search limits restrict what was tested or retained. Untested work is unknown. "
            "Display limits below only restrict this report.</p>"
        )
    ]
    parts += ["<p>" + esc(line) + "</p>" for line in coverage]
    saved = projection.get("section_coverage", projection.get("coverage"))
    parts.append(
        evidence_table(saved)
        if saved is not None
        else "<p>Search coverage is unavailable in this saved result.</p>"
    )
    parts.append("</div></details>")
    if any("(limited)" in line or "budget" in line for line in coverage):
        parts.append(
            '<p class="notice">Some search or retention limits were reached. '
            "Open Search coverage and limits before interpreting absent findings.</p>"
        )
    return "".join(parts)


def _connections(projection: Mapping[str, Any], max_findings: int, full: bool) -> str:
    network = projection["feature_network"]
    included = {f["id"] for f in projection["findings"][:max_findings]} if full else set()
    cards = []
    for node in network["nodes"]:
        card = [
            "<details data-record><summary>"
            + esc(node["column"])
            + '</summary><div class="content"><ul>'
        ]
        for edge in (e for e in network["relationships"] if node in e["features"]):
            text = (
                edge["kind"].replace("_", " ")
                + ": "
                + ", ".join(f["column"] for f in edge["features"])
            )
            if edge.get("determinant"):
                text += " (" + ", ".join(edge["determinant"]) + " → " + edge["target"] + ")"
            if edge.get("analysis_unit"):
                text += " · " + unit_label(edge["analysis_unit"])
            card.append("<li>" + esc(text))
            if full:
                finding_id = edge["evidence"]["overview_finding_id"]
                card.append(
                    ' · <a href="#' + esc(quote(finding_id, safe="")) + '">inspect evidence</a>'
                    if finding_id in included
                    else " · evidence outside display limit"
                )
            if edge.get("structure"):
                card.append(evidence_table(edge["structure"]))
            card.append("</li>")
        cards.append("".join(card) + "</ul></div></details>")
    return (
        '<details><summary>Browse feature connections</summary><div class="content">'
        "<p>Connections are leads to inspect, not proof of equivalence or causation.</p>"
        + collection("Feature connections", cards, len(cards), full=full)
        + "</div></details>"
    )


def _candidates_and_tests(projection: Mapping[str, Any], max_findings: int, full: bool) -> str:
    parts = []
    candidates = projection.get("candidates", projection.get("overview", {}).get("grains", []))
    if candidates:
        cards = [
            f"<details data-record><summary>{esc(grain_title(c))} "
            f'<span class="badge">{esc(c["role"])}</span></summary><div class="content">'
            + (
                f"<p>{esc(candidate_explanation(c))}</p>{evidence_table(c)}"
                if full
                else "<p>Structural role only; support counts were removed.</p>"
            )
            + "</div></details>"
            for c in candidates[:max_findings]
        ]
        parts.append(collection("Candidate grains", cards, len(candidates), full=full))
    if "dependencies" in projection:
        tests = projection["dependencies"]
        cards = [
            f"<details data-record><summary>{esc(dependency_label(row))}</summary>"
            f'<div class="content"><p>{esc(row["explanation"])}</p>'
            + evidence_table({k: v for k, v in row.items() if k != "explanation"})
            + "</div></details>"
            for row in tests[:max_findings]
        ]
        parts.append(
            collection(
                "Completed dependency tests (including below finding threshold)",
                cards,
                len(tests),
                full=True,
            )
        )
    return "".join(parts)


def _proposals(projection: Mapping[str, Any], max_findings: int) -> str:
    cards = [
        f"<details data-record><summary>{esc(p['label'])} "
        f'<span class="badge">{esc(p["role"])}</span></summary><div class="content">'
        + evidence_table({r["code"].lower(): r["value"] for r in p["reasons"]})
        + f"<p>Dependency evidence: {esc(p['fd_evidence'])}</p></div></details>"
        for p in projection["proposals"][:max_findings]
    ]
    return collection("Role proposals", cards, len(projection["proposals"]), full=True)


def _finding_cards(projection: Mapping[str, Any], max_findings: int, full: bool) -> str:
    rows = projection["findings"]
    cards = [_finding_card(row, projection["kind"], full) for row in rows[:max_findings]]
    return collection(
        "Inspect findings",
        cards,
        len(rows),
        full=full,
        patterns=sorted({r["pattern"] for r in rows[:max_findings]}),
        exceptions=full and projection["kind"] != "comparison",
    )


def _finding_card(row: Mapping[str, Any], kind: str, full: bool) -> str:
    has_exceptions = full and bool(row["exceptions"].get("total", 0))
    card = [
        f'<details data-record data-pattern="{esc(row["pattern"])}"'
        + (
            f' data-exceptions="{str(has_exceptions).lower()}" id="{esc(row["id"])}"'
            if full
            else ""
        )
        + '><summary><span class="badge">'
        + esc(row["pattern"].replace("_", " "))
        + "</span>"
        + esc(row["statement"])
        + (
            ' <span class="badge">' + esc(row["lead"]["reason"]) + "</span>"
            if "lead" in row
            else ""
        )
        + '</summary><div class="content">'
        + "<p>Analysis: "
        + esc(unit_label(row["analysis_unit"]))
        + "</p>"
    ]
    if full:
        card.append(f"<p>Finding {esc(row['id'])} · counting unit: {esc(row['counting_unit'])}</p>")
        if "explanation" in row:
            card.append("<p>" + esc(row["explanation"]) + "</p>")
        card.append(evidence_table(row["measurements"]))
        card.append(_comparison_note() if kind == "comparison" else _source_rows(row))
    if row.get("structure"):
        card.append(evidence_table(row["structure"]))
    return "".join(card) + "</div></details>"


def _comparison_note() -> str:
    return (
        "<p>Change is after minus before. A fraction change of 0.25 means "
        "25 percentage points. Compare the two populations and denominators "
        "before interpreting a change as improvement. Inspect the original "
        "before/after results for source rows.</p>"
    )


def _source_rows(row: Mapping[str, Any]) -> str:
    return (
        "<h3>Representative source rows</h3>"
        "<p>Positions are zero-based offsets in the original ordered source, "
        "not dataframe index labels. Saved samples are the first matches in source "
        "order; their size is not the total support.</p>"
        + "".join(
            "<p>" + esc(sample_label(key.title(), row[key])) + "</p>"
            for key in ("examples", "exceptions")
        )
        + "<p>In Python, with this result named <code>result</code> and its identical "
        "ordered source named <code>df</code>:</p><p><code>"
        + esc(f"result.inspect(df, {row['id']!r})")
        + "</code></p><p>Use <code>all_matches=True</code> to retrieve all "
        "matching rows and <code>exceptions=True</code> for exception rows. "
        "These operations require the original source and a Python session.</p>"
    )


def evidence_table(value: Any) -> str:
    """Saved evidence as nested tables and lists, not serialized dictionaries."""
    if isinstance(value, dict):
        return (
            '<div class="table-scroll"><table class="evidence-table">'
            + "".join(
                f'<tr><th scope="row">{esc(str(key).replace("_", " "))}</th>'
                f"<td>{evidence_table(item)}</td></tr>"
                for key, item in value.items()
            )
            + "</table></div>"
        )
    if isinstance(value, list):
        if any(isinstance(item, (dict, list)) for item in value):
            return (
                "<ul>"
                + "".join("<li>" + evidence_table(item) + "</li>" for item in value)
                + "</ul>"
            )
        return esc(", ".join(str(item) for item in value) or "none")
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return esc("unavailable / not defined" if value is None else str(value))
