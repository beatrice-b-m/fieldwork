"""Shared, offline HTML presentation primitives; no analytical data access."""

from __future__ import annotations

from ._explore.graphics import _esc

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
