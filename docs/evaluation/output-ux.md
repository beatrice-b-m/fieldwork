# Output UX review and qualification

Reviewed the package's plaintext, SVG, standalone HTML, saved-result inspection,
notebook examples, and display/search boundaries. Implemented changes are described
in [Exploring saved outputs](../output-ux.md). Analysis algorithms, source identity,
selection semantics, required dependencies, and public call signatures are unchanged.

## Review findings and implemented changes

| Finding | Change |
| --- | --- |
| Wide grain matrices placed every feature in an SVG column; scaling made text unreadable | Transposed both matrix formats; native HTML matrix with sticky headings, bounded scrolling, feature search, and candidate filter |
| Discovery records required manually opening many panels | Independent list search, pattern/exception filters, expand/collapse, and visible counts |
| Feature links could land on closed or filtered-out evidence | Reveal, focus, and scroll to the target; support direct fragment links and reload |
| Bounded examples looked like complete supporting populations | Explicit saved/total source positions, source-order sampling explanation, Python inspection handoff |
| Discovery search coverage was not surfaced in rendered outputs | Separate search/retention coverage from display limits, retain section-specific overview coverage |
| Empty, filtered, and display-suppressed reports looked alike | Distinct states with the appropriate next action |
| Candidate omissions could be silent and dependency SVG omissions counted findings instead of tests | Per-list omission notices for the lists actually rendered |
| Long controls overflowed enlarged narrow layouts | Constrain and wrap toolbar controls; stack key/value evidence on narrow screens |
| Long joint-count axis names extended outside the SVG | Wrap axis captions and allocate their height |
| Foundation selection had weak feedback and map controls stayed enabled in matrix mode | Live selection status, linked placement evidence, reset, keyboard support, relevant-control disabling |
| Empty HTML collections triggered script errors | Enhance only collections that contain controls |

## Qualification

The repeatable browser harness is `scripts/check_output_ux.cjs`; fixture generation
is `scripts/output_ux_fixtures.py`. The checked-in
[browser results](output-ux-browser.json) record Chromium version and cases.

- 16 synthetic reports, including 20,000 rows × 66 columns, 18 candidate keys,
  the 960-row laboratory example, long unbroken/Unicode/markup-like labels,
  duplicate indexes, deep census trees, sparse populations, pair contexts,
  comparisons, empty results, zero display limits, and topology output.
- 80 report/viewport combinations: widths 375, 600, 768, 1280, and 1920 CSS pixels.
- SVG Fit width and fixed scales from 50% to 200%; CSS layout zoom at 75%, 125%,
  and 200%. Layout zoom exercises reflow; it is not a browser-chrome zoom test.
- Matrix view is exercised separately at every width and layout zoom. Tests check
  sticky heading geometry and painting, search, candidate filtering, feature
  selection, and reset. Matrices with many keys retain local horizontal scrolling.
- Checks also cover document overflow, SVG label bounds, duplicate IDs, keyboard
  selection, filtered evidence links and reload, census collapse/reset, pair
  controls, JavaScript errors, unexpected network requests, and JavaScript-disabled
  discovery disclosures.
- Result: no document overflow, clipped SVG text, script errors, or external
  requests in the exercised cases. Screenshots were inspected, including the
  [375-pixel matrix](output-ux-matrix-mobile.png) and
  [1280-pixel matrix](output-ux-matrix-desktop.png).

Python regressions cover display/search/sample denominators, empty states,
comparison handoffs, overview coverage, escaping, topology disclosure, saved-result
immutability, matrix orientation, and population mismatch markers. All 358 Python tests passed, including public docstring examples and editor
contracts; type checks and lint/format checks also passed. Generated example assets and notebook outputs are
refreshed with the current renderer.

## Boundaries and deferred work

The HTML is generated from bounded saved evidence; it does not load unsaved records
or recompute analysis. Very large explicitly unbounded reports can still produce
large documents. DOM virtualization and live notebook widgets are deferred because
they require a larger state/runtime contract. A native matrix with hundreds of
candidate keys still requires filtering or horizontal scrolling. Fit-width SVGs
cannot preserve readable labels for arbitrarily large graphs. No automated browser
checks substitute for testing with screen readers, touch hardware, or other engines;
this qualification used Chromium only.

The stable documentation repository is deliberately unchanged until release
synchronization, in accordance with its source-version policy.
