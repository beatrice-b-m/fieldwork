from __future__ import annotations

import sys
from copy import deepcopy

import pandas as pd
import pytest

from fieldwork import ExplorerResult, census, explore, grain, infer_schema, levels, render_plaintext


def test_census_subtrees_are_contiguous_and_dimensions_named() -> None:
    frame = pd.DataFrame(
        {
            "site": ["North", "North", "South"],
            "modality": ["CT", "MRI", "CT"],
            "side": ["L", "R", "L"],
        }
    )
    result = census(frame, ["site", "modality", "side"])
    before = deepcopy(result.to_dict())
    text = render_plaintext(result, width=88)
    assert text.splitlines() == [
        "Fieldwork feature explorer v0.3",
        "Census (computed)",
        "  rows: 3 evaluated / 3 input; excluded: 0 missing, 0 restricted",
        "  path: site > modality > side",
        "  total: 3 rows",
        "  site='North': 2 rows",
        "    modality='CT': 1 row",
        "      side='L': 1 row",
        "    modality='MRI': 1 row",
        "      side='R': 1 row",
        "  site='South': 1 row",
        "    modality='CT': 1 row",
        "      side='L': 1 row",
    ]
    assert render_plaintext(before, width=88) == text
    assert result.to_dict() == before
    # Node depth cannot override explicit parentage.
    for node in before["tree"]["nodes"]:
        node["depth"] = 99
    assert render_plaintext(before, width=88) == text


def test_census_totals_omissions_and_renderer_budget_are_distinct() -> None:
    frame = pd.DataFrame({"site": ["North"] * 6 + ["South"] * 4})
    top = render_plaintext(census(frame, ["site"], top_n=1))
    assert "total: 10 rows" in top
    assert "4 rows in 1 child level omitted (top_n)" in top
    totals_only = render_plaintext(census(frame, ["site"], max_nodes=0))
    assert "total: 10 rows" in totals_only
    assert "10 rows in 2 child levels omitted (max_nodes)" in totals_only
    renderer_only = render_plaintext(census(frame, ["site"]), max_nodes=0)
    assert "total: 10 rows" in renderer_only
    assert "2 nodes not rendered (renderer max_nodes)" in renderer_only
    assert "site='North'" not in renderer_only


def test_typed_display_labels_do_not_collide() -> None:
    values = [
        1,
        "1",
        True,
        "True",
        1.0,
        "1.0",
        None,
        "<NA>",
        "'<NA>'",
        "",
        "\n",
        r"\n",
        pd.Timedelta(1, "ns"),
    ]
    result = levels(pd.DataFrame({"value": pd.Series(values, dtype=object)}))
    text = render_plaintext(result)
    labels = [
        line.strip().rsplit(": ", 1)[0]
        for line in text.splitlines()
        if line.startswith("    ") and not line.strip().startswith("Warning:")
    ]
    assert len(labels) == len(values)
    assert len(set(labels)) == len(values)
    assert {"1", "'1'", "True", "'True'", "<NA>", "'<NA>'", "0 days 00:00:00.000000001"} <= set(
        labels
    )
    custom = render_plaintext(levels(pd.DataFrame({"a": [None, "x"]})), missing_label="'x'")
    assert "string('x')" in custom


def test_levels_show_missing_exclusions_and_omitted_mass() -> None:
    text = render_plaintext(
        levels(pd.DataFrame({"a": ["x", "x", "y", None]}), top_n=1, dropna=True)
    )
    assert "2/3 evaluated rows reported" in text
    assert "3 evaluated / 4 input; excluded: 1 missing" in text
    assert "1 level / 1 row not reported" in text


def test_pairs_have_names_context_population_direction_and_absence_basis() -> None:
    frame = pd.DataFrame(
        {"a": ["x", "x", "x", "y"], "b": [1, 2, 1, 1], "site": ["N", "N", "S", "S"]}
    )
    result = explore(
        frame,
        ["a", "b"],
        pair_contexts=[{"site": "N"}, {"site": "S"}],
        include_absence=True,
        reference_domains={"a": ["x", "y", "z"]},
    )
    text = render_plaintext(result["sections"]["pairs"], width=100)
    assert "a / b [global]" in text
    assert "a / b [site='N']" in text
    assert "a / b [site='S']" in text
    assert "4 evaluated / 4 input" in text
    assert "2 evaluated / 4 input" in text
    assert "1:n (one A to many B)" in text
    assert "n:1 (many A to one B)" in text
    assert "Cramer's V: undefined (constant dimension)" in text
    assert "domain a: 3 levels (caller declared)" in text
    assert "domain b: 2 levels (empirical observed)" in text
    assert "zero support in cohort: 2 cells" in text
    assert "level absent in context:" in text
    assert "within supported margins:" in text
    assert "unobserved example:" in text
    assert "not evidence of impossibility" in text
    assert "f0 / f1" not in text


def test_pair_and_context_limits_are_reported_even_with_no_records() -> None:
    frame = pd.DataFrame({"a": ["x"], "b": [1], "c": ["z"], "site": ["N"]})
    result = explore(
        frame, ["a", "b", "c"], max_pairs=1, max_contexts=0, pair_contexts=[{"site": "N"}]
    )
    text = render_plaintext(result["sections"]["pairs"])
    assert "omitted: 2 requested pairs, 2 requested contexts" in text
    assert "Pairs (empty)" in text


def test_conditional_pair_and_grain_sections_show_original_population() -> None:
    frame = pd.DataFrame({"id": [1, 2, 3], "a": ["x", "x", "y"], "b": [1, 1, 2]})
    result = explore(
        frame, ["a", "b"], candidate_keys=["id"], top_n=1, top_n_mode="pre", top_n_applies_to="both"
    )
    for section in ("pairs", "grain"):
        text = render_plaintext(result["sections"][section], width=100)
        assert "conditional" in text
        assert "cohort from s2" in text
        assert "2 evaluated / 3 input" in text
        assert "0 missing, 1 restricted" in text


def test_grain_support_distinguishes_singletons_and_repeated_observations() -> None:
    frame = pd.DataFrame({"id": [1, 2], "target": ["a", "b"]})
    singletons = render_plaintext(grain(frame, ["id"]))
    repeated = render_plaintext(grain(pd.concat([frame, frame]), ["id"]))
    assert "2 singleton, 0 repeated groups" in singletons
    assert "0 singleton, 2 repeated groups" in repeated
    assert "observed dependency holds" in singletons
    conflict = render_plaintext(grain(pd.DataFrame({"id": [1, 1], "target": ["a", "b"]}), ["id"]))
    assert "observed dependency fails; 1/1 violating groups; 2 affected rows" in conflict
    empty = render_plaintext(
        grain(pd.DataFrame({"id": [None], "target": [None]}), ["id"], dropna=True)
    )
    assert "observed dependency undefined (no evaluated groups)" in empty
    assert "0 evaluated / 1 input; excluded: 1 missing" in empty


def test_schema_proposals_and_warnings_are_visible() -> None:
    frame = pd.DataFrame({"id": [1, 2], "a": ["x", "x"]})
    text = render_plaintext(infer_schema(frame))
    assert "Schema proposals (suggestions)" in text
    assert "id: suggested id" in text
    assert "cardinality: 2" in text
    assert "dependency evidence: not evaluated" in text
    text = render_plaintext(levels(frame, ["id"], schema={"id": "id"}))
    assert "Warning: EXPLICIT_ROLE_SELECTION" in text
    text = render_plaintext(explore(frame, ["a"], include_pairs=False))
    assert "Grain (not_requested)" in text
    assert "Pairs (not_requested)" in text


def test_line_budget_marks_only_actual_truncation() -> None:
    result = levels(pd.DataFrame({"a": ["x"]}))
    complete = render_plaintext(result)
    size = len(complete.splitlines())
    assert render_plaintext(result, max_lines=size) == complete
    assert render_plaintext(result, max_lines=size + 1) == complete
    limited = render_plaintext(result, max_lines=size - 1)
    assert len(limited.splitlines()) == size - 1
    assert limited.endswith("... more output not rendered (max_lines)")


@pytest.mark.parametrize("width", [1, 3, 20, 88])
@pytest.mark.parametrize("max_lines", [1, 2, 5, 100])
def test_safe_rendering_limits_controls_and_escapes(width: int, max_lines: int) -> None:
    result = levels(pd.DataFrame({"bad\x1b\nlabel": ["\x1b[31mred", "☃", "\u202eevil", "界"]}))
    text = render_plaintext(result, width=width, max_lines=max_lines, unicode_mode="safe")
    assert len(text.splitlines()) <= max_lines
    assert all(len(line) <= width for line in text.splitlines())
    assert text.isascii()
    assert "\x1b" not in text and "\u202e" not in text


@pytest.mark.parametrize("wcwidth_installed", [True, False])
def test_unicode_is_displayed_by_default_within_cell_width(
    monkeypatch: pytest.MonkeyPatch, wcwidth_installed: bool
) -> None:
    from fieldwork._explore import render

    render._width_function.cache_clear()
    if not wcwidth_installed:
        monkeypatch.setitem(sys.modules, "wcwidth", None)
    try:
        result = levels(pd.DataFrame({"a": ["界" * 10, "e\u0301", "\u202eevil"]}))
        text = render_plaintext(result, width=12)
        assert "界" in text and "\u202e" not in text
        assert all(sum(render._fallback_width(c) for c in line) <= 12 for line in text.splitlines())
    finally:
        render._width_function.cache_clear()


def test_render_does_not_require_result_serialization() -> None:
    class NoSerialization(ExplorerResult):
        def to_dict(self):
            raise AssertionError("renderer must consume payload directly")

    result = levels(pd.DataFrame({"a": [1]}))
    assert render_plaintext(NoSerialization(result.kind, result.payload)) == render_plaintext(
        result
    )


def test_topology_detail_suppresses_quantities_and_canonicalizes_ranked_values() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 1, 2, 3],
            "site": ["Z", "Z", "Z", "A"],
            "kind": ["x", "y", "x", "x"],
            "finding": ["ok", "bad", "ok", None],
        }
    )
    result = explore(
        frame,
        ["site", "kind"],
        features=["site", "kind", "finding"],
        candidate_keys=["id"],
        include_absence=True,
        reference_domains={"site": ["A", "Z", "Q"], "kind": ["x", "y"]},
    )
    text = render_plaintext(result, detail="topology", width=120)
    assert "Topology display (quantitative evidence suppressed)" in text
    assert text.index("    'A'") < text.index("    'Z'")
    assert text.index("  site='A'") < text.index("  site='Z'")
    assert "observed dependency holds" in text
    assert "observed dependency fails" in text
    assert "observed relation: n:m (many-to-many)" in text
    assert "unobserved example: 'A' / 'y'" in text
    assert "domain site: caller declared" in text
    forbidden = (
        " evaluated / ",
        " excluded:",
        " rows",
        " row",
        " levels;",
        " support:",
        " violating groups",
        " affected rows",
        "Cramer's V",
        "domain cells",
        " cells",
        "cardinality",
        "missing rows",
    )
    assert not any(fragment in text for fragment in forbidden)


def test_topology_detail_uses_nonquantitative_omission_markers() -> None:
    frame = pd.DataFrame({"site": ["Z"] * 6 + ["A"] * 4})
    analysis_limited = render_plaintext(census(frame, ["site"], top_n=1), detail="topology")
    assert "child branches omitted (top_n)" in analysis_limited
    assert "4 rows" not in analysis_limited
    renderer_limited = render_plaintext(census(frame, ["site"]), detail="topology", max_nodes=0)
    assert "additional nodes not rendered (renderer max_nodes)" in renderer_limited
    assert "2 nodes" not in renderer_limited
    level_limited = render_plaintext(levels(frame, ["site"], top_n=1), detail="topology")
    assert "additional levels not reported (analysis limits)" in level_limited


def test_topology_detail_suppresses_schema_metrics_and_support_counts() -> None:
    frame = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    text = render_plaintext(infer_schema(frame, candidate_keys=["id"]), detail="topology")
    assert "id: suggested id" in text
    assert "dtype:" in text
    assert "name hint id: True" in text
    assert "id: holds=True" in text
    assert "cardinality" not in text
    assert "missing rows" not in text
    assert "groups" not in text


def test_topology_detail_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="detail"):
        render_plaintext(levels(pd.DataFrame({"a": [1]})), detail="summary")
