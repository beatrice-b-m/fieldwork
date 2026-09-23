"""Contracts for presentation limits, source handoffs, and offline HTML evidence."""

from copy import deepcopy
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

import pandas as pd
import pytest

import fieldwork as fw


class Report(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.elements = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


@pytest.fixture
def frame():
    return pd.DataFrame({"key": [1, 1, 2, 2], "value": ["a", "a", "b", None]}, index=[9] * 4)


def test_display_limit_is_distinct_from_search_coverage(frame):
    result = fw.discover_dependencies(frame, max_dependency_tests=1, include_grain=False)
    markup = fw.render_html(result, max_findings=0)
    assert "Some search or retention limits were reached" in markup
    assert "1/2 dependency tests completed (limited)" in markup
    assert "0 of 1 records included" in markup
    assert "Search only covers included records" in markup
    assert not any("data-record" in attrs for _, attrs in Report(markup).elements)


def test_overview_retains_independent_section_coverage(frame):
    result = fw.explore(frame, section_options={"dependencies": {"max_dependency_tests": 0}})
    data = fw.visualization_data(result)
    assert data["section_coverage"]["missingness"]["pairs_evaluated"] == 1
    assert data["section_coverage"]["dependencies"]["dependency_tests"] == 0
    markup = fw.render_html(result)
    assert "dependencies: 0/2 dependency tests completed (limited)" in markup
    assert "Search limits reached" in str(result)
    assert "Suggested census paths" in str(result)
    assert "section_coverage" not in fw.visualization_data(result, detail="topology")


def test_samples_are_source_positions_with_full_support_and_executable_handoff(frame):
    result = fw.missingness(frame, example_limit=1)
    row = next(r for r in result["findings"] if r["pattern"] == "availability")
    assert row["features"][0]["column"] == "value"
    markup = fw.render_html(result)
    assert "Examples: 1/3 saved source positions [0]" in markup
    assert "not dataframe index labels" in markup
    assert "first matches in source order" in markup
    assert "Examples: 1/3 saved source positions [0]" in fw.render_plaintext(result)
    assert len(result.inspect(frame, row["id"])) == 1
    assert len(result.inspect(frame, row["id"], all_matches=True)) == 3


def test_comparison_has_no_misleading_inspection_handoff(frame):
    result = fw.compare(fw.missingness(frame), fw.missingness(frame.iloc[:2]))
    markup = fw.render_html(result)
    assert "Before: input; 4 evaluated rows" in markup
    assert "After: input; 2 evaluated rows" in markup
    assert "25 percentage points" in markup
    assert "result.inspect(" not in markup
    assert "Representative source rows" not in markup


def test_empty_and_suppressed_results_have_different_explanations(frame):
    empty = fw.discover_dependencies(frame, max_candidates=0, include_grain=False)
    assert "No records were saved" in fw.render_html(empty)
    assert "No findings saved; review coverage" in fw.render_plaintext(empty)
    assert "No records saved" in fw.render_svg(empty)
    suppressed = fw.render_html(fw.missingness(frame), max_findings=0)
    assert "Display limit reached" in suppressed
    assert "No records were saved" not in suppressed


def test_svg_omissions_count_each_dependency_list(frame):
    result = fw.discover_dependencies(frame, include_grain=False)
    text = " ".join(ET.fromstring(fw.render_svg(result, max_findings=0)).itertext())
    assert "Candidate Grains: 0/3 shown" in text
    assert "Dependency Tests: 0/2 shown" in text
    assert "candidate grains not rendered" in fw.render_plaintext(result, max_nodes=0)


def test_controls_are_labeled_and_evidence_links_have_included_targets(frame):
    markup = fw.render_html(fw.explore(frame), max_findings=2)
    parsed = Report(markup)
    ids = [attrs["id"] for _, attrs in parsed.elements if "id" in attrs]
    assert len(ids) == len(set(ids))
    for tag, attrs in parsed.elements:
        if tag == "a" and attrs.get("href", "").startswith("#"):
            assert attrs["href"][1:] in ids
    assert "data-pattern-filter" in markup
    assert "data-exceptions-filter" in markup
    assert 'role="status"' in markup
    assert "evidence outside display limit" in markup
    assert markup.count("<script>") == 1
    assert "<script src=" not in markup
    assert "data-enhance hidden" in markup


def test_html_escapes_labels_controls_and_python_handoffs(frame):
    label = '</script><img src=x onerror="alert(1)">\x00\u202e'
    frame = frame.rename(columns={"value": label})
    markup = fw.render_html(fw.explore(frame))
    parsed = Report(markup)
    assert all(tag != "img" for tag, _ in parsed.elements)
    assert sum(tag == "script" for tag, _ in parsed.elements) == 1
    assert "\x00" not in markup and "\u202e" not in markup
    assert "\\u0000" in markup


def test_topology_does_not_include_hidden_measurements_or_selectors(frame):
    result = fw.missingness(frame).to_dict()
    before = deepcopy(result)
    for record in result["findings"]:
        record["measurements"]["private_metric"] = 987654321
        record["examples"]["positions"] = [987654321]
    markup = fw.render_html(result, detail="topology")
    assert "987654321" not in markup
    assert "Search coverage and limits" not in markup
    assert "result.inspect(" not in markup
    assert "data-exceptions=" not in markup
    assert "data-exceptions-filter" not in {
        key for _, attrs in Report(markup).elements for key in attrs
    }
    fw.render_html(before)
    assert before == fw.missingness(frame).to_dict()


def test_renderers_do_not_mutate_saved_result(frame):
    result = fw.explore(frame).to_dict()
    before = deepcopy(result)
    for render in (fw.render_html, fw.render_svg, fw.render_plaintext):
        render(result)
        render(result, detail="topology")
    assert before == result


def test_wide_grain_matrix_grows_downward_and_keeps_feature_evidence():
    frame = pd.DataFrame({"key": [1, 1, 2, 2], **{f"feature_{i}": [0, 0, 1, 1] for i in range(60)}})
    result = fw.grain(frame, ["key"])
    svg = ET.fromstring(fw.render_svg(result, view="matrix"))
    assert int(svg.attrib["width"]) == 900
    assert int(svg.attrib["height"]) > 2000
    parsed = Report(fw.render_html(result))
    rows = [
        attrs for tag, attrs in parsed.elements if tag == "tr" and "data-matrix-feature" in attrs
    ]
    assert len(rows) == 61
    assert {row["data-matrix-feature"] for row in rows} == set(frame.columns)
    assert any(attrs.get("id") == "matrix-search" for _, attrs in parsed.elements)
    assert any(attrs.get("id") == "matrix-key" for _, attrs in parsed.elements)
    assert any(attrs.get("id") == "evidence-f60" for _, attrs in parsed.elements)


def test_matrix_explains_population_mismatch_and_undefined_support():
    frame = pd.DataFrame({"key": [1, 1, 2], "value": ["a", None, None]})
    result = fw.grain(frame, ["key"], dropna=True)
    markup = fw.render_html(result)
    assert "constant *" in markup
    assert "different target population" in markup
    assert "Undefined means no evaluated support" in markup
    no_support = fw.grain(frame.iloc[:0], ["key"], dropna=True)
    assert "undefined" in fw.render_html(no_support)
