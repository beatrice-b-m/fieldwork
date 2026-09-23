"""Presentation limits, source handoffs and offline HTML evidence, checked structurally.

HTML IDs, links, escaping, non-mutation and topology suppression for every result
kind are covered once in tests/test_rendering_contracts.py.
"""

from html.parser import HTMLParser

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

    def with_attribute(self, name):
        return [attrs for _, attrs in self.elements if name in attrs]


@pytest.fixture
def frame():
    return pd.DataFrame({"key": [1, 1, 2, 2], "value": ["a", "a", "b", None]}, index=[9] * 4)


def test_display_limit_is_distinct_from_search_coverage(frame):
    result = fw.discover_dependencies(
        frame, limits={"max_dependency_tests": 1}, include_grain=False
    )
    coverage = fw.visualization_data(result)["coverage"]
    # The search budget is recorded in the evidence ...
    assert (coverage["dependency_tests"], coverage["dependency_tests_possible"]) == (1, 2)
    assert coverage["dependency_tests_omitted"] == 1
    # ... while a display budget only hides records from the page.
    assert Report(fw.render_html(result)).with_attribute("data-record")
    assert not Report(fw.render_html(result, max_findings=0)).with_attribute("data-record")
    assert result["findings"]


def test_overview_retains_independent_section_coverage(frame):
    result = fw.explore(frame, options={"dependencies": {"limits": {"max_dependency_tests": 0}}})
    data = fw.visualization_data(result)
    assert data["section_coverage"]["missingness"]["pairs_evaluated"] == 1
    assert data["section_coverage"]["dependencies"]["dependency_tests"] == 0
    assert "section_coverage" not in fw.visualization_data(result, detail="topology")


def test_samples_are_source_positions_with_full_support_and_executable_handoff(frame):
    result = fw.missingness(frame, limits={"example_limit": 1})
    row = next(r for r in result["findings"] if r["pattern"] == "availability")
    assert row["features"][0]["column"] == "value"
    assert (row["examples"]["positions"], row["examples"]["total"]) == ([0], 3)
    assert len(result.inspect(frame, row["id"])) == 1
    assert len(result.inspect(frame, row["id"], all_matches=True)) == 3


def test_comparison_has_no_inspection_handoff(frame):
    result = fw.compare(fw.missingness(frame), fw.missingness(frame.iloc[:2]))
    assert all(f["examples"]["positions"] == [] for f in result["findings"])
    assert "result.inspect(" not in fw.render_html(result)


def test_empty_search_is_reported_in_coverage(frame):
    empty = fw.discover_dependencies(frame, limits={"max_candidates": 0}, include_grain=False)
    assert not empty["findings"]
    coverage = fw.visualization_data(empty)["coverage"]
    assert (coverage["candidates_evaluated"], coverage["candidate_space"]) == (0, 3)


def test_page_is_offline_and_finding_links_point_to_included_records(frame):
    result = fw.explore(frame)
    markup = fw.render_html(result, max_findings=2)
    report = Report(markup)
    findings = [attrs for attrs in report.with_attribute("data-record") if "data-pattern" in attrs]
    assert [attrs["id"] for attrs in findings] == [f["id"] for f in result.findings[:2]]
    shown = {attrs["id"] for attrs in findings}
    links = [a["href"][1:] for a in report.with_attribute("href") if a["href"].startswith("#f")]
    assert links and set(links) <= shown
    assert markup.count("<script>") == 1
    assert "<script src=" not in markup and "<link " not in markup


def test_wide_grain_matrix_keeps_every_feature():
    frame = pd.DataFrame({"key": [1, 1, 2, 2], **{f"feature_{i}": [0, 0, 1, 1] for i in range(60)}})
    rows = Report(fw.render_html(fw.grain(frame, ["key"]))).with_attribute("data-matrix-feature")
    assert {row["data-matrix-feature"] for row in rows} == set(frame.columns)
    assert len(rows) == len(frame.columns)


def test_matrix_marks_population_mismatch_and_undefined_support():
    frame = pd.DataFrame({"key": [1, 1, 2], "value": ["a", None, None]})
    data = fw.visualization_data(fw.grain(frame, ["key"], dropna=True))
    value = next(f for f in data["features"] if f["label"] == "value")
    assert value["reason"] == "different_target_population"
    empty = fw.visualization_data(fw.grain(frame.iloc[:0], ["key"], dropna=True))
    assert all(e["evaluated_groups"] == 0 for e in empty["evidence"])
