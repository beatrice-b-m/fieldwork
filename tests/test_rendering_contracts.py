"""Rendering contracts shared by every result kind and output medium.

Structure only: outputs are valid and self-contained, never mutate saved results,
escape hostile labels, keep HTML IDs and links consistent, and in topology mode
depend on no quantity. No sentence text is asserted.
"""

import json
from copy import deepcopy
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

import pandas as pd
import pytest

import fieldwork as fw
from fieldwork import KeySpec

DETAILS = ("full", "topology")


def delivery():
    """Twelve distinct export slots: exams nest in patients, CT/MR payloads exclude."""
    rows = []
    for exam in range(6):
        for side in "LR":
            ct = exam % 2 == 0
            rows.append(
                {
                    "patient": f"P{exam // 2}",
                    "exam": exam,
                    "side": side,
                    "site": "AB"[exam // 2 % 2],
                    "modality": "CT" if ct else "MR",
                    "dose": 1.5 * (exam + 1) if ct else None,
                    "image": None if ct else f"img-{exam}{side}",
                    "finding": ["clear", "scar"][(exam + (side == "R")) % 2],
                }
            )
    return pd.DataFrame(rows)


KINDS = {
    "levels": lambda df: fw.levels(df, ["side", "finding"]),
    "census": lambda df: fw.census(df, ["site", "patient", "exam"]),
    "grain": lambda df: fw.grain(df, ["patient", "exam", KeySpec("exam_side", ("exam", "side"))]),
    "pairs": lambda df: fw.explore(
        df, ["side", "finding"], pair_contexts=[{"site": "A"}], include_absence=True
    )["sections"]["pairs"],
    "joint_counts": lambda df: fw.joint_counts(df, ["side", "finding"]),
    "explicit_explore": lambda df: fw.explore(df, ["site", "exam"], candidate_keys=["patient"]),
    "missingness": lambda df: fw.missingness(df, by=["site"], entity="patient"),
    "entity_missingness": lambda df: fw.missingness(df, entity="patient", unit="entities"),
    "dependencies": lambda df: fw.discover_dependencies(df, by=["site"]),
    "paths": lambda df: fw.suggest_paths(df),
    "value_patterns": lambda df: fw.value_patterns(df, by=["site"]),
    "overview": lambda df: fw.explore(df),
    "comparison": lambda df: fw.compare(
        fw.missingness(df), fw.missingness(df, scope=fw.Scope.from_positions(df, range(12)))
    ),
}


def exported(result):
    return result.to_dict() if hasattr(result, "to_dict") else deepcopy(dict(result))


def render(medium, result, **options):
    if medium == "data":
        return json.dumps(fw.visualization_data(result, **options), allow_nan=False)
    return {"text": fw.render_plaintext, "svg": fw.render_svg, "html": fw.render_html}[medium](
        result, **options
    )


MEDIA = ("text", "svg", "html", "data")


class Elements(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.elements = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


@pytest.mark.parametrize("kind", KINDS)
def test_outputs_are_valid_self_contained_and_leave_saved_results_unchanged(kind):
    result = KINDS[kind](pd.concat([delivery()] * 2, ignore_index=True))
    saved = json.loads(json.dumps(exported(result), allow_nan=False))
    before = deepcopy(saved)
    for detail in DETAILS:
        # Saved evidence renders without the source, exactly as the live result.
        for medium in MEDIA:
            assert render(medium, saved, detail=detail) == render(medium, result, detail=detail)
        assert ET.fromstring(render("svg", saved, detail=detail)).tag.endswith("svg")
        document = render("html", saved, detail=detail)
        assert document.startswith("<!doctype html>")
        assert document.count("<script>") == 1
        assert "<script src=" not in document and "<link " not in document
    assert saved == before
    assert exported(result) == before


@pytest.mark.parametrize("kind", KINDS)
def test_topology_depends_on_structure_not_quantities(kind):
    # Every distinct row twice, against a third copy of some rows only: every
    # count and fraction changes, while every relationship and grouping holds.
    base = delivery()
    first = KINDS[kind](pd.concat([base, base], ignore_index=True))
    second = KINDS[kind](pd.concat([base, base, base.iloc[[0, 3, 4, 9]]], ignore_index=True))
    for medium in MEDIA:
        assert render(medium, first, detail="topology") == render(
            medium, second, detail="topology"
        ), medium
    assert render("text", first) != render("text", second)
    projection = json.dumps(fw.visualization_data(first, detail="topology"))
    for key in ("positions", "selection_positions", "dataset_id", "measurements"):
        assert f'"{key}"' not in projection


SENTINEL = 987654321
QUANTITIES = {
    "count",
    "rows",
    "total",
    "populated",
    "missing",
    "denominator",
    "input_rows",
    "evaluated_rows",
    "groups",
    "evaluated_groups",
    "repeated_groups",
    "repeated_rows",
    "singleton_groups",
    "violating_groups",
    "affected_rows",
    "global_targets_tested",
}


def inject(value):
    """Replace every quantity, position and support list; add a private measurement."""
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            if key in {"positions", "selection_positions"} and isinstance(child, list):
                child = [SENTINEL]
            elif key == "determines_with_repeated_support":
                child = ["secret-value"]
            elif key in QUANTITIES and type(child) is int:
                child = SENTINEL
            else:
                child = inject(child)
            if key == "measurements" and isinstance(child, dict):
                child = {**child, "private_metric": SENTINEL}
            output[key] = child
        return output
    if isinstance(value, list):
        return [inject(child) for child in value]
    return value


@pytest.mark.parametrize("kind", KINDS)
def test_topology_allowlist_hides_injected_quantities_and_fields(kind):
    saved = inject(exported(KINDS[kind](delivery())))
    saved["unrecognized_field"] = "secret-value"
    for medium in MEDIA:
        output = render(medium, saved, detail="topology")
        assert str(SENTINEL) not in output, medium
        assert "secret-value" not in output, medium
    assert str(SENTINEL) in render("html", saved)


HOSTILE = '</script><img src=x onerror="alert(1)">\x1b[31m\x00\u202e'


@pytest.mark.parametrize(
    "make",
    [
        lambda df: fw.levels(df),
        lambda df: fw.grain(df, [KeySpec(HOSTILE, (HOSTILE,))]),
        lambda df: fw.missingness(df),
        lambda df: fw.explore(df),
    ],
    ids=["levels", "grain", "missingness", "overview"],
)
def test_hostile_labels_and_values_are_escaped_everywhere(make):
    frame = pd.DataFrame(
        {
            HOSTILE: pd.Series([HOSTILE, "☃", None, True, 1, "1"], dtype=object),
            "plain": ["a", "a", "b", "b", None, "c"],
        }
    )
    result = make(frame)
    for detail in DETAILS:
        for mode in ("safe", "display"):
            text = fw.render_plaintext(result, detail=detail, unicode_mode=mode)
            assert not {"\x1b", "\x00", "\u202e"} & set(text)
            assert text.isascii() or mode == "display"
        for width, max_lines in [(3, 2), (40, 10)]:
            text = fw.render_plaintext(
                result, width=width, max_lines=max_lines, unicode_mode="safe"
            )
            assert len(text.splitlines()) <= max_lines
            assert all(len(line) <= width for line in text.splitlines())
        svg = ET.fromstring(fw.render_svg(result, detail=detail))
        assert not list(svg.iter("{http://www.w3.org/2000/svg}script"))
        document = fw.render_html(result, detail=detail)
        tags = [tag for tag, _ in Elements(document).elements]
        assert "img" not in tags and tags.count("script") == 1
        assert not {"\x1b", "\x00", "\u202e"} & set(document)


@pytest.mark.parametrize(
    ("make", "section"),
    [
        (lambda df: fw.grain(df, ["patient", "exam"]), None),
        (
            lambda df: fw.explore(
                df, ["side", "finding"], pair_contexts=[{"site": "A"}, {"site": "B"}]
            ),
            "pairs",
        ),
        (lambda df: fw.missingness(df, by=["site"]), None),
        (lambda df: fw.explore(df), None),
    ],
    ids=["grain", "pairs_with_contexts", "missingness", "overview"],
)
@pytest.mark.parametrize("max_findings", [None, 2])
def test_html_ids_are_unique_and_internal_links_resolve(make, section, max_findings):
    options = {"section": section} if section else {}
    if max_findings is not None:
        options["max_findings"] = max_findings
    elements = Elements(fw.render_html(make(delivery()), **options)).elements
    ids = [attrs["id"] for _, attrs in elements if "id" in attrs]
    assert len(ids) == len(set(ids))
    for tag, attrs in elements:
        href = attrs.get("href", "")
        # A bare "#" is the hidden selection link that the page script fills in.
        if tag == "a" and href.startswith("#") and href != "#":
            assert href[1:] in ids
