from __future__ import annotations

import sys
from copy import deepcopy

import pandas as pd
import pytest

from fieldwork import (
    ExplorerResult,
    census,
    explore,
    infer_schema,
    levels,
    render_plaintext,
    visualization_data,
)

# Escaping, width/line bounds, non-mutation and topology suppression across all
# media are covered once in tests/test_rendering_contracts.py.


def node_lines(text, dimensions):
    """Lines that present a census node, in output order."""
    return [
        line.strip() for line in text.splitlines() if line.strip().startswith(tuple(dimensions))
    ]


def test_census_text_lists_every_node_in_tree_order():
    frame = pd.DataFrame(
        {
            "site": ["North", "North", "South"],
            "modality": ["CT", "MRI", "CT"],
            "side": ["L", "R", "L"],
        }
    )
    dimensions = [f"{d}=" for d in frame.columns]
    result = census(frame, list(frame.columns))
    children = {}
    for node in result["tree"]["nodes"]:
        children.setdefault(node["parent_id"], []).append(node)

    def preorder(parent):
        for node in children.get(parent, []):
            yield node
            yield from preorder(node["node_id"])

    expected = [f"{node['column']}={node['value']}" for node in preorder("root")]
    lines = node_lines(render_plaintext(result, width=88), dimensions)
    # Each subtree follows its parent contiguously; repeated labels stay separate.
    assert [line.split(":")[0] for line in lines] == expected
    assert all(str(node["count"]) in line for node, line in zip(preorder("root"), lines))
    saved = result.to_dict()
    for node in saved["tree"]["nodes"]:
        node["depth"] = 99  # explicit parentage, not depth, defines the tree
    assert render_plaintext(saved, width=88) == render_plaintext(result, width=88)


def test_renderer_node_budget_hides_nodes_without_changing_the_analysis():
    frame = pd.DataFrame({"site": ["North"] * 6 + ["South"] * 4})
    result = census(frame, ["site"])
    before = deepcopy(result.to_dict())
    hidden = render_plaintext(result, max_nodes=0)
    assert not node_lines(hidden, ["site="])
    assert len(node_lines(render_plaintext(result), ["site="])) == 2
    assert result.to_dict() == before
    # An analysis budget instead records the omitted mass in the result.
    limited = census(frame, ["site"], max_nodes=0)["tree"]["root"]
    assert (limited["omitted_child_rows"], limited["omitted_child_levels"]) == (10, 2)


def test_typed_display_labels_do_not_collide():
    values = [1, "1", True, "True", 1.0, "1.0", None, "<NA>", "'<NA>'", "", "\n", r"\n"]
    values.append(pd.Timedelta(1, "ns"))
    result = levels(pd.DataFrame({"value": pd.Series(values, dtype=object)}))
    labels = [row["label"] for row in visualization_data(result)["features"][0]["rows"]]
    assert len(labels) == len(set(labels)) == len(values)
    text = render_plaintext(result)
    assert all(label in text for label in labels if label.isprintable())


def test_pair_text_names_columns_and_contexts_not_internal_ids():
    frame = pd.DataFrame(
        {"alpha": ["x", "x", "x", "y"], "beta": [1, 2, 1, 1], "site": ["N", "N", "S", "S"]}
    )
    result = explore(frame, ["alpha", "beta"], pair_contexts=[{"site": "N"}, {"site": "S"}])
    text = render_plaintext(result["sections"]["pairs"], width=100)
    contexts = visualization_data(result, section="pairs")["contexts"]
    assert all(context["label"] in text for context in contexts[1:])  # after the global one
    assert "alpha" in text and "beta" in text
    assert "f0" not in text and "f1" not in text


def test_warning_codes_and_unrequested_sections_are_visible():
    frame = pd.DataFrame({"id": [1, 2], "a": pd.Series(["x", 1], dtype=object)})
    assert "EXPLICIT_ROLE_SELECTION" in render_plaintext(levels(frame, ["id"], schema={"id": "id"}))
    assert "MIXED_LEVEL_TYPES" in render_plaintext(levels(frame, ["a"]))
    text = render_plaintext(explore(frame, ["a"], include_pairs=False))
    assert text.count("not_requested") == 2  # grain and pairs


def test_line_budget_marks_only_actual_truncation():
    result = levels(pd.DataFrame({"a": ["x"]}))
    complete = render_plaintext(result)
    size = len(complete.splitlines())
    assert render_plaintext(result, max_lines=size) == complete
    assert render_plaintext(result, max_lines=size + 1) == complete
    limited = render_plaintext(result, max_lines=size - 1).splitlines()
    assert len(limited) == size - 1
    assert limited[:-1] == complete.splitlines()[: size - 2]
    assert limited[-1] not in complete.splitlines()


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


def test_topology_orders_levels_canonically_not_by_count():
    frame = pd.DataFrame({"site": ["Z", "Z", "Z", "A"], "kind": ["x", "y", "x", "x"]})
    for result in (levels(frame, ["site"]), census(frame, ["site", "kind"])):
        full = node_lines(render_plaintext(result), ["site=", "Z", "A"])
        topology = node_lines(render_plaintext(result, detail="topology"), ["site=", "Z", "A"])
        assert "Z" in full[0]  # full detail ranks by count
        assert "A" in topology[0]


def test_schema_topology_does_not_depend_on_quantities():
    frame = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "b"]})
    changed = pd.concat([frame, frame.iloc[[2] * 5]], ignore_index=True)
    first = infer_schema(frame.assign(id=[1, 2, 3]), candidate_keys=["id"])
    second = infer_schema(changed.assign(id=range(8)), candidate_keys=["id"])
    assert render_plaintext(first, detail="topology") == render_plaintext(second, detail="topology")
    assert render_plaintext(first) != render_plaintext(second)


def test_topology_detail_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="detail"):
        render_plaintext(levels(pd.DataFrame({"a": [1]})), detail="summary")
