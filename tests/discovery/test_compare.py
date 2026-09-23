"""compare aligns availability by feature name across differing deliveries."""

import pandas as pd
import pytest

import fieldwork as fw


def changes(before, after):
    return {c["feature"]: c for c in fw.compare(before, after)["changes"]}


def test_added_removed_and_reordered_features_align_by_name():
    before = fw.missingness(pd.DataFrame({"a": [1, None, 3, 4], "b": [None, None, None, 2]}))
    after = fw.missingness(pd.DataFrame({"c": [1, 2], "b": [1, None]}))
    result = changes(before, after)
    assert list(result) == ["a", "b", "c"]
    # Removed feature: only the before side exists.
    assert result["a"]["before"]["populated"] == 3
    assert result["a"]["after"] is None
    assert result["a"]["populated_fraction_delta"] is None
    # Shared feature: after minus before, in fractions of each own denominator.
    assert result["b"]["before"]["denominator"] == 4
    assert result["b"]["after"]["denominator"] == 2
    assert result["b"]["populated_fraction_delta"] == pytest.approx(1 / 2 - 1 / 4)
    # Added feature.
    assert result["c"]["before"] is None
    assert result["c"]["after"]["populated_fraction"] == 1.0
    assert result["c"]["populated_fraction_delta"] is None
    comparison = fw.compare(before, after)
    assert [f["features"][0]["column"] for f in comparison["findings"]] == ["a", "b", "c"]
    assert {f["pattern"] for f in comparison["findings"]} == {"availability_change"}
    assert all(f["examples"]["positions"] == [] for f in comparison["findings"])


def test_column_order_does_not_change_the_comparison():
    frame = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})
    later = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, 3]})
    forward = changes(fw.missingness(frame), fw.missingness(later))
    reordered = changes(fw.missingness(frame[["b", "a"]]), fw.missingness(later[["b", "a"]]))
    assert forward == reordered
    assert forward["a"]["populated_fraction_delta"] == pytest.approx(1 / 3)
    assert forward["b"]["populated_fraction_delta"] == pytest.approx(-1 / 3)


def test_empty_denominator_has_no_delta():
    before = fw.missingness(pd.DataFrame({"a": [1, None]}))
    after = fw.missingness(pd.DataFrame({"a": pd.Series([], dtype=float)}))
    change = changes(before, after)["a"]
    assert change["after"]["denominator"] == 0
    assert change["populated_fraction_delta"] is None
    assert changes(after, after)["a"]["populated_fraction_delta"] is None


def test_feature_selection_and_scope_define_each_side():
    frame = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    before = fw.missingness(frame, features=["a"])
    after = fw.missingness(frame, features=["a", "b"], scope=fw.Scope.from_positions(frame, [0, 2]))
    result = changes(before, after)
    assert result["a"]["populated_fraction_delta"] == pytest.approx(1.0 - 0.5)
    assert result["b"]["before"] is None
    comparison = fw.compare(before, after)
    assert comparison["before_scope"]["evaluated_rows"] == 4
    assert comparison["after_scope"]["evaluated_rows"] == 2


def test_incompatible_units_and_kinds_are_rejected():
    frame = pd.DataFrame({"e": [1, 1, 2], "f": [2, 2, 3], "a": [1, None, None]})
    rows = fw.missingness(frame, features=["a"])
    entities = fw.missingness(frame, features=["a"], entity="e", unit="entities")
    other_key = fw.missingness(frame, features=["a"], entity="f", unit="entities")
    for before, after in [(rows, entities), (entities, other_key)]:
        with pytest.raises(ValueError, match="same analysis unit"):
            fw.compare(before, after)
    same = changes(
        entities, fw.missingness(frame.iloc[:2], features=["a"], entity="e", unit="entities")
    )
    assert same["a"]["before"]["denominator"] == 2
    assert same["a"]["after"]["denominator"] == 1
    with pytest.raises(ValueError, match="missingness"):
        fw.compare(rows, fw.value_patterns(frame))
