"""Native pandas dtypes produce the same evidence as their object-dtype equivalents."""

import pandas as pd
import pyarrow as pa
import pytest

import fieldwork as fw

KEY = [1, 1, 2, 2, 3, 3, 4, 4]
STRINGS = ["AB-1", "AB-1", "CD-22", None, "x", "x", "AB-3", "AB-4"]
INTEGERS = [10, 10, 20, None, 30, 30, 40, 50]
TIMES = pd.to_datetime(
    [
        "2024-01-01",
        "2024-01-01",
        None,
        "2024-01-03",
        "2024-01-05",
        "2024-01-05",
        "2024-01-06",
        "2024-01-07",
    ]
)


def native_columns():
    return {
        # An unobserved category must not appear as a zero-count level.
        "categorical": pd.Categorical(
            ["a", "a", "b", "b", None, "c", "c", "a"], categories=["a", "b", "c", "unused"]
        ),
        "categorical_int": pd.Categorical([1, 1, 2, None, 3, 3, 2, 1], categories=[1, 2, 3, 9]),
        "Int64": pd.array(INTEGERS, dtype="Int64"),
        "boolean": pd.array([True, True, False, None, True, True, False, False], dtype="boolean"),
        "string": pd.array(STRINGS, dtype="string"),
        "string_pyarrow": pd.array(STRINGS, dtype="string[pyarrow]"),
        "int64_pyarrow": pd.array(INTEGERS, dtype=pd.ArrowDtype(pa.int64())),
        "datetime_tz": TIMES.tz_localize("Europe/Paris"),
        "timestamp_pyarrow": pd.array(
            TIMES.tz_localize("UTC"), dtype=pd.ArrowDtype(pa.timestamp("ns", tz="UTC"))
        ),
        "float32": pd.array([0.5, 0.5, 1.25, None, 2.0, 2.0, 3.5, 0.1], dtype="float32"),
    }


NATIVE = list(native_columns())
NUMERIC = {"Int64", "int64_pyarrow", "float32"}


def frames(kind):
    native = pd.DataFrame({"key": KEY, "value": native_columns()[kind]})
    return native, native.astype({"value": object})


def analytical(result):
    """Drop source identity and dtypes; everything else must agree."""

    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in {"source", "dataset_id"}}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(result.to_dict())


ANALYSES = {
    "levels": lambda df: fw.levels(df),
    "census": lambda df: fw.census(df, ["key", "value"]),
    "missingness": lambda df: fw.missingness(df),
    "dependencies": lambda df: fw.discover_dependencies(df),
    "paths": lambda df: fw.suggest_paths(df),
}


@pytest.mark.parametrize("kind", NATIVE)
def test_native_dtype_counts_match_object_equivalent(kind):
    native, equivalent = frames(kind)
    assert native["value"].dtype != object
    for name, analysis in ANALYSES.items():
        assert analytical(analysis(native)) == analytical(analysis(equivalent)), name
    levels = fw.levels(native)["per_feature"][1]
    assert levels["levels_total"] == native["value"].nunique(dropna=False)
    availability = fw.missingness(native)["availability"][1]
    assert availability["missing"] == native["value"].isna().sum() == 1


@pytest.mark.parametrize("kind", NATIVE)
def test_native_dtype_value_patterns_match_object_equivalent(kind):
    native, equivalent = frames(kind)
    patterns = analytical(fw.value_patterns(native, by=["key"]))
    baseline = analytical(fw.value_patterns(equivalent, by=["key"]))

    # Numeric summaries require a numeric dtype, so an object column of numbers
    # has none; every other finding and measurement must agree.
    def summary(finding):
        return (
            finding["pattern"] == "numeric_range" and finding["measurements"]["feature"] == "value"
        )

    compared = ("pattern", "features", "measurements", "examples", "exceptions")
    assert [{k: f[k] for k in compared} for f in patterns["findings"] if not summary(f)] == [
        {k: f[k] for k in compared} for f in baseline["findings"]
    ]
    numeric = [f for f in patterns["findings"] if summary(f)]
    assert len(numeric) == (kind in NUMERIC)
    if numeric:
        as_float = native.astype({"value": "float64"})
        expected = next(f for f in fw.value_patterns(as_float)["findings"] if summary(f))
        assert numeric[0]["measurements"] == expected["measurements"]


def test_all_native_dtypes_together_match_object_equivalent():
    native = pd.DataFrame({"key": KEY, **native_columns()})
    equivalent = native.astype(object)
    dimensions = ["categorical", "boolean", "string_pyarrow", "datetime_tz"]
    for analysis in (
        fw.levels,
        fw.missingness,
        fw.suggest_paths,
        lambda df: fw.census(df, dimensions),
        # Single-column keys pair every dtype with every other; composite keys
        # are covered per dtype above.
        lambda df: fw.discover_dependencies(df, max_key_size=1),
    ):
        assert analytical(analysis(native)) == analytical(analysis(equivalent))
