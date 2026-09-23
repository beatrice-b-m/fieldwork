import json
import runpy
from pathlib import Path

import pandas as pd
import pytest


def test_worked_example():
    namespace = runpy.run_path(str(Path(__file__).parents[2] / "examples/investigation.py"))
    _, _, exceptions, paths, _ = namespace["investigate"]()
    assert len(exceptions) == 1
    assert paths.best.dimensions.index("site") < paths.best.dimensions.index("exam_id")


def test_notebook_executes():
    path = Path(__file__).parents[2] / "examples/investigation.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)  # noqa: S102 - execute the repository-owned notebook

    # The companion script builds the same deliveries.
    companion = runpy.run_path(str(path.with_suffix(".py")))
    pd.testing.assert_frame_equal(namespace["df"], companion["sample"]())
    pd.testing.assert_frame_equal(
        namespace["next_delivery"], companion["corrected_delivery"](namespace["df"])
    )

    # Each assertion below is a conclusion stated in the narrative.
    # image_2 is populated on 7/12 rows overall; the MR cohort has eight rows.
    overall = namespace["availability"].to_frame("availability").set_index("feature")
    assert (overall.loc["image_2", "populated"], overall.loc["image_2", "denominator"]) == (7, 12)
    assert len(namespace["mr_examples"]) == 1
    assert len(namespace["mr_scope"].positions) == 8
    change = namespace["population_change"].set_index("feature")
    assert change.loc["image_2", "populated_fraction_delta"] == pytest.approx(7 / 8 - 7 / 12)
    # 7/8 slots, 4/4 exams with any second image, 3/4 with one on every slot.
    assert namespace["unit_summary"]["populated"].tolist() == [7, 4, 3]
    assert namespace["unit_summary"]["denominator"].tolist() == [8, 4, 4]
    # The one exception is South / S01 / slot 2 at source position 8.
    assert namespace["exception_scope"].positions == (8,)
    assert namespace["affected_rows"][["exam_id", "slot"]].values.tolist() == [
        ["S01", 1],
        ["S01", 2],
    ]
    # The census evaluates the eight MR rows of 12, restricting out four CT rows.
    population = namespace["population"]
    assert (population["input_rows"], population["evaluated_rows"]) == (12, 8)
    assert (population["restriction_excluded_rows"], population["missing_excluded_rows"]) == (4, 0)
    # The corrected delivery raises image_2 from 3/4 to 4/4 exams (+0.25).
    changes = namespace["delivery_changes"].set_index("feature")
    assert changes.loc["image_2", "before.populated"] == 3
    assert changes.loc["image_2", "after.populated"] == 4
    assert changes.loc["image_2", "populated_fraction_delta"] == 0.25
    assert namespace["before_delivery"]["analysis_unit"]["denominator"] == 4
    assert namespace["after_delivery"]["analysis_unit"]["denominator"] == 4
