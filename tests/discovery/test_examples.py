import json
import runpy
from pathlib import Path

import pandas as pd


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

    # Keep the narrative's numerical conclusions and companion dataset in sync.
    companion = runpy.run_path(str(path.with_suffix(".py")))
    pd.testing.assert_frame_equal(namespace["df"], companion["sample"]())
    pd.testing.assert_frame_equal(
        namespace["next_delivery"], companion["corrected_delivery"](namespace["df"])
    )
    assert len(namespace["mr_examples"]) == 1
    assert len(namespace["mr_scope"].positions) == 8
    assert namespace["unit_summary"]["populated"].tolist() == [7, 4, 3]
    assert namespace["unit_summary"]["denominator"].tolist() == [8, 4, 4]
    assert namespace["exception_scope"].positions == (8,)
    assert namespace["affected_rows"][["exam_id", "slot"]].values.tolist() == [
        ["S01", 1],
        ["S01", 2],
    ]
    changes = namespace["delivery_changes"].set_index("feature")
    assert changes.loc["image_2", "before.populated"] == 3
    assert changes.loc["image_2", "after.populated"] == 4
    assert changes.loc["image_2", "populated_fraction_delta"] == 0.25
    assert namespace["before_delivery"]["analysis_unit"]["denominator"] == 4
    assert namespace["after_delivery"]["analysis_unit"]["denominator"] == 4
