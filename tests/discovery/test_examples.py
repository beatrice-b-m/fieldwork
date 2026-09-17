import json
import runpy
from pathlib import Path


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
