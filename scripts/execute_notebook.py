"""Refresh the worked notebook's saved outputs using the existing IPython dependency.

The notebook uses explicit display/print calls for its outputs. Execute every cell
in one fresh shell and write only after all cells succeed; no Jupyter server or
additional notebook packages are required.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config


def execute():
    path = Path(__file__).resolve().parents[1] / "examples/investigation.ipynb"
    notebook = json.loads(path.read_text())
    with TemporaryDirectory() as directory:
        config = Config({"HistoryManager": {"hist_file": ":memory:"}})
        shell = InteractiveShell.instance(ipython_dir=directory, config=config)
        execution_count = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            with capture_output() as captured:
                shell.run_cell("".join(cell["source"]), store_history=False).raise_error()
            outputs = [
                {"output_type": "stream", "name": name, "text": content.splitlines(True)}
                for name, content in [("stdout", captured.stdout), ("stderr", captured.stderr)]
                if content
            ]
            outputs.extend(
                {"output_type": "display_data", "data": output.data, "metadata": output.metadata}
                for output in captured.outputs
            )
            cell.update(execution_count=execution_count, outputs=outputs)
        InteractiveShell.clear_instance()
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Executed {execution_count} cells and saved outputs to {path}")


if __name__ == "__main__":
    execute()
