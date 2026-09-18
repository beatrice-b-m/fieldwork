"""Editor-level signature, hover, and navigation probes against public imports."""

import inspect
import sys
from pathlib import Path

import jedi
import pytest

import fieldwork as fw


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("editor-project")
    package_root = Path(fw.__file__).resolve().parents[1]
    old_cache = jedi.settings.cache_directory
    jedi.settings.cache_directory = str(tmp_path_factory.mktemp("jedi"))
    yield jedi.Project(
        str(root),
        added_sys_path=[str(package_root)],
        environment_path=sys.executable,
        smart_sys_path=False,
    )
    jedi.settings.cache_directory = old_cache


def script(project, expression):
    return jedi.Script(
        "import fieldwork as fw\n" + expression,
        path=str(Path(project.path) / "editor_probe.py"),
        project=project,
    )


@pytest.mark.parametrize(
    ("expression", "required", "forbidden"),
    [
        ("fw.missingness(", {"unit=", "progress=", "cancel=", "timeout="}, set()),
        ("fw.census(", {"top_n=", "scope=", "timeout="}, {"candidate_keys=", "_encoded="}),
        ("fw.explore(", {"sections=", "candidate_keys=", "scope=", "top_n="}, {"_encoded="}),
        ("fw.grain(", {"candidate_keys=", "progress="}, {"_encoded=", "_cache="}),
        (
            "fw.suggest_paths(None).path().census(",
            {"top_n=", "progress="},
            {"dimensions=", "scope=", "missing=", "table_id=", "candidate_keys="},
        ),
    ],
)
def test_editor_keyword_completion(project, expression, required, forbidden):
    probe = script(project, expression)
    assert probe.get_signatures(), expression
    names = {c.name for c in probe.complete() if c.type == "param"}
    assert required <= names, names
    assert not names & forbidden, names


@pytest.mark.parametrize("name", [n for n in fw.__all__ if inspect.isfunction(getattr(fw, n))])
def test_editor_hover_has_exported_documentation(project, name):
    inferred = script(project, f"fw.{name}").infer()
    docs = [value.docstring() for value in inferred]
    assert any("Parameters\n" in doc and "Returns\n" in doc for doc in docs), (name, docs)


def test_result_method_navigation(project):
    definitions = script(project, "fw.missingness(None).select").goto(follow_imports=True)
    assert any(d.name == "select" and d.module_name == "fieldwork.evidence" for d in definitions)
    definitions = script(project, "fw.suggest_paths(None).path().census").goto(follow_imports=True)
    assert any(d.name == "census" and d.module_name == "fieldwork.navigation" for d in definitions)
