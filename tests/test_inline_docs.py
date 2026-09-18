"""Coverage and executable examples for the supported editor-facing API."""

import ast
import doctest
import inspect
import re
from pathlib import Path

import pytest

import fieldwork as fw
from fieldwork import typing as api_types
from fieldwork.navigation import Path as CensusPath


def public_objects():
    for name in fw.__all__:
        obj = getattr(fw, name)
        yield name, obj
        if inspect.isclass(obj):
            yield from public_members(name, obj)
    yield "Path", CensusPath
    yield from public_members("Path", CensusPath)


def public_members(prefix, cls):
    for name, member in vars(cls).items():
        if name.startswith("_") and name != "__call__":
            continue
        if isinstance(member, classmethod | staticmethod):
            member = member.__func__
        if isinstance(member, property):
            member = member.fget
        if inspect.isfunction(member):
            yield f"{prefix}.{name}", member


OBJECTS = list(public_objects())


def doc_parameters(doc):
    match = re.search(r"\nParameters\n-+\n(.*?)(?=\n\w[^\n]*\n-+\n|\Z)", doc, re.DOTALL)
    if not match:
        return set()
    return {
        part.strip().lstrip("*")
        for line in match.group(1).splitlines()
        if line and not line[0].isspace() and " : " in line
        for part in line.split(" : ", 1)[0].split(",")
    }


@pytest.mark.parametrize(("name", "obj"), OBJECTS, ids=[n for n, _ in OBJECTS])
def test_public_documentation_and_annotations(name, obj):
    target = inspect.unwrap(obj)
    # __doc__ must be supplied directly: inspect.getdoc can silently inherit a
    # parent method's description or a generated dataclass constructor string.
    doc = inspect.cleandoc(target.__doc__ or "")
    assert len(doc.splitlines()) > 2, name
    assert "\n" in doc and not doc.startswith(f"{target.__name__}("), name
    if inspect.isclass(target):
        if name not in {"AnalysisCancelled", "CancellationToken"}:
            assert "Parameters\n----------" in doc, name
        return
    assert "Returns\n-------" in doc, name
    parameters = {
        key: value
        for key, value in inspect.signature(target).parameters.items()
        if key not in {"self", "cls"}
    }
    assert doc_parameters(doc) == parameters.keys(), name
    assert all(p.annotation is not inspect.Parameter.empty for p in parameters.values()), name
    assert inspect.signature(target).return_annotation is not inspect.Signature.empty, name


@pytest.mark.parametrize(("name", "obj"), OBJECTS, ids=[n for n, _ in OBJECTS])
def test_public_examples(name, obj):
    parser = doctest.DocTestParser()
    test = parser.get_doctest(inspect.unwrap(obj).__doc__ or "", {}, name, "<docstring>", 0)
    runner = doctest.DocTestRunner()
    runner.run(test)
    assert runner.failures == 0, name


def test_option_dictionary_fields_have_static_documentation():
    source = Path(inspect.getfile(api_types)).read_text()
    tree = ast.parse(source)
    documented = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in api_types.__all__:
            continue
        assert ast.get_docstring(node), node.name
        documented.add(node.name)
        for index, field in enumerate(node.body):
            if isinstance(field, ast.AnnAssign):
                following = node.body[index + 1]
                assert isinstance(following, ast.Expr), (node.name, field.target)
                assert isinstance(following.value, ast.Constant), node.name
                assert isinstance(following.value.value, str) and following.value.value, node.name
    assert documented


def test_runtime_controls_exist_in_source_signatures():
    for name, obj in OBJECTS:
        if inspect.isclass(obj):
            continue
        wrapped = inspect.signature(obj)
        if "progress" in wrapped.parameters:
            original = inspect.signature(inspect.unwrap(obj))
            assert wrapped == original, name
            for key in ("progress", "cancel", "timeout"):
                assert original.parameters[key].kind == inspect.Parameter.KEYWORD_ONLY, name
    for obj in (fw.census, fw.grain, CensusPath.census):
        signature = inspect.signature(obj)
        assert all(not n.startswith("_") for n in signature.parameters)
        assert not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )


def test_explore_overloads_document_forwarded_keywords_and_defaults():
    from typing import get_overloads

    from fieldwork._explore.orchestration import explore as foundation_explore

    doc = inspect.getdoc(fw.explore)
    primary = doc_parameters(doc)
    # Forwarded keywords are a distinct NumPy section because the runtime
    # implementation intentionally retains **options and its duplicate rules.
    other = doc.split("Other Parameters\n----------------\n", 1)[1]
    forwarded = doc_parameters("\nParameters\n----------\n" + other)
    overloads = get_overloads(fw.explore)
    for overload in overloads:
        parameters = inspect.signature(overload).parameters
        assert parameters.keys() <= primary | forwarded
        assert all(p.annotation is not inspect.Parameter.empty for p in parameters.values())
    explicit = inspect.signature(overloads[1]).parameters
    for name, parameter in inspect.signature(foundation_explore).parameters.items():
        if name not in {"df", "dimensions"}:
            assert explicit[name].default == parameter.default, name
