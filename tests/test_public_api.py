"""Smoke test for the documented interface: exports, their members and Path.

Docstring examples run through ``pytest --doctest-modules`` (configured in
pyproject.toml); ``tests/typing/public_api.py`` is checked by pyright.
"""

import inspect

import fieldwork as fw
from fieldwork.navigation import Path as CensusPath


def public_callables():
    exported = [(name, getattr(fw, name)) for name in fw.__all__] + [("Path", CensusPath)]
    for name, obj in exported:
        yield name, obj
        if inspect.isclass(obj):
            for member_name, member in vars(obj).items():
                if member_name.startswith("_") and member_name != "__call__":
                    continue
                if isinstance(member, classmethod | staticmethod):
                    member = member.__func__
                if isinstance(member, property):
                    member = member.fget
                if inspect.isfunction(member):
                    yield f"{name}.{member_name}", member


def test_public_callables_have_docstrings_and_annotated_public_signatures():
    problems = []
    for name, obj in public_callables():
        target = inspect.unwrap(obj)
        if not (target.__doc__ or "").strip():
            problems.append(f"{name}: no docstring")
        if inspect.isclass(target):
            continue
        signature = inspect.signature(obj)
        for parameter in signature.parameters.values():
            if parameter.name in {"self", "cls"}:
                continue
            if parameter.name.startswith("_"):
                problems.append(f"{name}: private parameter {parameter.name}")
            if parameter.annotation is inspect.Parameter.empty:
                problems.append(f"{name}: unannotated {parameter.name}")
        if signature.return_annotation is inspect.Signature.empty:
            problems.append(f"{name}: no return annotation")
    assert not problems
