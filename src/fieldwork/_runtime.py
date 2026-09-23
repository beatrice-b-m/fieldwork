"""Private call-scoped resources. No cache survives a public analysis call."""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import ParamSpec, TypeVar, cast

from .progress import AnalysisCancelled, CancellationToken, ProgressDisplay, ProgressEvent

_current = ContextVar("fieldwork_analysis", default=None)


class Session:
    def __init__(self, operation, progress, cancel, timeout):
        if progress is True:
            progress = ProgressDisplay()
        if progress is not None and progress is not False and not callable(progress):
            raise TypeError("progress must be True, False, or a callback")
        if cancel is not None and not isinstance(cancel, CancellationToken):
            raise TypeError("cancel must be a CancellationToken")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be finite nonnegative seconds")
        self.operation = operation
        self.callback = None if progress is None or progress is False else progress
        self.cancel = cancel
        self.started = time.monotonic()
        self.deadline = self.started + timeout if timeout is not None else None
        self.stack = []
        self.next_phase = 0
        self.last_emit = -math.inf
        self.fingerprints = {}
        self.labelled = {}
        self.prepared = {}

    def check(self):
        if self.cancel is not None and self.cancel.cancelled:
            raise AnalysisCancelled("Analysis cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise AnalysisCancelled("Analysis deadline exceeded")

    def close(self):
        # Tracebacks or a callback may retain this object after the call exits.
        # Release cached frames/arrays even in that case.
        self.fingerprints.clear()
        self.labelled.clear()
        self.prepared.clear()

    def emit(self, phase, status, *, force=False):
        now = time.monotonic()
        if self.callback is None or (not force and now - self.last_emit < 0.2):
            return
        self.last_emit = now
        event = ProgressEvent(
            self.operation,
            phase.name,
            phase.id,
            phase.parent,
            phase.completed,
            phase.total,
            phase.unit,
            now - self.started,
            now - phase.started,
            phase.detail,
            status,
        )
        try:
            self.callback(event)
        except BaseException:
            # Preserve the original callback error instead of calling it again
            # while unwinding every nested phase.
            self.callback = None
            raise


class Phase:
    def __init__(self, session, name, total, unit):
        self.session = session
        self.name, self.total, self.unit = name, total, unit
        self.completed, self.detail = 0, None
        self.started = time.monotonic()
        self.id = session.next_phase if session else 0
        self.parent = session.stack[-1].id if session and session.stack else None
        if session:
            session.next_phase += 1

    def advance(self, amount=1, *, detail=None):
        # A miscounted total is cosmetic; never let it abort the analysis.
        self.completed += amount
        if self.total is not None:
            self.completed = min(self.completed, self.total)
        self.detail = detail
        if self.session:
            self.session.check()
            self.session.emit(self, "running")


@contextmanager
def phase(name, total=None, unit="items"):
    session = _current.get()
    item = Phase(session, name, total, unit)
    if session:
        session.stack.append(item)
    try:
        if session:
            session.emit(item, "started", force=True)
            session.check()
        yield item
        if session:
            session.check()
            session.emit(item, "completed", force=True)
    except (AnalysisCancelled, KeyboardInterrupt):
        if session:
            session.emit(item, "cancelled", force=True)
        raise
    except BaseException:
        if session:
            session.emit(item, "failed", force=True)
        raise
    finally:
        if session:
            session.stack.pop()


def checkpoint():
    session = _current.get()
    if session:
        session.check()
        if session.stack:
            session.emit(session.stack[-1], "running")


def current_session():
    return _current.get()


_CONTROLS = {
    "progress": "Progress",
    "cancel": "CancellationToken | None",
    "timeout": "float | None",
}
P = ParamSpec("P")
R = TypeVar("R")


def operation(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Add common runtime controls without putting them in analytical payloads."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        # Signatures declare the controls once, as ``**runtime: Unpack[Runtime]``.
        # That var-keyword must not let misspelled options through silently.
        signature = inspect.signature(function)
        parameters = list(signature.parameters.values())
        strict = any(p.kind is p.VAR_KEYWORD and p.name == "runtime" for p in parameters)
        named = {p.name for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        # Introspection (help, IPython, editors without a type checker) lists the
        # controls as ordinary keyword-only parameters.
        parameters = [p for p in parameters if not (strict and p.kind is p.VAR_KEYWORD)]
        position = next(
            (i for i, p in enumerate(parameters) if p.kind is p.VAR_KEYWORD), len(parameters)
        )
        parameters[position:position] = [
            inspect.Parameter(
                control, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=kind
            )
            for control, kind in _CONTROLS.items()
        ]

        @wraps(function)
        def run(*args, progress=None, cancel=None, timeout=None, **kwargs):
            unknown = sorted(kwargs.keys() - named) if strict else []
            if unknown:
                raise TypeError(
                    f"{function.__name__}() got an unexpected keyword argument {unknown[0]!r}"
                )
            parent = _current.get()
            token = None
            if parent is None:
                owned = Session(name, progress, cancel, timeout)
                token = _current.set(owned)
            elif progress is not None or cancel is not None or timeout is not None:
                # An explicitly controlled reentrant call gets its own lifetime.
                owned = Session(name, progress, cancel, timeout)
                token = _current.set(owned)
            try:
                with phase(name):
                    return function(*args, **kwargs)
            finally:
                if token is not None:
                    owned.close()
                    _current.reset(token)

        run.__signature__ = signature.replace(parameters=parameters)  # type: ignore[attr-defined]
        return cast(Callable[P, R], run)

    return decorate
