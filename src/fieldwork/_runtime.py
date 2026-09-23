"""Private call-scoped resources. No cache survives a public analysis call."""

from __future__ import annotations

import math
import statistics
import time
from collections import OrderedDict, deque
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
        self.prepared = {}
        self.encodings = OrderedDict()
        self.dictionary_tokens = 0

    def remember_encoding(self, key, value):
        # Scalar identities can dwarf code arrays on wide continuous frames.
        # Bound retained dictionaries by cardinality, independently of row codes.
        old = self.encodings.pop(key, None)
        if old is not None:
            self.dictionary_tokens -= len(old[1])
        if len(value[1]) > 100_000:
            return
        self.encodings[key] = value
        self.dictionary_tokens += len(value[1])
        while self.dictionary_tokens > 100_000:
            _, removed = self.encodings.popitem(last=False)
            self.dictionary_tokens -= len(removed[1])

    def check(self):
        if self.cancel is not None and self.cancel.cancelled:
            raise AnalysisCancelled("Analysis cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise AnalysisCancelled("Analysis deadline exceeded")

    def close(self):
        # Tracebacks or a callback may retain this object after the call exits.
        # Release cached frames/arrays even in that case.
        self.fingerprints.clear()
        self.prepared.clear()
        self.encodings.clear()
        self.dictionary_tokens = 0

    def emit(self, phase, status, *, force=False):
        now = time.monotonic()
        if self.callback is None or (not force and now - self.last_emit < 0.2):
            return
        self.last_emit = now
        eta = None
        rates = list(phase.rates)
        if (
            status == "running"
            and phase.total is not None
            and len(rates) >= 4
            and now - phase.started >= 0.5
            and now - phase.sampled <= 1.0
        ):
            rate = statistics.mean(rates)
            if rate > 0 and statistics.pstdev(rates) / rate < 0.35:
                eta = max(0, phase.total - phase.completed) / rate
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
            eta,
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
        self.started = self.sampled = time.monotonic()
        self.sample_count = 0
        self.rates = deque(maxlen=8)
        self.id = session.next_phase if session else 0
        self.parent = session.stack[-1].id if session and session.stack else None
        if session:
            session.next_phase += 1

    def advance(self, amount=1, *, detail=None):
        # A miscounted estimate is cosmetic; never let it abort the analysis.
        self.completed += amount
        if self.total is not None:
            self.completed = min(self.completed, self.total)
        self.detail = detail
        if self.session:
            self.session.check()
            now = time.monotonic()
            if now - self.sampled >= 0.1:
                self.rates.append((self.completed - self.sample_count) / (now - self.sampled))
                self.sample_count, self.sampled = self.completed, now
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


P = ParamSpec("P")
R = TypeVar("R")


def operation(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Add common runtime controls without putting them in analytical payloads."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def run(*args, progress=None, cancel=None, timeout=None, **kwargs):
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

        return cast(Callable[P, R], run)

    return decorate
