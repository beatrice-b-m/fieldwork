"""Dependency-free progress events, terminal/notebook display, and cancellation."""

from __future__ import annotations

import html
import sys
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Literal, TextIO, TypeAlias


class AnalysisCancelled(RuntimeError):
    """Raised when cancellation or a timeout stops an analysis; no partial result.

    Checks run between work items, so a running pandas/NumPy step finishes
    first. Exceptions from a progress callback propagate unchanged instead.

    Examples
    --------
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> token = fw.CancellationToken()
    >>> token.cancel()
    >>> try:
    ...     fw.levels(pd.DataFrame({'x': [1]}), cancel=token)
    ... except fw.AnalysisCancelled:
    ...     print('cancelled')
    cancelled
    """


class CancellationToken:
    """Request cooperative cancellation from a thread or progress callback.

    Pass it as an analysis's ``cancel``. Thread-safe and one-way: create a fresh
    token for a later run.

    Examples
    --------
    >>> import fieldwork as fw
    >>> token = fw.CancellationToken()
    >>> token.cancelled
    False
    >>> token.cancel()
    >>> token.cancelled
    True
    """

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        """Request cancellation at the next checkpoint; repeated calls are harmless."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancel has been called (it stays True)."""
        return self._event.is_set()


@dataclass(frozen=True)
class ProgressEvent:
    """An immutable snapshot of one analysis phase.

    Parameters
    ----------
    operation : str
        Descriptive public operation name; not a stable enum.
    phase : str
        Descriptive current phase label; use IDs, not labels, for hierarchy.
    phase_id : int
        Phase instance identifier, unique within this call.
    parent_id : int or None
        Parent phase ID; None marks the root operation.
    completed : int
        Work items completed in this phase, in unit units.
    total : int or None
        Phase work total when known; None means unknown, not zero.
    unit : str
        Work-unit label such as rows, columns, tests, or items.
    elapsed_seconds : float
        Monotonic seconds since the outer controlled call began.
    phase_elapsed_seconds : float
        Monotonic seconds since this phase began.
    detail : str or None
        Optional current-work description, which may include feature names.
    status : {'started', 'running', 'completed', 'cancelled', 'failed'}
        Phase state. Running updates are throttled; phase boundaries are emitted.

    Notes
    -----
    Callbacks receive events synchronously; running updates are throttled to five
    per second, phase boundaries always reported. Nested phases overlap, so their
    durations do not sum. See docs/performance.md.
    """

    operation: str
    phase: str
    phase_id: int
    parent_id: int | None
    completed: int
    total: int | None
    unit: str
    elapsed_seconds: float
    phase_elapsed_seconds: float
    detail: str | None
    status: Literal["started", "running", "completed", "cancelled", "failed"]


Progress: TypeAlias = bool | Callable[[ProgressEvent], None] | None
"""Silent mode (None/False), the built-in display (True), or an event callback."""


class ProgressDisplay:
    """Display progress in a terminal, notebook, or redirected stream.

    Parameters
    ----------
    stream : text stream or None, optional
        Writable stream with write and flush; default None uses sys.stderr.
        Interactive terminals update one line; other streams receive one line
        per event. Notebook output uses IPython instead of the stream.
    notebook : bool or None, optional
        Default None detects an IPython ZMQ notebook on the first event. True
        explicitly uses IPython display; False forces stream output.

    Pass an instance as ``progress`` (``progress=True`` uses a default one).
    Terminal controls and HTML in labels are escaped.

    Examples
    --------
    >>> import io
    >>> import pandas as pd
    >>> import fieldwork as fw
    >>> stream = io.StringIO()
    >>> result = fw.levels(pd.DataFrame({'x': [1]}),
    ...                    progress=fw.ProgressDisplay(stream, notebook=False))
    >>> 'completed' in stream.getvalue()
    True
    """

    def __init__(self, stream: TextIO | None = None, *, notebook: bool | None = None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self._handle = None
        self._notebook = notebook
        self._width = 0

    def __call__(self, event: ProgressEvent) -> None:
        """Display one event (write a line, or update the notebook display).

        Raises
        ------
        ImportError
            Explicit notebook mode requires IPython.
        """
        if self._notebook is None:
            try:
                from IPython import get_ipython

                self._notebook = type(get_ipython()).__name__ == "ZMQInteractiveShell"
            except ImportError:
                self._notebook = False
        count = ""
        if event.total is None and event.completed:
            count = f" · {event.completed} {event.unit}"
        if event.total is not None:
            fraction = event.completed / event.total if event.total else 1
            filled = min(20, int(20 * fraction))
            bar = f"[{'#' * filled}{'-' * (20 - filled)}] {fraction:.0%}"
            count = f" {bar} · {event.completed}/{event.total} {event.unit}"
        # Never allow feature names or callback details to inject terminal controls.
        detail = f" · {event.detail}" if event.detail else ""
        text = (
            f"{event.phase}{count} · {event.elapsed_seconds:.1f}s elapsed{detail} · {event.status}"
        )
        text = "".join(c if c.isprintable() else " " for c in text)
        terminal = event.parent_id is None and event.status in {"completed", "cancelled", "failed"}
        if self._notebook:
            from IPython.display import HTML, display

            content = HTML(
                f'<div role="status" aria-live="polite"><code>{html.escape(text)}</code></div>'
            )
            if self._handle is None:
                self._handle = display(content, display_id=True)
            else:
                self._handle.update(content)
        elif getattr(self.stream, "isatty", lambda: False)():
            self.stream.write("\r" + text.ljust(self._width) + ("\n" if terminal else ""))
            self._width = len(text)
            self.stream.flush()
        else:
            self.stream.write(text + "\n")
            self.stream.flush()
