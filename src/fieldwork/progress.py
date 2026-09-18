"""Dependency-free progress events, terminal/notebook display, and cancellation."""

from __future__ import annotations

import html
import sys
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Literal, TextIO, TypeAlias


class AnalysisCancelled(RuntimeError):
    """Signal cooperative cancellation or timeout without a partial result.

    Notes
    -----
    Raised when a CancellationToken is cancelled or a finite nonnegative timeout
    expires at a checkpoint. A running pandas/NumPy work item must return first;
    timeout is not a hard process deadline. This RuntimeError subclass is distinct
    from KeyboardInterrupt and from exceptions raised by a progress callback,
    which propagate unchanged.

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

    Attributes
    ----------
    cancelled : bool
        Whether cancel has been called. Initially False; cancellation is sticky.

    Notes
    -----
    Thread-safe and one-way: create a fresh token for a later uncancelled run.
    Pass this object as an analysis's cancel argument. Cancellation is checked
    between work items, raises AnalysisCancelled, and returns no partial result.

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
        """Request cancellation at the next analytical checkpoint.

        Returns
        -------
        None
            Sets the thread-safe flag permanently. Repeated calls are harmless.
        """
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Report whether cancellation has been requested.

        Returns
        -------
        bool
            True after cancel has been called, even after a previous analysis ends.
        """
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
    estimated_remaining_seconds : float or None
        Current-phase ETA only; None until throughput is sufficiently sampled,
        stable, and fresh. Never a guaranteed whole-operation duration.
    detail : str or None
        Optional current-work description, which may include feature names.
    status : {'started', 'running', 'completed', 'cancelled', 'failed'}
        Phase state. Running updates are throttled; phase boundaries are emitted.

    Attributes
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
    estimated_remaining_seconds : float or None
        Current-phase ETA only; None until throughput is sufficiently sampled,
        stable, and fresh. Never a guaranteed whole-operation duration.
    detail : str or None
        Optional current-work description, which may include feature names.
    status : {'started', 'running', 'completed', 'cancelled', 'failed'}
        Phase state. Running updates are throttled; phase boundaries are emitted.

    Notes
    -----
    Callbacks receive these records synchronously. Running updates are throttled
    to at most five per second across a call, while phase boundaries are always
    reported. Nested phases overlap: their durations must not be summed as total
    runtime. IDs are local to one call and labels may change between releases.
    ETA requires at least four rate samples, half a second of observation, stable
    throughput, and a fresh sample. It may disappear when those conditions fail.
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
    estimated_remaining_seconds: float | None
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

    Notes
    -----
    Pass an instance as progress to an analysis, or use progress=True for the
    default adapter. The callable accepts one ProgressEvent and returns None.
    HTML and terminal controls in labels/details are escaped. ETA is for the
    current phase, not the entire analysis. Explicit notebook mode requires
    IPython; stream mode has no optional display dependencies.

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
        """Display one event using the selected output mode.

        Parameters
        ----------
        event : ProgressEvent
            Current phase snapshot supplied by an analysis callback.

        Returns
        -------
        None
            Writes/flushed text or updates one notebook display.

        Raises
        ------
        ImportError
            Explicit notebook mode requires IPython.
        OSError
            The output stream cannot be written or flushed.
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
            count = f" [{'#' * filled}{'-' * (20 - filled)}] {event.completed}/{event.total} {event.unit}"
        eta = (
            f" · phase ETA ~{event.estimated_remaining_seconds:.0f}s"
            if event.estimated_remaining_seconds is not None
            else ""
        )
        # Never allow feature names or callback details to inject terminal controls.
        detail = f" · {event.detail}" if event.detail else ""
        text = f"{event.phase}{count} · {event.elapsed_seconds:.1f}s elapsed{eta}{detail} · {event.status}"
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
