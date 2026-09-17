"""Dependency-free progress events, terminal/notebook display, and cancellation."""

from __future__ import annotations

import html
import sys
from dataclasses import dataclass
from threading import Event


class AnalysisCancelled(RuntimeError):
    """Analysis stopped cooperatively; no apparently complete result is returned."""


class CancellationToken:
    """May be cancelled from another thread or a progress callback."""

    def __init__(self):
        self._event = Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()


@dataclass(frozen=True)
class ProgressEvent:
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
    status: str


class ProgressDisplay:
    """One updating display in a terminal or IPython notebook.

    A non-interactive stream receives newline-delimited updates. Custom callbacks
    can instead consume ProgressEvent objects without any display dependencies.
    """

    def __init__(self, stream=None, *, notebook=None):
        self.stream = stream if stream is not None else sys.stderr
        self._handle = None
        self._notebook = notebook
        self._width = 0

    def __call__(self, event):
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
