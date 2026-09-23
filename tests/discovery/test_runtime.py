"""Runtime feedback must not change analytical payloads or outlive a call."""

import inspect
import io
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import fieldwork as fw
from fieldwork import _runtime, evidence


def sample():
    return pd.DataFrame({"key": [1, 1, 2, 2], "value": ["a", None, "b", "b"]})


def test_overview_reuses_fingerprint_and_discards_cache(monkeypatch):
    actual = evidence._fingerprint
    calls = []

    def counted(*args):
        calls.append(args[0])
        return actual(*args)

    monkeypatch.setattr(evidence, "_fingerprint", counted)
    frame = sample()
    events = []
    first = fw.explore(frame, progress=events.append)
    assert len(calls) == 1
    assert _runtime.current_session() is None
    assert first.to_dict() == fw.explore(frame).to_dict()
    assert len(calls) == 2
    frame.loc[0, "key"] = 3
    second = fw.explore(frame)
    assert len(calls) == 3
    assert first["source"]["dataset_id"] != second["source"]["dataset_id"]
    by_phase = {}
    for event in events:
        by_phase.setdefault(event.phase_id, []).append(event)
    for phase_events in by_phase.values():
        assert phase_events[0].status == "started"
        assert phase_events[-1].status == "completed"
        assert [e.completed for e in phase_events] == sorted(e.completed for e in phase_events)
        assert all(e.total is None or e.completed <= e.total for e in phase_events)
        parent = phase_events[0].parent_id
        assert parent is None or parent in by_phase
    assert events[-1].parent_id is None
    assert events[-1].operation == "overview"
    assert "progress" in inspect.signature(fw.explore).parameters
    assert not {"progress", "cancel", "timeout"} & first["parameters"].keys()


@pytest.mark.parametrize("control", ["token", "deadline", "keyboard", "callback", "invalid"])
def test_cleanup_and_original_error_on_interruption(control):
    events = []
    token = fw.CancellationToken()

    def callback(event):
        events.append(event)
        if event.phase == "fingerprinting" and event.status == "started":
            if control == "token":
                token.cancel()
            elif control == "keyboard":
                raise KeyboardInterrupt("interrupt")
            elif control == "callback":
                raise LookupError("callback failed")

    error = {
        "token": fw.AnalysisCancelled,
        "deadline": fw.AnalysisCancelled,
        "keyboard": KeyboardInterrupt,
        "callback": LookupError,
        "invalid": KeyError,
    }[control]
    with pytest.raises(error):
        fw.missingness(
            sample(),
            progress=callback,
            cancel=token,
            timeout=0 if control == "deadline" else None,
            features=["unknown"] if control == "invalid" else None,
        )
    assert _runtime.current_session() is None
    if control in {"token", "deadline"}:
        assert events[-1].status == "cancelled"
    elif control == "invalid":
        assert events[-1].status == "failed"
    assert fw.missingness(sample())["status"] == "computed"


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": -1},
        {"timeout": True},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"cancel": object()},
        {"progress": 3},
    ],
)
def test_runtime_validation(options):
    with pytest.raises((TypeError, ValueError)):
        fw.levels(sample(), **options)
    assert _runtime.current_session() is None


def test_throttled_phase_eta_requires_stable_fresh_samples(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(_runtime.time, "monotonic", lambda: now[0])
    events = []

    @_runtime.operation("test")
    def run():
        with _runtime.phase("work", 10, "columns") as phase:
            for i in range(4):
                now[0] += 0.25
                phase.advance()
                assert (events[-1].estimated_remaining_seconds is None) == (i < 3)
            assert events[-1].estimated_remaining_seconds == pytest.approx(1.5)
            count = len(events)
            _runtime.checkpoint()
            assert len(events) == count
            now[0] += 2
            _runtime.checkpoint()
            assert events[-1].estimated_remaining_seconds is None
            now[0] += 0.25
            phase.advance()
            assert events[-1].estimated_remaining_seconds is None

    run(progress=events.append)
    assert events[-1].status == "completed"


def test_terminal_and_notebook_display_escape_and_update(monkeypatch):
    event = fw.ProgressEvent(
        "op", "<work>", 0, None, 3, 10, "items", 2, 1, 4, "\x1b\n<x>", "running"
    )
    stream = io.StringIO()
    fw.ProgressDisplay(stream, notebook=False)(event)
    assert "3/10 items" in stream.getvalue()
    assert "phase ETA ~4s" in stream.getvalue()
    assert "\x1b" not in stream.getvalue()
    displayed, updated = [], []
    handle = SimpleNamespace(update=updated.append)

    def display(content, **kwargs):
        displayed.append(content)
        assert kwargs == {"display_id": True}
        return handle

    monkeypatch.setitem(
        sys.modules, "IPython.display", SimpleNamespace(HTML=lambda x: x, display=display)
    )
    adapter = fw.ProgressDisplay(notebook=True)
    adapter(event)
    adapter(event)
    assert len(displayed) == len(updated) == 1
    assert "&lt;work&gt;" in displayed[0] and 'aria-live="polite"' in displayed[0]


def test_recipe_and_scope_runtime_controls():
    with pytest.raises(ValueError, match="Runtime controls"):
        fw.Recipe("missingness", {"progress": True})
    events = []
    frame = sample()
    scope = fw.Scope.from_positions(frame, [0, 2], progress=events.append)
    assert scope.positions == (0, 2)
    assert events[-1].status == "completed"
    events.clear()
    fw.Recipe("missingness").run(frame, progress=events.append)
    assert events[-1].operation == "recipe"


def test_scalar_dictionary_cache_is_bounded():
    session = _runtime.Session("test", None, None, None)
    frame = sample()
    values = [None] * 60_000
    session.remember_encoding("a", (frame, values, None))
    session.remember_encoding("b", (frame, values, None))
    assert list(session.encodings) == ["b"]
    assert session.dictionary_tokens == 60_000
    session.remember_encoding("huge", (frame, [None] * 100_001, None))
    assert list(session.encodings) == ["b"]
    session.remember_encoding("b", (frame, [None], None))
    assert session.dictionary_tokens == 1


@pytest.mark.parametrize("fail", [False, True])
def test_retained_session_releases_cached_frames_on_exit(fail):
    retained = []

    @_runtime.operation("test")
    def run():
        session = _runtime.current_session()
        retained.append(session)
        evidence.prepare(sample())
        assert session.prepared and session.encodings and session.fingerprints
        if fail:
            raise ValueError("test failure")

    if fail:
        with pytest.raises(ValueError, match="test failure"):
            run()
    else:
        run()
    assert not retained[0].prepared
    assert not retained[0].encodings
    assert not retained[0].fingerprints
    assert retained[0].dictionary_tokens == 0


def test_falsey_callable_progress_is_still_called():
    class Recorder(list):
        __call__ = list.append

    events = Recorder()
    fw.levels(sample(), progress=events)
    assert events[-1].status == "completed"


def test_progress_overrun_is_clamped_not_fatal():
    events = []

    @_runtime.operation("test")
    def run():
        with _runtime.phase("work", 2, "items") as phase:
            for _ in range(5):
                phase.advance()
        return "done"

    assert run(progress=events.append) == "done"
    assert all(e.completed <= e.total for e in events if e.total is not None)
