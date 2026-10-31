"""Reporters: the plain lines a person reads, and the events.jsonl scripts read."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

from vera_mh.workers import events
from vera_mh.workers.events import Event
from vera_mh.workers.reporters import (
    EventsFileReporter,
    PlainLineReporter,
    Reporters,
    duration,
)

AT = datetime(2026, 10, 30, 19, 44, 6, tzinfo=timezone.utc)


def job_event(event_type: str, **data) -> Event:
    return Event(event_type, job="gpt-5.2 Brian", kind="session", data=data, time=AT)


def test_plain_lines_show_each_job_event() -> None:
    reporter = PlainLineReporter(io.StringIO())

    started = reporter.format(job_event(events.JOB_STARTED))
    finished = reporter.format(job_event(events.JOB_FINISHED, seconds=161))
    failed = reporter.format(
        job_event(events.JOB_FAILED, error="RateLimitError: slow down")
    )
    skipped = reporter.format(job_event(events.JOB_SKIPPED))

    assert started == "19:44:06 session     start  gpt-5.2 Brian"
    assert finished == "19:44:06 session     done   gpt-5.2 Brian   2m41s"
    assert failed == (
        "19:44:06 session     FAIL   gpt-5.2 Brian   RateLimitError: slow down"
    )
    assert skipped == "19:44:06 session     skip   gpt-5.2 Brian"


def test_the_stage_line_counts_outcomes() -> None:
    reporter = PlainLineReporter(io.StringIO())
    stage = Event(
        events.STAGE_FINISHED,
        data={"stage": "generate", "done": 196, "failed": 2, "skipped": 0},
        time=AT,
    )

    assert reporter.format(stage) == (
        "19:44:06 generate finished: 196 done, 2 failed, 0 skipped"
    )


def test_in_job_events_show_only_when_verbose() -> None:
    turn = job_event(events.TURN_STARTED, turn=14, speaker="chatbot")
    tokens = job_event(events.CALL_FINISHED, input_tokens=10, output_tokens=5)

    assert PlainLineReporter(io.StringIO()).format(turn) is None
    verbose = PlainLineReporter(io.StringIO(), verbose=True)
    assert verbose.format(turn) == (
        "19:44:06 session     turn   gpt-5.2 Brian   turn 14 chatbot responding"
    )
    # Token usage goes to events.jsonl, not the screen.
    assert verbose.format(tokens) is None


def test_plain_lines_are_written_and_ascii() -> None:
    stream = io.StringIO()
    reporter = PlainLineReporter(stream)

    reporter.emit(job_event(events.JOB_STARTED))
    reporter.emit(job_event(events.TURN_STARTED, turn=1, speaker="user"))
    reporter.emit(job_event(events.JOB_FINISHED, seconds=45))

    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert all(line.isascii() for line in lines)


def test_events_file_has_one_json_object_per_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    reporter = EventsFileReporter(path)

    reporter.emit(job_event(events.JOB_STARTED))
    reporter.emit(job_event(events.CALL_FINISHED, input_tokens=10, output_tokens=5))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {
            "time": "2026-10-30T19:44:06+00:00",
            "type": "job_started",
            "kind": "session",
            "job": "gpt-5.2 Brian",
        },
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "time": "2026-10-30T19:44:06+00:00",
            "type": "call_finished",
            "kind": "session",
            "job": "gpt-5.2 Brian",
        },
    ]


def test_reporters_send_each_event_to_all() -> None:
    first, second = io.StringIO(), io.StringIO()
    reporter = Reporters(PlainLineReporter(first), PlainLineReporter(second))

    reporter.emit(job_event(events.JOB_STARTED))

    assert first.getvalue() == second.getvalue() != ""


def test_durations() -> None:
    assert duration(45) == "45s"
    assert duration(161) == "2m41s"
    assert duration(3720) == "1h02m"
