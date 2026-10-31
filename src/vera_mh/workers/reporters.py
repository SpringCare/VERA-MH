"""Reporters: where events go.

- `PlainLineReporter` prints one flushed, timestamped line per event, so output
  piped to `tee` or a CI log is never silent while work is happening. ASCII only.
- `EventsFileReporter` appends each event to `events.jsonl` as one JSON object,
  for scripts and notebooks.
- `Reporters` sends each event to several reporters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TextIO

from vera_mh.workers.events import (
    JOB_FAILED,
    JOB_FINISHED,
    JOB_RETRYING,
    JOB_SKIPPED,
    JOB_STARTED,
    QUESTION_ASKED,
    STAGE_FINISHED,
    TURN_STARTED,
    Event,
    Reporter,
)


class Reporters:
    """Sends each event to every reporter given, in order."""

    def __init__(self, *reporters: Reporter):
        self.reporters = reporters

    def emit(self, event: Event) -> None:
        for reporter in self.reporters:
            reporter.emit(event)


class EventsFileReporter:
    """Appends each event to a JSON Lines file, one object per line."""

    def __init__(self, path: Path):
        self.path = path

    def emit(self, event: Event) -> None:
        # Opened per event: a run makes a few events per model call, so the cost
        # is nothing, and a crash never leaves a buffered event unwritten.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str))
            f.write("\n")


# The word each job event shows as, padded so the labels line up.
JOB_WORDS = {
    JOB_STARTED: "start",
    JOB_FINISHED: "done",
    JOB_FAILED: "FAIL",
    JOB_RETRYING: "retry",
    JOB_SKIPPED: "skip",
}
# In-job events are shown only with `verbose`: a run has many of them.
IN_JOB_WORDS = {TURN_STARTED: "turn", QUESTION_ASKED: "ask"}


class PlainLineReporter:
    """Prints one line per event, flushed at once:

    19:44:06 session     done   gpt-5.2 Brian   2m41s
    19:45:10 session     FAIL   claude-opus-4-5 Abigail   RateLimitError: ...
    """

    def __init__(self, stream: TextIO | None = None, verbose: bool = False):
        self.stream = stream or sys.stdout
        self.verbose = verbose

    def emit(self, event: Event) -> None:
        line = self.format(event)
        if line is not None:
            print(line, file=self.stream, flush=True)

    def format(self, event: Event) -> str | None:
        """The line for `event`, or None if it isn't shown."""
        clock = event.time.strftime("%H:%M:%S")
        if event.type == STAGE_FINISHED:
            counts = event.data
            return (
                f"{clock} {counts.get('stage', '')} finished: "
                f"{counts.get('done', 0)} done, {counts.get('failed', 0)} failed, "
                f"{counts.get('skipped', 0)} skipped"
            )
        if event.type in JOB_WORDS:
            word = JOB_WORDS[event.type]
        elif self.verbose and event.type in IN_JOB_WORDS:
            word = IN_JOB_WORDS[event.type]
        else:
            return None
        detail = _detail(event)
        line = f"{clock} {event.kind or '':<11} {word:<6} {event.job or ''}"
        return f"{line}   {detail}" if detail else line


def _detail(event: Event) -> str:
    data = event.data
    if event.type == JOB_FINISHED:
        return duration(data.get("seconds", 0))
    if event.type == JOB_FAILED:
        return str(data.get("error", ""))
    if event.type == JOB_RETRYING:
        return f"attempt {data.get('attempt')} failed: {data.get('error', '')}"
    if event.type == TURN_STARTED:
        return f"turn {data.get('turn')} {data.get('speaker')} responding"
    if event.type == QUESTION_ASKED:
        return f"question {data.get('question_id')}"
    return ""


def duration(seconds: float) -> str:
    """`45s`, `2m41s`, `1h02m`."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
