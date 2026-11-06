"""Reporters: where events go.

- `PlainLineReporter` prints one flushed, timestamped line per event, so output
  piped to `tee` or a CI log is never silent while work is happening. ASCII only.
- `EventsFileReporter` appends each event to `events.jsonl` as one JSON object,
  for scripts and notebooks.

A `JobLoop` takes a list of them and sends every event to each.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import TextIO

from vera_mh.workers.events import VERBOSE_ONLY, WORDS, Event, EventType, Reporter


class EventsFileReporter(Reporter):
    """Appends each event to a JSON Lines file, one object per line."""

    def __init__(self, path: Path):
        self.path = path

    def emit(self, event: Event) -> None:
        # Opened per event: a run makes a few events per model call, so the cost
        # is nothing, and a crash never leaves a buffered event unwritten.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str))
            f.write("\n")


class PlainLineReporter(Reporter):
    """Prints one line per event, flushed at once:

    19:44:06 session     done   gpt-5.2 Brian   0:02:41
    19:45:10 session     FAIL   claude-opus-4-5 Abigail   RateLimitError: ...
    """

    def __init__(self, stream: TextIO | None = None, verbose: bool = False):
        # Looked up when the reporter is made, not when Python reads this file,
        # so a test (or anything else) that swaps sys.stdout still gets the lines.
        self.stream = stream or sys.stdout
        self.verbose = verbose

    def emit(self, event: Event) -> None:
        line = self.format(event)
        if line is not None:
            print(line, file=self.stream, flush=True)

    def format(self, event: Event) -> str | None:
        """The line for `event`, or None if it isn't shown."""
        clock = event.time.strftime("%H:%M:%S")
        if event.type == EventType.STAGE_FINISHED:
            counts = event.data
            return (
                f"{clock} {counts['stage']} finished: {counts['done']} done, "
                f"{counts['failed']} failed, {counts['skipped']} skipped"
            )
        word = WORDS.get(event.type)
        if word is None or (event.type in VERBOSE_ONLY and not self.verbose):
            return None
        detail = _detail(event)
        line = f"{clock} {event.kind or '':<11} {word:<6} {event.job or ''}"
        return f"{line}   {detail}" if detail else line


def _detail(event: Event) -> str:
    data = event.data
    if event.type == EventType.JOB_FINISHED:
        return str(timedelta(seconds=round(data["elapsed"])))
    if event.type == EventType.JOB_FAILED:
        return data["error"]
    if event.type == EventType.CALL_RETRYING:
        attempt = f"{data['attempt']}/{data['max_attempts']}"
        return f"call attempt {attempt} failed: {data['error']}"
    if event.type == EventType.TURN_STARTED:
        return f"turn {data['turn']} {data['speaker']} responding"
    if event.type == EventType.QUESTION_ASKED:
        return f"question {data['question_id']}"
    return ""
