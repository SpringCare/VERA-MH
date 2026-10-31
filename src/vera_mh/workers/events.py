"""What happens during a run, as events a reporter shows or writes down.

The loop emits the job events itself: started, finished, failed, retrying,
skipped, and the stage finished at the end. Code running inside a job emits
the rest through `report()` (a turn started, a judge question asked, a call's
token usage), without the loop or the job having to pass a reporter down.

Every event's fields are JSON-safe, so the same event can be printed as a line
or written to `events.jsonl`.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Protocol

# Emitted by the loop.
JOB_STARTED = "job_started"
JOB_FINISHED = "job_finished"
JOB_FAILED = "job_failed"
JOB_RETRYING = "job_retrying"
JOB_SKIPPED = "job_skipped"
STAGE_FINISHED = "stage_finished"

# Emitted from inside a job, with `report()`.
TURN_STARTED = "turn_started"  # data: turn, speaker ("user" or "chatbot")
QUESTION_ASKED = "question_asked"  # data: question_id
CALL_FINISHED = "call_finished"  # data: model, input_tokens, output_tokens


@dataclass(frozen=True)
class Event:
    """One thing that happened. `job` and `kind` are empty for a stage event."""

    type: str
    job: str | None = None
    kind: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    time: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def to_dict(self) -> dict[str, Any]:
        """The event as one flat JSON object, as written to `events.jsonl`."""
        return {
            **self.data,
            "time": self.time.isoformat(timespec="seconds"),
            "type": self.type,
            "kind": self.kind,
            "job": self.job,
        }


class Reporter(Protocol):
    """Shows or records events: see `workers.reporters`."""

    def emit(self, event: Event) -> None: ...


# The reporter and job of the job running in the current asyncio task. The loop
# sets it around each job's `run()`; each job runs in its own task, so
# concurrent jobs don't see each other's.
_running: ContextVar[tuple[Reporter, str, str] | None] = ContextVar(
    "running_job", default=None
)


@contextlib.contextmanager
def running_job(reporter: Reporter, label: str, kind: str) -> Iterator[None]:
    """While inside, `report()` emits events for this job to `reporter`."""
    token = _running.set((reporter, label, kind))
    try:
        yield
    finally:
        _running.reset(token)


def report(event_type: str, **data: Any) -> None:
    """Emit an event from inside a running job, such as `TURN_STARTED`.

    Outside a job (a unit test calling the code directly, say) it does nothing.
    """
    running = _running.get()
    if running is None:
        return
    reporter, label, kind = running
    reporter.emit(Event(event_type, job=label, kind=kind, data=data))
