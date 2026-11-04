"""What happens during a run, as events a reporter shows or writes down.

The loop emits the job events itself: started, finished, failed, skipped, and
the stage finished at the end. Code running inside a job emits the rest through
`report()` (a turn started, a judge question asked, an LLM call retrying, a
call's token usage), without the loop or the job having to pass a reporter
down.

Every event's fields are JSON-safe, so the same event can be printed as a line
or written to `events.jsonl`.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterator, Protocol

# Emitted by the loop.
JOB_STARTED = "job_started"
JOB_FINISHED = "job_finished"
JOB_FAILED = "job_failed"
JOB_SKIPPED = "job_skipped"
STAGE_FINISHED = "stage_finished"

# Emitted from inside a job, with `report()`.
TURN_STARTED = "turn_started"  # data: turn, speaker ("user" or "chatbot")
QUESTION_ASKED = "question_asked"  # data: question_id
# LLM calls retry themselves (llm_clients/); the loop never retries a job.
CALL_RETRYING = "call_retrying"  # data: attempt, max_attempts, error
CALL_FINISHED = "call_finished"  # data: model, input_tokens, output_tokens


def _now() -> datetime:
    return datetime.now().astimezone()


@dataclass(frozen=True)
class Event:
    """One thing that happened. `job` and `kind` are empty for a stage event."""

    type: str
    job: str | None = None
    kind: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    time: datetime = field(default_factory=_now)

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


# Reporting from inside a job
#
# Code deep inside a job, such as an LLM client retrying a call, has no handle
# on the loop, and passing one down through the simulator, the judge, and every
# client would touch all of them. Instead the loop records which job the
# current asyncio task is running, and `report()` reads it back:
#
#     # in the loop, around each job (each job runs in its own task)
#     with reporting_for(loop.emit, "gpt-5.2 Brian", "session"):
#         await job.run()
#
#     # anywhere inside that job, however deep
#     report(CALL_RETRYING, attempt=2, max_attempts=4, error="RateLimitError")
#
#     # every reporter then receives
#     Event(type="call_retrying", job="gpt-5.2 Brian", kind="session",
#           data={"attempt": 2, "max_attempts": 4, "error": "RateLimitError"})
#
# A `ContextVar` is a variable whose value is per asyncio task, so ten jobs
# running at once each see their own job, never another's.


@dataclass(frozen=True)
class _CurrentJob:
    emit: Callable[[Event], None]
    label: str
    kind: str


_current_job: ContextVar[_CurrentJob | None] = ContextVar("current_job", default=None)


@contextlib.contextmanager
def reporting_for(
    emit: Callable[[Event], None], label: str, kind: str
) -> Iterator[None]:
    """Inside this block, `report()` sends events to `emit`, as this job's."""
    token = _current_job.set(_CurrentJob(emit, label, kind))
    try:
        yield
    finally:
        _current_job.reset(token)


def report(event_type: str, **data: Any) -> None:
    """Emit an event from inside a running job, such as `CALL_RETRYING`.

    Outside a job (a unit test calling the code directly, say) it does nothing.
    """
    job = _current_job.get()
    if job is not None:
        job.emit(Event(event_type, job=job.label, kind=job.kind, data=data))
