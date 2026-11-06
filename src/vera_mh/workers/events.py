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

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Callable, Protocol


class EventType(StrEnum):
    """Every kind of event. A `StrEnum`, so each one is also its plain string
    value ("job_started"), which is what `events.jsonl` records."""

    # Emitted by the loop, about one job.
    JOB_STARTED = "job_started"
    JOB_FINISHED = "job_finished"  # data: elapsed
    JOB_FAILED = "job_failed"  # data: error
    JOB_SKIPPED = "job_skipped"
    # Emitted by the loop once, when `JobLoop.run()` ends: about the whole run.
    STAGE_FINISHED = "stage_finished"  # data: stage, done, failed, skipped

    # Emitted from inside a job, with `report()`.
    TURN_STARTED = "turn_started"  # data: turn, speaker ("user" or "chatbot")
    QUESTION_ASKED = "question_asked"  # data: question_id
    # LLM calls retry themselves (llm_clients/); the loop never retries a job.
    CALL_RETRYING = "call_retrying"  # data: attempt, max_attempts, error
    CALL_FINISHED = "call_finished"  # data: model, input_tokens, output_tokens


# The word each event shows as on the terminal (`PlainLineReporter`). A call
# retrying is always shown, since it's how a slow or failing provider shows up;
# turns and questions only with `verbose`, as a run has many of them. Token
# usage only goes to `events.jsonl`.
WORDS = {
    EventType.JOB_STARTED: "start",
    EventType.JOB_FINISHED: "done",
    EventType.JOB_FAILED: "FAIL",
    EventType.JOB_SKIPPED: "skip",
    EventType.CALL_RETRYING: "retry",
}
VERBOSE_WORDS = {EventType.TURN_STARTED: "turn", EventType.QUESTION_ASKED: "ask"}


def _now() -> datetime:
    # A named function, not a lambda, so test_run_root's clock rule can allow
    # this one `datetime.now()` call by name.
    return datetime.now().astimezone()


@dataclass(frozen=True)
class Event:
    """One thing that happened.

    `job` and `kind` name the job it's about. `STAGE_FINISHED` is the one event
    that isn't about a job but about the whole run (its counts are in `data`),
    so it has neither.
    """

    type: EventType
    job: str | None = None
    kind: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    time: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """The event as one flat JSON object, as written to `events.jsonl`."""
        return {
            **self.data,
            "time": self.time.isoformat(timespec="seconds"),
            "type": self.type.value,
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
# client would touch all of them. So the loop writes down, at the start of each
# job, which job is running and where its events go, and `report()` reads that
# back:
#
#     # in the loop, at the start of each job
#     set_current_job(loop.emit, "gpt-5.2 Brian", "session")
#     await job.run()
#
#     # anywhere inside that job, however deep
#     report(EventType.CALL_RETRYING, attempt=2, max_attempts=4, error="...")
#
#     # every reporter then receives
#     Event(type="call_retrying", job="gpt-5.2 Brian", kind="session",
#           data={"attempt": 2, "max_attempts": 4, "error": "..."})
#
# Where it's written down is a `ContextVar`: a variable with a separate value in
# each asyncio task. The loop runs every job in its own task, so ten jobs
# running at once each set and read their own value, never another job's, and a
# job's value disappears with its task when the job ends.


@dataclass(frozen=True)
class _CurrentJob:
    emit: Callable[[Event], None]
    label: str
    kind: str


# The running job's `_CurrentJob`, one per asyncio task; None outside a job.
_current_job: ContextVar[_CurrentJob | None] = ContextVar("current_job", default=None)


def set_current_job(emit: Callable[[Event], None], label: str, kind: str) -> None:
    """From now on in this asyncio task, `report()` sends events to `emit` as
    this job's. The loop calls it at the start of each job's task."""
    _current_job.set(_CurrentJob(emit, label, kind))


def report(event_type: EventType, **data: Any) -> None:
    """Emit an event from inside a running job, such as `CALL_RETRYING`.

    Outside a job (a unit test calling the code directly, say) it does nothing.
    """
    job = _current_job.get()
    if job is not None:
        job.emit(Event(event_type, job=job.label, kind=job.kind, data=data))
