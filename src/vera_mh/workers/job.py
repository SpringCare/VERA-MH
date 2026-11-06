"""What a job is, what can happen to one, and how it reports from inside.

`JobLoop` (workers/loop.py) runs jobs; generation and judging define them: one
session of a conversation, or one evaluation.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Protocol

from vera_mh.workers.events import Event, EventType


class Status(StrEnum):
    """What happened to a job."""

    FINISHED = "finished"  # it ran and wrote its output
    FAILED = "failed"  # it raised; rerun it later with --into
    SKIPPED = "skipped"  # its output was already on disk


class Job(Protocol):
    """One unit of work: one session of a conversation, or one evaluation.

    Keep jobs plain data, with JSON-safe fields, so they could be sent to an
    external queue one day (job-loop.md, decision 7).
    """

    kind: str  # "session" or "evaluation": picks the concurrency cap
    label: str  # how the job is shown, such as "gpt-5.2 Brian"

    def done(self) -> bool:
        """Is this job's output already on disk?"""
        ...

    async def run(self) -> None:
        """Do the work and write the output."""
        ...

    def follow_ups(self) -> list[Job]:
        """The jobs waiting on this one, worked out from its fields and disk
        alone, so a job skipped as done hands them on too."""
        ...


@dataclass
class Outcome:
    """What happened to one job."""

    job: Job
    status: Status
    error: BaseException | None = None
    elapsed: float = 0.0  # seconds the job ran; 0 if it was skipped


# Reporting from inside a job
#
# Code deep inside a job, such as an LLM client retrying a call, has no handle
# on the loop, and passing one down through the simulator, the judge, and every
# client would touch all of them. So the loop writes down, at the start of each
# job, which job is running and where its events go, and `report()` reads that
# back:
#
#     # in the loop, at the start of each job
#     set_current_job(job, loop.emit)
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
    job: Job
    emit: Callable[[Event], None]  # where its events go: `JobLoop.emit`


# The running job, one per asyncio task; None outside a job.
_current_job: ContextVar[_CurrentJob | None] = ContextVar("current_job", default=None)


def set_current_job(job: Job, emit: Callable[[Event], None]) -> None:
    """From now on in this asyncio task, `report()` sends events to `emit` as
    `job`'s. The loop calls it at the start of each job's task."""
    _current_job.set(_CurrentJob(job, emit))


def report(event_type: EventType, **data: Any) -> None:
    """Emit an event from inside a running job, such as `CALL_RETRYING`.

    Outside a job (a unit test calling the code directly, say) it does nothing.
    """
    current = _current_job.get()
    if current is not None:
        job = current.job
        current.emit(Event(event_type, job=job.label, kind=job.kind, data=data))
