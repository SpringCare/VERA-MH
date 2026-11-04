"""The job loop: all the concurrency code for generation and judging, in one place.

A job is plain data (paths, model names, indices) plus three methods; see `Job`.
A stage creates a `JobLoop`, hands it its first jobs with `submit`, and awaits
`run`. Each job kind (`session`, `evaluation`) has its own concurrency cap. A
finished job's follow-ups (the next session of a conversation, or a finished
conversation's evaluations) are submitted the same way, as soon as it finishes.

A job whose output is already on disk is skipped, but still hands on its
follow-ups, so resuming a run reaches every missing job however far down the
chain it is, and redoes nothing. The loop never retries a job: LLM calls retry
themselves, inside it. A failed job is recorded and rerun later with `--into`.

Design: docs/design/job-loop.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence

from vera_mh.workers.events import (
    JOB_FAILED,
    JOB_FINISHED,
    JOB_SKIPPED,
    JOB_STARTED,
    STAGE_FINISHED,
    Event,
    Reporter,
    reporting_for,
)


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


Status = Literal["finished", "failed", "skipped"]


@dataclass
class Outcome:
    """What happened to one job."""

    job: Job
    status: Status
    error: BaseException | None = None
    seconds: float = 0.0


class JobLoop:
    """Runs jobs with one concurrency cap per job kind.

    A class rather than a function because `submit` is called both by the stage
    and by finished jobs handing on their follow-ups, and both need the same
    caps and the same set of running jobs.
    """

    def __init__(
        self,
        max_concurrent: Mapping[str, int | None],
        *,
        reporters: Sequence[Reporter] = (),
        stage: str = "jobs",
    ):
        """
        Args:
            max_concurrent: How many jobs of each kind may run at once, such as
                `{"session": 10, "evaluation": 20}`. None or 0 means no limit.
                Every kind a job can have needs an entry.
            reporters: Where events go, such as a `PlainLineReporter` for the
                terminal and an `EventsFileReporter` for `events.jsonl`. Every
                event goes to each of them.
            stage: The name the final `STAGE_FINISHED` event gives this run,
                such as "generate".
        """
        # A semaphore per capped kind: a job waits for a free slot of its kind.
        self.limits = {
            kind: asyncio.Semaphore(cap) if cap else contextlib.nullcontext()
            for kind, cap in max_concurrent.items()
        }
        self.reporters = reporters
        self.stage = stage
        self.tasks: set[asyncio.Task[None]] = set()
        self.outcomes: list[Outcome] = []

    def submit(self, job: Job) -> None:
        """Start a job in the background; it waits for a free slot of its kind.
        Never waits itself. Call it from inside a running event loop."""
        self.tasks.add(asyncio.create_task(self._run_one(job)))

    async def run(self) -> list[Outcome]:
        """Wait until every job has finished, follow-ups included; return every
        job's outcome, skipped ones included."""
        while self.tasks:
            # Waits for all the jobs started so far. Their follow-ups add new
            # tasks while it waits, so go round until no task is left.
            finished, _ = await asyncio.wait(self.tasks)
            self.tasks -= finished

        counts = Counter(outcome.status for outcome in self.outcomes)
        totals = {
            "stage": self.stage,
            "done": counts["finished"],
            "failed": counts["failed"],
            "skipped": counts["skipped"],
        }
        self.emit(Event(STAGE_FINISHED, data=totals))
        return self.outcomes

    def emit(self, event: Event) -> None:
        for reporter in self.reporters:
            reporter.emit(event)

    async def _run_one(self, job: Job) -> None:
        # Any error, from `done()`, `run()` or `follow_ups()`, becomes this
        # job's outcome: one job failing never stops the others.
        seconds = 0.0
        try:
            if job.done():
                status: Status = "skipped"
            else:
                async with self.limits[job.kind]:
                    self.emit(Event(JOB_STARTED, job=job.label, kind=job.kind))
                    start = time.monotonic()
                    with reporting_for(self.emit, job.label, job.kind):
                        await job.run()
                    seconds = time.monotonic() - start
                status = "finished"
            follow_ups = job.follow_ups()
        except Exception as error:
            self._record(Outcome(job, "failed", error=error, seconds=seconds))
            return
        self._record(Outcome(job, status, seconds=seconds))
        for follow_up in follow_ups:
            self.submit(follow_up)

    def _record(self, outcome: Outcome) -> None:
        self.outcomes.append(outcome)
        job = outcome.job
        if outcome.status == "skipped":
            self.emit(Event(JOB_SKIPPED, job=job.label, kind=job.kind))
        elif outcome.status == "finished":
            data = {"seconds": round(outcome.seconds, 3)}
            self.emit(Event(JOB_FINISHED, job=job.label, kind=job.kind, data=data))
        else:
            data = {"error": f"{type(outcome.error).__name__}: {outcome.error}"}
            self.emit(Event(JOB_FAILED, job=job.label, kind=job.kind, data=data))
