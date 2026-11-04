"""The job loop: all the concurrency code for generation and judging, in one place.

A job is plain data (paths, model names, indices) plus three methods; see `Job`.
Each job kind (`session`, `evaluation`) has its own queue and its own cap.
Everything that adds work goes through `JobLoop.submit`, which never waits: the
stage loading its first jobs, and a finished job handing on its follow-ups (the
next session of a conversation, or a finished conversation's evaluations).
`JobLoop.run` starts queued jobs while their kind has room, until nothing is
queued or running.

A job whose output is already on disk is skipped, but still hands on its
follow-ups, so resuming a run reaches every missing job however far down the
chain it is, and redoes nothing. The loop never retries a job: LLM calls retry
themselves, inside it. A failed job is recorded and rerun later with `--into`.

Design: docs/design/job-loop.md.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol

from vera_mh.workers.events import (
    JOB_FAILED,
    JOB_FINISHED,
    JOB_SKIPPED,
    JOB_STARTED,
    STAGE_FINISHED,
    Event,
    Reporter,
    running_job,
)
from vera_mh.workers.reporters import Reporters


class Job(Protocol):
    """One unit of work: one session of a conversation, or one evaluation.

    Keep jobs plain data, with JSON-safe fields, so they could be sent to an
    external queue one day (job-loop.md, decision 7).
    """

    kind: str  # "session" or "evaluation": picks the queue and the cap
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
    """Runs jobs with one queue and one concurrency cap per job kind.

    A class rather than a function because `submit` is called both by the stage
    and by finished jobs handing on their follow-ups, and both need the same
    queues.
    """

    def __init__(
        self,
        max_concurrent: Mapping[str, int | None],
        *,
        reporter: Reporter | None = None,
        stage: str = "jobs",
    ):
        """`max_concurrent` caps how many jobs of each kind run at once; None
        or 0 means no limit, and every job's kind must have an entry. `stage`
        names the run in the final `STAGE_FINISHED` event."""
        self.caps = dict(max_concurrent)
        self.queues: dict[str, deque[Job]] = {kind: deque() for kind in self.caps}
        self.running: dict[asyncio.Task[None], tuple[Job, float]] = {}
        self.outcomes: list[Outcome] = []
        self.report = reporter or Reporters()
        self.stage = stage

    def submit(self, job: Job) -> None:
        """Put a job on its kind's queue, or skip it if its output is on disk.
        Never waits."""
        try:
            is_done = job.done()
        except Exception as error:
            # A corrupt file, say: fail this job rather than the whole run.
            self._record(Outcome(job, "failed", error=error))
            return
        if is_done:
            self._hand_on(Outcome(job, "skipped"))
        else:
            self.queues[job.kind].append(job)

    async def run(self) -> list[Outcome]:
        """Run until every queue is empty and nothing is running; return every
        job's outcome, skipped ones included."""
        while True:
            self._start_jobs_that_fit()
            if not self.running:
                break
            finished, _ = await asyncio.wait(
                self.running, return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished:
                job, started = self.running.pop(task)
                seconds = time.monotonic() - started
                error = task.exception()
                if error is None:
                    self._hand_on(Outcome(job, "finished", seconds=seconds))
                else:
                    self._record(Outcome(job, "failed", error=error, seconds=seconds))

        counts = Counter(outcome.status for outcome in self.outcomes)
        totals = {
            "stage": self.stage,
            "done": counts["finished"],
            "failed": counts["failed"],
            "skipped": counts["skipped"],
        }
        self.report.emit(Event(STAGE_FINISHED, data=totals))
        return self.outcomes

    def _start_jobs_that_fit(self) -> None:
        for kind, queue in self.queues.items():
            while queue and self._has_room(kind):
                job = queue.popleft()
                self.report.emit(Event(JOB_STARTED, job=job.label, kind=job.kind))
                task = asyncio.create_task(self._run_one(job))
                self.running[task] = (job, time.monotonic())

    async def _run_one(self, job: Job) -> None:
        # Each job runs in its own task, so `report()` inside it reaches the
        # reporter with this job's name, and concurrent jobs don't mix.
        with running_job(self.report, job.label, job.kind):
            await job.run()

    def _has_room(self, kind: str) -> bool:
        cap = self.caps[kind]
        running = sum(1 for job, _ in self.running.values() if job.kind == kind)
        return not cap or running < cap

    def _hand_on(self, outcome: Outcome) -> None:
        """Record a finished or skipped job, and submit its follow-ups."""
        try:
            follow_ups = outcome.job.follow_ups()
        except Exception as error:
            # The jobs waiting on this one can't be listed, so they would go
            # missing without a trace: count the job as failed instead.
            outcome = Outcome(
                outcome.job, "failed", error=error, seconds=outcome.seconds
            )
            follow_ups = []
        self._record(outcome)
        for follow_up in follow_ups:
            self.submit(follow_up)

    def _record(self, outcome: Outcome) -> None:
        self.outcomes.append(outcome)
        job = outcome.job
        if outcome.status == "skipped":
            self.report.emit(Event(JOB_SKIPPED, job=job.label, kind=job.kind))
        elif outcome.status == "finished":
            data = {"seconds": round(outcome.seconds, 3)}
            self.report.emit(
                Event(JOB_FINISHED, job=job.label, kind=job.kind, data=data)
            )
        else:
            data = {"error": f"{type(outcome.error).__name__}: {outcome.error}"}
            self.report.emit(Event(JOB_FAILED, job=job.label, kind=job.kind, data=data))
