"""The job loop: all the concurrency code for generation and judging, in one place.

A job is plain data (paths, model names, indices) plus three methods; see
`workers/job.py`.
A stage creates a `JobLoop`, adds its jobs with `submit`, and awaits `run`.
Each job kind (`session`, `evaluation`) has its own concurrency cap. `run`
also starts each finished job's follow-ups (the next session of a
conversation, or a finished conversation's evaluations) as soon as it finishes.

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
from typing import AbstractSet, Any, Mapping, Sequence

from vera_mh.workers.events import Event, EventType, JobKind, Reporter
from vera_mh.workers.job import Job, Outcome, Status, set_current_job


class ReporterError(Exception):
    """A reporter raised. That's a bug, not the job's failure, so it stops the
    run instead of becoming the outcome of whichever job was reporting."""


class JobLoop:
    """Runs jobs with one concurrency cap per job kind.

    A class rather than a function because the jobs the stage submits and the
    follow-ups that finished jobs hand on share the same caps, the same running
    tasks, and the same outcomes.
    """

    def __init__(
        self,
        max_concurrent: Mapping[JobKind, int | None],
        *,
        cap_per_group: AbstractSet[JobKind] = frozenset(),
        reporters: Sequence[Reporter] = (),
        stage: str = "jobs",
    ):
        """
        Args:
            max_concurrent: How many jobs of each kind may run at once, such as
                `{JobKind.SESSION: 10, JobKind.EVALUATION: 20}`. None or 0 means
                no limit. Every kind a job can have needs an entry.
            cap_per_group: Kinds whose cap applies to each group of jobs on its
                own instead of to all of them, such as `{JobKind.EVALUATION}`
                for `--per-judge`. A job of such a kind names its group with a
                `group` attribute (a judge's model).
            reporters: Where events go, such as a `PlainLineReporter` for the
                terminal and an `EventsFileReporter` for `events.jsonl`. Every
                event goes to each of them.
            stage: The name the final `STAGE_FINISHED` event gives this run,
                such as "generate".
        """
        self.max_concurrent = dict(max_concurrent)
        self.cap_per_group = frozenset(cap_per_group)
        # One limit per kind, or per (kind, group) for `cap_per_group` kinds,
        # made when its first job starts. See `_limit`.
        self.limits: dict[Any, Any] = {}
        self.reporters = reporters
        self.stage = stage
        self.jobs: list[Job] = []
        self.outcomes: list[Outcome] = []

    def submit(self, job: Job) -> None:
        """Add a job for `run()` to run."""
        self.jobs.append(job)

    async def run(self) -> list[Outcome]:
        """Run every submitted job, and the follow-ups they hand on, until none
        is left; return every job's outcome, skipped ones included."""
        # A TaskGroup waits for all its tasks, including those added while it
        # waits: the follow-ups each job adds as it ends (see `_run_one`). Each
        # job waits inside its own task for a free slot of its kind.
        # A job's own errors are its outcome, so an error that escapes a task is
        # a bug in the loop or a reporter: the group then cancels the other
        # jobs and `run()` raises it, rather than lose it.
        async with asyncio.TaskGroup() as tasks:
            self.tasks = tasks
            for job in self.jobs:
                tasks.create_task(self._run_one(job))

        counts = Counter(outcome.status for outcome in self.outcomes)
        totals = {
            "stage": self.stage,
            "done": counts[Status.FINISHED],
            "failed": counts[Status.FAILED],
            "skipped": counts[Status.SKIPPED],
        }
        self.emit(Event(EventType.STAGE_FINISHED, data=totals))
        return self.outcomes

    def emit(self, event: Event) -> None:
        """Send `event` to every reporter. Jobs' own events come here too,
        through `report()`."""
        for reporter in self.reporters:
            try:
                reporter.emit(event)
            except Exception as error:
                name = type(reporter).__name__
                raise ReporterError(
                    f"{name} failed on {event.type}: {error}"
                ) from error

    async def _run_one(self, job: Job) -> None:
        # Any error, from `done()`, `run()` or `follow_ups()`, becomes this
        # job's outcome: one job failing never stops the others.
        elapsed = 0.0
        try:
            if job.done():
                status = Status.SKIPPED
            else:
                async with self._limit(job):
                    self.emit(
                        Event(EventType.JOB_STARTED, job=job.label, kind=job.kind)
                    )
                    # This coroutine is the job's own task (see `run`), so this
                    # applies to this job only, and ends with it.
                    set_current_job(job, self.emit)
                    start = time.monotonic()
                    try:
                        await job.run()
                    finally:
                        # Also when it fails, after minutes of LLM retries, say.
                        elapsed = time.monotonic() - start
                status = Status.FINISHED
            follow_ups = job.follow_ups()
        except ReporterError:
            raise  # not this job's failure: it stops the run (see `run`)
        except Exception as error:
            self._record(Outcome(job, Status.FAILED, error=error, elapsed=elapsed))
            return
        self._record(Outcome(job, status, elapsed=elapsed))
        for follow_up in follow_ups:
            self.tasks.create_task(self._run_one(follow_up))

    def _limit(self, job: Job) -> Any:
        """What a job waits on for a free slot. A semaphore lets in up to `cap`
        jobs at once and makes the rest wait their turn; `nullcontext()` lets
        everything in, for a kind with no cap."""
        key: Any = job.kind
        if job.kind in self.cap_per_group:
            key = (job.kind, getattr(job, "group"))
        if key not in self.limits:
            cap = self.max_concurrent[job.kind]
            self.limits[key] = (
                asyncio.Semaphore(cap) if cap else contextlib.nullcontext()
            )
        return self.limits[key]

    def _record(self, outcome: Outcome) -> None:
        self.outcomes.append(outcome)
        job = outcome.job
        if outcome.status == Status.SKIPPED:
            self.emit(Event(EventType.JOB_SKIPPED, job=job.label, kind=job.kind))
        elif outcome.status == Status.FINISHED:
            data = {"elapsed": round(outcome.elapsed, 3)}
            self.emit(
                Event(EventType.JOB_FINISHED, job=job.label, kind=job.kind, data=data)
            )
        else:
            data = {"error": f"{type(outcome.error).__name__}: {outcome.error}"}
            self.emit(
                Event(EventType.JOB_FAILED, job=job.label, kind=job.kind, data=data)
            )
