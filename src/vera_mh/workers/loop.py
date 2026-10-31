"""The job loop: all the concurrency code for generation and judging, in one place.

A job is plain data (paths, model names, indices) plus four methods; see `Job`.
`run_jobs` skips every job whose output is already on disk, runs the rest with
one concurrency cap per job kind, retries what the job says is worth retrying,
and records each job's outcome. A job's failure is an outcome, never an
exception, so one bad conversation doesn't stop the others.

Jobs that wait on another job are its follow-ups: the next session of a
conversation, or a finished conversation's evaluations. A job hands them on
when it finishes, and also when it is skipped as done, so resuming a run
reaches every missing job however far down the chain it is, and redoes nothing.

Design: docs/design/job-loop.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import Counter
from dataclasses import dataclass
from typing import Awaitable, Iterable, Literal, Mapping, Protocol

from vera_mh.workers.events import (
    JOB_FAILED,
    JOB_FINISHED,
    JOB_RETRYING,
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

    def retryable(self, error: Exception) -> bool:
        """Is running the job again worth it after this error?"""
        ...


Status = Literal["finished", "failed", "skipped"]


@dataclass
class Outcome:
    """What happened to one job."""

    job: Job
    status: Status
    error: Exception | None = None
    attempts: int = 0
    seconds: float = 0.0


async def run_jobs(
    jobs: Iterable[Job],
    max_concurrent: Mapping[str, int | None],
    attempts: int = 1,
    *,
    reporter: Reporter | None = None,
    stage: str = "jobs",
) -> list[Outcome]:
    """Run `jobs` and their follow-ups; return every job's outcome.

    `max_concurrent` caps how many jobs of each kind run at once; None or 0
    means no limit. Every job's kind must have an entry. `attempts` is how many
    times a job may run, counting the first. `stage` names the run in the
    final `STAGE_FINISHED` event.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, not {attempts}")
    report = reporter or Reporters()
    # No semaphore for an unlimited kind: one sized from `jobs` would throttle
    # the follow-ups, which can outnumber them.
    limits = {
        kind: asyncio.Semaphore(cap) if cap else contextlib.nullcontext()
        for kind, cap in max_concurrent.items()
    }

    async def run_one(job: Job) -> list[Outcome]:
        try:
            if job.done():
                outcome = Outcome(job, "skipped")
            else:
                async with limits[job.kind]:
                    outcome = await _run_with_retries(job, attempts, report)
            follow_ups = job.follow_ups() if outcome.status != "failed" else []
        except Exception as error:
            # `done()` or `follow_ups()` crashed, on a corrupt file say. Count the
            # job as failed rather than stop the run, or lose the jobs waiting on it
            # without a trace.
            outcome, follow_ups = Outcome(job, "failed", error=error), []
        _emit_outcome(report, outcome)
        return [
            outcome,
            *await _gather_flat(run_one(follow_up) for follow_up in follow_ups),
        ]

    outcomes = await _gather_flat(run_one(job) for job in jobs)
    counts = Counter(outcome.status for outcome in outcomes)
    report.emit(
        Event(
            STAGE_FINISHED,
            data={
                "stage": stage,
                "done": counts["finished"],
                "failed": counts["failed"],
                "skipped": counts["skipped"],
            },
        )
    )
    return outcomes


async def _run_with_retries(job: Job, attempts: int, report: Reporter) -> Outcome:
    report.emit(Event(JOB_STARTED, job=job.label, kind=job.kind))
    start = time.monotonic()
    attempt = 1
    with running_job(report, job.label, job.kind):
        while True:
            try:
                await job.run()
                return Outcome(job, "finished", attempts=attempt, seconds=_since(start))
            except Exception as error:
                if attempt == attempts or not job.retryable(error):
                    return Outcome(
                        job,
                        "failed",
                        error=error,
                        attempts=attempt,
                        seconds=_since(start),
                    )
                retry = {"attempt": attempt, "error": _describe(error)}
                report.emit(
                    Event(JOB_RETRYING, job=job.label, kind=job.kind, data=retry)
                )
                attempt += 1


def _emit_outcome(report: Reporter, outcome: Outcome) -> None:
    job = outcome.job
    if outcome.status == "skipped":
        report.emit(Event(JOB_SKIPPED, job=job.label, kind=job.kind))
    elif outcome.status == "finished":
        data = {"seconds": round(outcome.seconds, 3), "attempts": outcome.attempts}
        report.emit(Event(JOB_FINISHED, job=job.label, kind=job.kind, data=data))
    else:
        data = {"error": _describe(outcome.error), "attempts": outcome.attempts}
        report.emit(Event(JOB_FAILED, job=job.label, kind=job.kind, data=data))


async def _gather_flat(runs: Iterable[Awaitable[list[Outcome]]]) -> list[Outcome]:
    results = await asyncio.gather(*runs)
    return [outcome for result in results for outcome in result]


def _since(start: float) -> float:
    return time.monotonic() - start


def _describe(error: BaseException | None) -> str:
    return f"{type(error).__name__}: {error}" if error else ""
