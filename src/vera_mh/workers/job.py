"""What a job is, and what can happen to one.

`JobLoop` (workers/loop.py) runs jobs; generation and judging define them: one
session of a conversation, or one evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

# What happened to a job: it ran and wrote its output, it raised, or its output
# was already on disk.
Status = Literal["finished", "failed", "skipped"]


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
