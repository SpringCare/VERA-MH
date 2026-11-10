"""Run the stages of one `vera` run on a single job loop.

The pipeline is the complete case: generate, then judge, then score. `vera
generate` and `vera judge` are the same run with fewer stages. Whatever the
stages, one function owns the run's one `JobLoop` (`workers/loop.py`): it asks
each stage for its first jobs, runs them all under the caps the stages set, and
then lets each stage finish (judging writes `results.csv`, say). The CLI never
calls the loop; stages never create one.

Nothing calls `run_stages` yet: each stage joins as it moves onto the loop
(judging first). Design: docs/design/job-loop.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from vera_mh.utils.debug import is_debug
from vera_mh.workers.events import JobKind, Reporter
from vera_mh.workers.job import Job, Outcome
from vera_mh.workers.loop import JobLoop
from vera_mh.workers.reporters import EventsFileReporter, PlainLineReporter


class Stage(Protocol):
    """One stage of a run, such as judging: its jobs, and what it does once
    they're all done."""

    name: str  # such as "judge"; names the run's final event
    kind: JobKind  # the kind of job the stage runs, whose cap it sets
    max_concurrent: int | None  # that cap; None or 0 is no limit
    cap_per_group: bool  # the cap applies to each job group (`--per-judge`)
    folder: str  # where the stage writes; its `events.jsonl` goes here

    def jobs(self) -> list[Job]:
        """The stage's first jobs. Their follow-ups come from the jobs."""
        ...

    def finish(self, outcomes: list[Outcome]) -> None:
        """Run once every job of the run has ended, such as writing
        `results.csv` from the evaluations on disk."""
        ...


async def run_stages(stages: Sequence[Stage]) -> list[Outcome]:
    """Run every stage's jobs on one loop, then finish each stage, in order.

    Returns every job's outcome, from all stages.
    """
    reporters: list[Reporter] = [PlainLineReporter(verbose=is_debug())]
    reporters += [
        EventsFileReporter(Path(stage.folder) / "events.jsonl") for stage in stages
    ]
    loop = JobLoop(
        {stage.kind: stage.max_concurrent for stage in stages},
        cap_per_group={stage.kind for stage in stages if stage.cap_per_group},
        reporters=reporters,
        stage="+".join(stage.name for stage in stages),
    )
    for stage in stages:
        for job in stage.jobs():
            loop.submit(job)
    outcomes = await loop.run()
    for stage in stages:
        stage.finish(outcomes)
    return outcomes
