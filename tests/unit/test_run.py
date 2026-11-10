"""`run_stages`: one loop runs every stage's jobs, under the caps the stages
set, and each stage finishes once all of them have ended."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from vera_mh.run import run_stages
from vera_mh.workers.events import JobKind
from vera_mh.workers.job import Job, Outcome


@dataclass
class Job_:
    """A job that records how many of its kind (and group) run at once."""

    label: str
    kind: JobKind
    running: Counter
    peak: Counter
    group: str = ""
    next_jobs: list[Job] = field(default_factory=list)

    def done(self) -> bool:
        return False

    async def run(self) -> None:
        keys = [self.kind, f"{self.kind} {self.group}"]
        for key in keys:
            self.running[key] += 1
            self.peak[key] = max(self.peak[key], self.running[key])
        await asyncio.sleep(0.02)
        for key in keys:
            self.running[key] -= 1

    def follow_ups(self) -> list[Job]:
        return list(self.next_jobs)


@dataclass
class FakeStage:
    name: str
    kind: JobKind
    folder: str
    first_jobs: list[Job]
    max_concurrent: int | None = None
    cap_per_group: bool = False
    finished_with: list[list[Outcome]] = field(default_factory=list)

    def jobs(self) -> list[Job]:
        return list(self.first_jobs)

    def finish(self, outcomes: list[Outcome]) -> None:
        self.finished_with.append(outcomes)


def _jobs(kind: JobKind, count: int, running, peak, groups=("",)) -> list[Job]:
    return [
        Job_(f"{kind} {i}", kind, running, peak, group=groups[i % len(groups)])
        for i in range(count)
    ]


async def test_every_stage_runs_on_one_loop_under_its_own_cap(tmp_path: Path):
    running, peak = Counter(), Counter()
    generate = FakeStage(
        "generate",
        JobKind.SESSION,
        str(tmp_path),
        _jobs(JobKind.SESSION, 4, running, peak),
        max_concurrent=1,
    )
    judge = FakeStage(
        "judge",
        JobKind.EVALUATION,
        str(tmp_path),
        _jobs(JobKind.EVALUATION, 4, running, peak),
        max_concurrent=2,
    )

    outcomes = await run_stages([generate, judge])

    assert len(outcomes) == 8
    assert peak["session"] == 1
    assert peak["evaluation"] == 2


async def test_a_stage_can_cap_each_group_on_its_own(tmp_path: Path):
    running, peak = Counter(), Counter()
    judge = FakeStage(
        "judge",
        JobKind.EVALUATION,
        str(tmp_path),
        _jobs(JobKind.EVALUATION, 6, running, peak, groups=("a", "b")),
        max_concurrent=1,
        cap_per_group=True,
    )

    await run_stages([judge])

    assert peak["evaluation a"] == 1
    assert peak["evaluation b"] == 1
    assert peak["evaluation"] == 2


async def test_each_stage_finishes_once_after_every_job_including_follow_ups(
    tmp_path: Path,
):
    """A stage with no first jobs of its own still caps and finishes the jobs
    another stage hands on, as judging will for generation's conversations."""
    running, peak = Counter(), Counter()
    evaluations = _jobs(JobKind.EVALUATION, 3, running, peak)
    session = Job_("session", JobKind.SESSION, running, peak, next_jobs=evaluations)
    generate = FakeStage("generate", JobKind.SESSION, str(tmp_path), [session])
    judge = FakeStage("judge", JobKind.EVALUATION, str(tmp_path), [], max_concurrent=1)

    await run_stages([generate, judge])

    assert peak["evaluation"] == 1
    for stage in (generate, judge):
        (outcomes,) = stage.finished_with
        assert len(outcomes) == 4


async def test_events_go_to_the_terminal_and_each_stage_folder(tmp_path: Path, capsys):
    running, peak = Counter(), Counter()
    judge = FakeStage(
        "judge",
        JobKind.EVALUATION,
        str(tmp_path),
        _jobs(JobKind.EVALUATION, 2, running, peak),
    )

    await run_stages([judge])

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [e["type"] for e in events].count("job_finished") == 2
    assert events[-1]["type"] == "stage_finished"
    assert events[-1]["stage"] == "judge"
    assert "judge finished: 2 done, 0 failed, 0 skipped" in capsys.readouterr().out
