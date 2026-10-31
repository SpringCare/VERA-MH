"""The job loop: skipping, concurrency caps, retries, follow-ups, and events."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

from vera_mh.workers import events
from vera_mh.workers.events import Event
from vera_mh.workers.loop import Job, run_jobs

UNLIMITED: dict[str, int | None] = {"session": None, "evaluation": None}


class Tracker:
    """What the fake jobs did: the order things happened in, and how many jobs
    of each kind ran at once."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.running: Counter[str] = Counter()
        self.peak: Counter[str] = Counter()


@dataclass
class FakeJob:
    label: str
    tracker: Tracker
    kind: str = "session"
    is_done: bool = False
    # run() raises this many times before it succeeds.
    failures: int = 0
    error: type[Exception] = RuntimeError
    retry_on: tuple[type[Exception], ...] = (RuntimeError,)
    next_jobs: list[FakeJob] = field(default_factory=list)
    # How long run() takes, so jobs overlap.
    seconds: float = 0.0
    runs: int = 0

    def done(self) -> bool:
        return self.is_done

    async def run(self) -> None:
        self.runs += 1
        self.tracker.log.append(f"start {self.label}")
        self.tracker.running[self.kind] += 1
        self.tracker.peak[self.kind] = max(
            self.tracker.peak[self.kind], self.tracker.running[self.kind]
        )
        try:
            await asyncio.sleep(self.seconds)
            if self.runs <= self.failures:
                raise self.error(f"{self.label} failed")
        finally:
            self.tracker.running[self.kind] -= 1
            self.tracker.log.append(f"end {self.label}")

    def follow_ups(self) -> list[Job]:
        return list(self.next_jobs)

    def retryable(self, error: Exception) -> bool:
        return isinstance(error, self.retry_on)


class ListReporter:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def types(self, label: str) -> list[str]:
        return [event.type for event in self.events if event.job == label]


def statuses(outcomes) -> dict[str, str]:
    return {outcome.job.label: outcome.status for outcome in outcomes}


async def test_a_done_job_is_skipped_and_not_run() -> None:
    tracker = Tracker()
    done = FakeJob("done", tracker, is_done=True)
    todo = FakeJob("todo", tracker)

    outcomes = await run_jobs([done, todo], UNLIMITED)

    assert statuses(outcomes) == {"done": "skipped", "todo": "finished"}
    assert done.runs == 0
    assert todo.runs == 1


async def test_each_kind_has_its_own_cap() -> None:
    tracker = Tracker()
    sessions = [FakeJob(f"s{i}", tracker, seconds=0.02) for i in range(6)]
    evaluations = [
        FakeJob(f"e{i}", tracker, kind="evaluation", seconds=0.02) for i in range(6)
    ]

    outcomes = await run_jobs(
        [*sessions, *evaluations], {"session": 2, "evaluation": 3}
    )

    assert all(outcome.status == "finished" for outcome in outcomes)
    assert tracker.peak == Counter({"session": 2, "evaluation": 3})


async def test_a_failure_is_an_outcome_not_an_exception() -> None:
    tracker = Tracker()
    bad = FakeJob("bad", tracker, failures=1, error=ValueError, retry_on=())
    good = FakeJob("good", tracker)

    outcomes = await run_jobs([bad, good], UNLIMITED)

    assert statuses(outcomes) == {"bad": "failed", "good": "finished"}
    failed = next(outcome for outcome in outcomes if outcome.status == "failed")
    assert isinstance(failed.error, ValueError)


async def test_retries_stop_at_attempts() -> None:
    tracker = Tracker()
    job = FakeJob("flaky", tracker, failures=5)

    (outcome,) = await run_jobs([job], UNLIMITED, attempts=3)

    assert outcome.status == "failed"
    assert job.runs == 3
    assert outcome.attempts == 3


async def test_a_retry_can_succeed() -> None:
    tracker = Tracker()
    job = FakeJob("flaky", tracker, failures=1)

    (outcome,) = await run_jobs([job], UNLIMITED, attempts=3)

    assert outcome.status == "finished"
    assert outcome.attempts == 2


async def test_an_error_that_isnt_retryable_is_not_retried() -> None:
    tracker = Tracker()
    job = FakeJob("broken", tracker, failures=5, error=ValueError)

    (outcome,) = await run_jobs([job], UNLIMITED, attempts=3)

    assert outcome.status == "failed"
    assert job.runs == 1


async def test_follow_ups_run_after_their_parent() -> None:
    tracker = Tracker()
    child = FakeJob("child", tracker, kind="evaluation")
    parent = FakeJob("parent", tracker, next_jobs=[child])

    outcomes = await run_jobs([parent], UNLIMITED)

    assert statuses(outcomes) == {"parent": "finished", "child": "finished"}
    assert tracker.log == ["start parent", "end parent", "start child", "end child"]


async def test_follow_ups_dont_run_when_their_parent_fails() -> None:
    tracker = Tracker()
    child = FakeJob("child", tracker)
    parent = FakeJob("parent", tracker, failures=1, next_jobs=[child])

    outcomes = await run_jobs([parent], UNLIMITED)

    assert statuses(outcomes) == {"parent": "failed"}
    assert child.runs == 0


async def test_a_skipped_job_hands_on_its_follow_ups_each_checked_for_done() -> None:
    """Resume: sessions 1 and 2 are on disk, session 3 isn't, and after the
    last session come the evaluations. Only the missing work runs."""
    tracker = Tracker()
    evaluation = FakeJob("evaluation", tracker, kind="evaluation")
    session_3 = FakeJob("session 3", tracker, next_jobs=[evaluation])
    session_2 = FakeJob("session 2", tracker, is_done=True, next_jobs=[session_3])
    session_1 = FakeJob("session 1", tracker, is_done=True, next_jobs=[session_2])

    outcomes = await run_jobs([session_1], UNLIMITED)

    assert statuses(outcomes) == {
        "session 1": "skipped",
        "session 2": "skipped",
        "session 3": "finished",
        "evaluation": "finished",
    }
    assert session_1.runs == session_2.runs == 0


@pytest.mark.parametrize("unlimited", [None, 0])
async def test_follow_ups_of_an_unlimited_kind_all_run_at_once(unlimited) -> None:
    """One job hands on five follow-ups. With no cap they must all run at
    once, even though the list `run_jobs` was given held a single job."""
    tracker = Tracker()
    evaluations = [
        FakeJob(f"e{i}", tracker, kind="evaluation", seconds=0.05) for i in range(5)
    ]
    session = FakeJob("session", tracker, next_jobs=evaluations)

    await run_jobs([session], {"session": unlimited, "evaluation": unlimited})

    assert tracker.peak["evaluation"] == 5


@pytest.mark.parametrize("unlimited", [None, 0])
async def test_none_and_zero_both_mean_no_limit(unlimited) -> None:
    tracker = Tracker()
    jobs = [FakeJob(f"s{i}", tracker, seconds=0.05) for i in range(4)]

    await run_jobs(jobs, {"session": unlimited})

    assert tracker.peak["session"] == 4


async def test_done_or_follow_ups_raising_counts_the_job_as_failed() -> None:
    class BrokenDone(FakeJob):
        def done(self) -> bool:
            raise OSError("disk gone")

    class BrokenFollowUps(FakeJob):
        def follow_ups(self) -> list[Job]:
            raise OSError("disk gone")

    tracker = Tracker()
    outcomes = await run_jobs(
        [BrokenDone("done", tracker), BrokenFollowUps("follow-ups", tracker)],
        UNLIMITED,
    )

    assert statuses(outcomes) == {"done": "failed", "follow-ups": "failed"}


async def test_attempts_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        await run_jobs([], UNLIMITED, attempts=0)


# ------------------------------------------------------------------------ events


async def test_the_loop_reports_each_job_and_the_stage() -> None:
    tracker = Tracker()
    reporter = ListReporter()
    jobs = [
        FakeJob("ok", tracker),
        FakeJob("flaky", tracker, failures=1),
        FakeJob("bad", tracker, failures=1, error=ValueError),
        FakeJob("done", tracker, is_done=True),
    ]

    await run_jobs(jobs, UNLIMITED, attempts=2, reporter=reporter, stage="generate")

    assert reporter.types("ok") == [events.JOB_STARTED, events.JOB_FINISHED]
    assert reporter.types("flaky") == [
        events.JOB_STARTED,
        events.JOB_RETRYING,
        events.JOB_FINISHED,
    ]
    assert reporter.types("bad") == [events.JOB_STARTED, events.JOB_FAILED]
    assert reporter.types("done") == [events.JOB_SKIPPED]
    stage = reporter.events[-1]
    assert stage.type == events.STAGE_FINISHED
    assert stage.data == {"stage": "generate", "done": 2, "failed": 1, "skipped": 1}


async def test_a_job_reports_from_inside_its_run() -> None:
    @dataclass
    class Talker:
        label: str
        kind: str = "session"

        def done(self) -> bool:
            return False

        async def run(self) -> None:
            events.report(events.TURN_STARTED, turn=1, speaker="user")

        def follow_ups(self) -> list[Job]:
            return []

        def retryable(self, error: Exception) -> bool:
            return False

    reporter = ListReporter()

    await run_jobs([Talker("a"), Talker("b")], UNLIMITED, reporter=reporter)

    turns = [event for event in reporter.events if event.type == events.TURN_STARTED]
    assert {event.job for event in turns} == {"a", "b"}
    assert all(event.kind == "session" for event in turns)
    assert turns[0].data == {"turn": 1, "speaker": "user"}


def test_report_outside_a_job_does_nothing() -> None:
    events.report(events.TURN_STARTED, turn=1, speaker="user")


# --------------------------------------------------------- jobs stay serializable


@dataclass
class SampleSession:
    """What a real job looks like: JSON-safe fields, plus the four methods."""

    label: str
    persona: str
    repeat: int
    session: int
    run_dir: str
    kind: str = "session"

    def output(self) -> Path:
        name = f"{self.persona}_run{self.repeat}_session{self.session}.txt"
        return Path(self.run_dir) / name

    def done(self) -> bool:
        return self.output().exists()

    async def run(self) -> None:
        self.output().write_text("user: hi\n", encoding="utf-8")

    def follow_ups(self) -> list[Job]:
        return []

    def retryable(self, error: Exception) -> bool:
        return False


async def test_a_job_survives_a_json_round_trip_and_runs(tmp_path: Path) -> None:
    job = SampleSession("gpt-5.2 Brian", "Brian", 1, 2, str(tmp_path))

    copy = SampleSession(**json.loads(json.dumps(asdict(job))))

    assert copy == job
    (outcome,) = await run_jobs([copy], UNLIMITED)
    assert outcome.status == "finished"
    assert job.done()
