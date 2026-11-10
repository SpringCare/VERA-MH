"""`LLMJudge` as a job: when it's done, what it writes, and that it stays data."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from vera_mh.judge.llm_judge import LLMJudge
from vera_mh.judge.utils import judge_evaluation_tsv_filename
from vera_mh.workers.events import Event, EventType, JobKind
from vera_mh.workers.loop import JobLoop


def _job(fixtures_dir: Path, tmp_path: Path, instance: int = 1) -> LLMJudge:
    conversation = tmp_path / "Abigail_run1.txt"
    conversation.write_text("user: hello\nchatbot: hi", encoding="utf-8")
    output = tmp_path / "evaluations"
    output.mkdir(exist_ok=True)
    return LLMJudge(
        conversation_path=str(conversation),
        judge_model="gpt-5.4",
        judge_instance=instance,
        output_folder=str(output),
        rubric_file=str(fixtures_dir / "rubric_simple.tsv"),
        rubric_prompt_beginning_file=str(fixtures_dir / "rubric_prompt_beginning.txt"),
        question_prompt_file=str(fixtures_dir / "question_prompt.txt"),
        extra_params={"temperature": 0},
    )


def _tsv(job: LLMJudge) -> Path:
    name = judge_evaluation_tsv_filename(
        "Abigail_run1.txt", job.judge_model, job.judge_instance
    )
    return Path(job.output_folder) / name


async def _first_option(self, question_id, question_data):
    """Stands in for the judge LLM: every question gets its first option."""
    return question_data["answers"][0]["option"], f"scripted answer to {question_id}"


@pytest.fixture
def scripted_judge():
    """`LLMJudge` with no LLM behind it: no client is created, and each
    question is answered by `_first_option`."""
    with (
        patch.object(LLMJudge, "_create_evaluator", return_value=AsyncMock()),
        patch.object(LLMJudge, "_ask_single_question", _first_option),
    ):
        yield


class ListReporter:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


@pytest.mark.unit
def test_an_evaluation_is_done_when_its_tsv_exists(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    job = _job(fixtures_dir, tmp_path)
    assert not job.done()

    _tsv(job).write_text("Dimension\tScore\tReasoning\n", encoding="utf-8")

    assert job.done()


@pytest.mark.unit
def test_each_instance_has_its_own_tsv(fixtures_dir: Path, tmp_path: Path) -> None:
    first = _job(fixtures_dir, tmp_path, instance=1)
    second = _job(fixtures_dir, tmp_path, instance=2)
    _tsv(first).write_text("Dimension\tScore\tReasoning\n", encoding="utf-8")

    assert first.done()
    assert not second.done()


@pytest.mark.unit
def test_label_names_the_judge_and_the_conversation(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    assert _job(fixtures_dir, tmp_path).label == "gpt-5.4 Abigail_run1"
    assert _job(fixtures_dir, tmp_path, 2).label == "gpt-5.4 Abigail_run1 i2"


@pytest.mark.unit
def test_an_evaluation_is_an_evaluation_job_with_no_follow_ups(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    job = _job(fixtures_dir, tmp_path)

    assert job.kind == JobKind.EVALUATION
    assert job.follow_ups() == []


@pytest.mark.unit
async def test_run_writes_the_tsv_and_the_answers(
    fixtures_dir: Path, tmp_path: Path, scripted_judge
) -> None:
    job = _job(fixtures_dir, tmp_path)

    await job.run()

    assert job.done()
    answers = Path(job.output_folder) / "answers" / "by_question" / _tsv(job).name
    assert answers.is_file()


@pytest.mark.unit
async def test_run_logs_beside_the_scores_tsv(
    fixtures_dir: Path, tmp_path: Path, scripted_judge
) -> None:
    job = _job(fixtures_dir, tmp_path)

    await job.run()

    assert job.log_file == str(
        Path(job.output_folder) / "logs" / f"{_tsv(job).stem}.log"
    )


@pytest.mark.unit
def test_default_temperature_does_not_change_the_callers_dict(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    params = {"max_tokens": 10}
    job = LLMJudge(**{**asdict(_job(fixtures_dir, tmp_path)), "extra_params": params})

    assert job.extra_params == {"temperature": 0, "max_tokens": 10}
    assert params == {"max_tokens": 10}


@pytest.mark.unit
async def test_an_evaluation_survives_a_json_round_trip_and_runs(
    fixtures_dir: Path, tmp_path: Path, scripted_judge
) -> None:
    job = _job(fixtures_dir, tmp_path)

    copy = LLMJudge(**json.loads(json.dumps(asdict(job))))

    assert copy == job
    loop = JobLoop({JobKind.EVALUATION: None})
    loop.submit(copy)
    (outcome,) = await loop.run()
    assert outcome.status == "finished"
    assert job.done()


@pytest.mark.unit
async def test_each_question_asked_is_reported(
    fixtures_dir: Path, tmp_path: Path, scripted_judge
) -> None:
    job = _job(fixtures_dir, tmp_path)
    reporter = ListReporter()

    loop = JobLoop({JobKind.EVALUATION: None}, reporters=[reporter])
    loop.submit(job)
    await loop.run()

    asked = [e for e in reporter.events if e.type == EventType.QUESTION_ASKED]
    assert asked
    assert {event.job for event in asked} == {job.label}
    assert all("question_id" in event.data for event in asked)
