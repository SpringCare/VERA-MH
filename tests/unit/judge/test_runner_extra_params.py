"""Unit tests for judge runner extra parameters functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from tests.mocks.run_record import minimal_run_record
from vera_mh.judge.rubric_config import ConversationData, RubricConfig
from vera_mh.judge.run import run_judging
from vera_mh.judge.runner import _create_evaluation_jobs, _run_workers_with_queue
from vera_mh.judge.utils import build_judge_task_log_path, judge_evaluation_tsv_filename
from vera_mh.utils.config_schema import JudgingConfig, ModelSpec, RubricFiles

MOCK_EVALUATION_RESULT = {
    "Safety": {
        "score": "Best Practice",
        "reasoning": "Test",
        "yes_question_id": "",
        "yes_reasoning": "",
    }
}


def _conversation(tmp_path: Path, index: int = 0) -> ConversationData:
    """Single ConversationData for tests."""
    if index == 0:
        content = "User: Hello\nAssistant: Hi!"
    else:
        content = f"User: Hello {index}\nAssistant: Hi {index}!"
    fname = "test_conv.txt" if index == 0 else f"conv_{index}.txt"
    return ConversationData(
        content=content,
        metadata={
            "filename": fname,
            "run_id": "test" if index == 0 else f"test_{index}",
            "source_path": str(tmp_path / fname),
        },
    )


def _conversations(tmp_path: Path, count: int) -> list[ConversationData]:
    """List of ConversationData for tests (count >= 1)."""
    return [_conversation(tmp_path, i) for i in range(count)]


def _conversation_folder(tmp_path: Path) -> Path:
    """A folder holding one conversation file, for `run_judging` to load."""
    folder = tmp_path / "conversations"
    folder.mkdir()
    (folder / "test_conv.txt").write_text("user: Hello\nchatbot: Hi!", encoding="utf-8")
    return folder


def _judging_config(
    fixtures_dir: Path,
    conversations: Path,
    output: Path,
    judge_model: str = "claude-sonnet-4-5",
    extra_params: dict | None = None,
) -> JudgingConfig:
    """A judging config using the `rubric_simple.tsv` fixture rubric."""
    return JudgingConfig(
        models=[
            ModelSpec(name=judge_model, repeats=1, extra_params=extra_params or {})
        ],
        rubrics=[
            RubricFiles(
                rubric_file=str((fixtures_dir / "rubric_simple.tsv").resolve()),
                rubric_prompt_beginning_file=str(
                    (fixtures_dir / "rubric_prompt_beginning.txt").resolve()
                ),
                question_prompt_file=str(
                    (fixtures_dir / "question_prompt.txt").resolve()
                ),
            )
        ],
        max_concurrent=None,
        per_judge=False,
        conversations=[str(conversations)],
        output=str(output),
    )


async def _evaluate_all(
    conversations: list[ConversationData],
    output_folder: str,
    rubric_config: RubricConfig,
    judge_model_extra_params: dict | None = None,
) -> list[dict]:
    """Every evaluation of `conversations` by one judge, through the worker pool."""
    jobs = _create_evaluation_jobs(
        conversations,
        {"claude-sonnet-4-5": 1},
        output_folder,
        rubric_config,
        judge_model_extra_params,
    )
    return await _run_workers_with_queue(jobs, max_concurrent=None, per_judge=False)


@pytest.fixture
def mock_llm_judge_class():
    """Patch LLMJudge; return mock with evaluate_conversation_question_flow stubbed."""
    with patch("vera_mh.judge.runner.LLMJudge") as mock_class:
        mock_inst = MagicMock()
        mock_inst.evaluate_conversation_question_flow = AsyncMock(
            return_value=MOCK_EVALUATION_RESULT
        )
        mock_class.return_value = mock_inst
        yield mock_class


@pytest.mark.unit
class TestRunnerExtraParams:
    """Test that extra parameters are properly handled in runner functions."""

    @pytest.mark.asyncio
    async def test_pool_passes_extra_params_to_the_judge(
        self, tmp_path: Path, rubric_config_factory, mock_llm_judge_class
    ):
        """Each evaluation job hands its extra params to LLMJudge."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        extra_params = {"temperature": 0.7, "max_tokens": 1000}

        results = await _evaluate_all(
            [_conversation(tmp_path)], str(tmp_path), rubric_config, extra_params
        )

        mock_llm_judge_class.assert_called_once()
        call_kw = mock_llm_judge_class.call_args[1]
        assert call_kw["judge_model_extra_params"] == extra_params
        assert "log_file" in call_kw
        assert call_kw["log_file"].endswith(".log")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_pool_extra_params_default_to_none(
        self, tmp_path: Path, rubric_config_factory, mock_llm_judge_class
    ):
        """Test that extra params default to None when not provided."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

        results = await _evaluate_all(
            [_conversation(tmp_path)], str(tmp_path), rubric_config
        )

        mock_llm_judge_class.assert_called_once()
        call_kw = mock_llm_judge_class.call_args[1]
        assert call_kw["judge_model_extra_params"] is None
        assert "log_file" in call_kw
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_run_judging_passes_the_models_extra_params(
        self, tmp_path: Path, fixtures_dir: Path
    ):
        """`run_judging` gives every evaluation job the judge model's params."""
        extra_params = {"temperature": 0.5, "max_tokens": 500}
        config = _judging_config(
            fixtures_dir,
            _conversation_folder(tmp_path),
            tmp_path / "output",
            extra_params=extra_params,
        )

        with patch("vera_mh.judge.run._run_workers_with_queue") as mock_pool:
            mock_pool.return_value = []
            await run_judging(config, run_record=minimal_run_record())

        mock_pool.assert_called_once()
        (jobs, _, _) = mock_pool.call_args.args
        # Extra params are each job's 7th element (index 6).
        assert [job[6] for job in jobs] == [extra_params]

    @pytest.mark.asyncio
    async def test_run_judging_without_extra_params_passes_an_empty_dict(
        self, tmp_path: Path, fixtures_dir: Path
    ):
        """A judge model with no params gives every job an empty dict."""
        config = _judging_config(
            fixtures_dir, _conversation_folder(tmp_path), tmp_path / "output"
        )

        with patch("vera_mh.judge.run._run_workers_with_queue") as mock_pool:
            mock_pool.return_value = []
            await run_judging(config, run_record=minimal_run_record())

        (jobs, _, _) = mock_pool.call_args.args
        assert [job[6] for job in jobs] == [{}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("conversation_count", [3, 5])
    async def test_pool_extra_params_with_multiple_conversations(
        self,
        tmp_path: Path,
        rubric_config_factory,
        mock_llm_judge_class,
        conversation_count: int,
    ):
        """Extra params are passed to LLMJudge for every conversation in batch."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        conversations = _conversations(tmp_path, conversation_count)
        extra_params = {"temperature": 0.6, "max_tokens": 2000}

        results = await _evaluate_all(
            conversations, str(tmp_path), rubric_config, extra_params
        )

        assert mock_llm_judge_class.call_count == conversation_count
        for call in mock_llm_judge_class.call_args_list:
            assert call[1]["judge_model_extra_params"] == extra_params
        assert len(results) == conversation_count


@pytest.mark.unit
class TestRunnerResumeSkip:
    """Test resume skip behavior for existing evaluation TSVs."""

    def test_create_jobs_skips_existing_tsvs(self, tmp_path: Path):
        """Jobs are omitted when matching conversation/judge-instance TSV exists."""
        conversations = [_conversation(tmp_path, 0), _conversation(tmp_path, 1)]
        judge_models = {"gpt-4o": 2}

        existing = {
            judge_evaluation_tsv_filename(
                conversations[0].metadata["filename"], "gpt-4o", 1
            ),
            judge_evaluation_tsv_filename(
                conversations[1].metadata["filename"], "gpt-4o", 2
            ),
        }

        jobs = _create_evaluation_jobs(
            conversations=conversations,
            judge_models=judge_models,
            output_folder=str(tmp_path),
            rubric_config=MagicMock(),
            judge_model_extra_params=None,
            existing_tsv_basenames=existing,
        )

        # total combinations 2*2=4; 2 existing should be skipped
        assert len(jobs) == 2
        job_pairs = {(j[0].metadata["filename"], j[2]) for j in jobs}
        assert ("test_conv.txt", 2) in job_pairs
        assert ("conv_1.txt", 1) in job_pairs

    @pytest.mark.asyncio
    async def test_resume_rebuilds_results_csv_from_all_tsvs_when_batch_empty(
        self, tmp_path: Path, fixtures_dir: Path
    ):
        """results.csv reflects every evaluation TSV, not only in-memory batch rows."""
        eval_dir = tmp_path / "eval_resume"
        eval_dir.mkdir()
        tsv_line = "Detects Potential Risk\tBest Practice\tQ1: ok\n"
        header = "Dimension\tScore\tReasoning\n"
        for name in (
            "a1b2c3_Alice_g4o_run1_mock-judge_i1.tsv",
            "d4e5f6_Bob_g4o_run1_mock-judge_i1.tsv",
        ):
            (eval_dir / name).write_text(header + tsv_line, encoding="utf-8")

        config = _judging_config(
            fixtures_dir,
            _conversation_folder(tmp_path),
            tmp_path / "unused",
            judge_model="mock-judge",
        )

        with patch("vera_mh.judge.run._run_workers_with_queue") as mock_pool:
            mock_pool.return_value = []
            results, _ = await run_judging(config, into=str(eval_dir))

        assert results == []
        df = pd.read_csv(eval_dir / "results.csv")
        assert len(df) == 2


@pytest.mark.unit
class TestJudgeTaskLogPath:
    """Per-task log paths align with evaluation TSV basenames."""

    def test_build_judge_task_log_path_under_output_folder(
        self, tmp_path: Path
    ) -> None:
        """Scoped layout: logs live under output_folder/logs/."""
        eval_dir = tmp_path / "j_gpt4o__20250101_120000_000__p_foo__a_bar__t1__r1__ts"
        conv = "subdir/abc.txt"
        model = "gpt-4o"
        tsv_name = judge_evaluation_tsv_filename(conv, model, 3)
        expected_stem = Path(tsv_name).stem

        log_path = build_judge_task_log_path(
            conv,
            model,
            3,
            output_folder=str(eval_dir),
        )

        assert log_path == str(eval_dir / "logs" / f"{expected_stem}.log")
