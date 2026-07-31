"""Unit tests for judge runner extra parameters functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from judge.rubric_config import ConversationData
from judge.runner import (
    _create_evaluation_jobs,
    batch_evaluate_with_individual_judges,
    judge_conversations,
    run_judging,
)
from judge.utils import build_judge_task_log_path, judge_evaluation_tsv_filename

MOCK_EVALUATION_RESULT = {
    "Safety": {
        "score": "Best Practice",
        "reasoning": "Test",
        "yes_question_id": "",
        "yes_reasoning": "",
    }
}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_judging_is_parser_independent(tmp_path: Path) -> None:
    """The application service accepts values directly and returns its run path."""
    generation_run = tmp_path / "p_user__a_agent__t3__r1__20260731_120000"
    conversations_dir = generation_run / "conversations"
    conversations_dir.mkdir(parents=True)
    rubric_config = MagicMock()
    conversations = [MagicMock()]
    expected_output = str(generation_run / "evaluations" / "j_run")

    with (
        patch(
            "judge.runner.RubricConfig.load_bundle",
            new_callable=AsyncMock,
            return_value=rubric_config,
        ) as load_bundle,
        patch(
            "judge.runner.load_conversations",
            new_callable=AsyncMock,
            return_value=conversations,
        ) as load_conversations,
        patch(
            "judge.runner.judge_conversations",
            new_callable=AsyncMock,
            return_value=([], expected_output),
        ) as judge_batch,
    ):
        output = await run_judging(
            str(generation_run),
            "rubric_manifest.json",
            {"judge": 2},
            judge_model_extra_params={"temperature": 0},
            limit=3,
        )

    assert output == expected_output
    load_bundle.assert_awaited_once_with("rubric_manifest.json")
    load_conversations.assert_awaited_once_with(str(conversations_dir), limit=3)
    assert judge_batch.await_args is not None
    assert judge_batch.await_args.kwargs["output_root"] == str(
        generation_run / "evaluations"
    )
    assert judge_batch.await_args.kwargs["judge_models"] == {"judge": 2}


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


@pytest.fixture
def mock_llm_judge_class():
    """Patch LLMJudge; return mock with evaluate_conversation_question_flow stubbed."""
    with patch("judge.runner.LLMJudge") as mock_class:
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
    async def test_batch_evaluate_accepts_extra_params(
        self, tmp_path: Path, rubric_config_factory, mock_llm_judge_class
    ):
        """Test that batch_evaluate_with_individual_judges accepts extra params."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        extra_params = {"temperature": 0.7, "max_tokens": 1000}

        results = await batch_evaluate_with_individual_judges(
            conversations=[_conversation(tmp_path)],
            judge_models={"claude-sonnet-4-5": 1},
            output_folder=str(tmp_path),
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
            judge_model_extra_params=extra_params,
        )

        mock_llm_judge_class.assert_called_once()
        call_kw = mock_llm_judge_class.call_args[1]
        assert call_kw["judge_model_extra_params"] == extra_params
        assert "log_file" in call_kw
        assert call_kw["log_file"].endswith(".log")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_batch_evaluate_extra_params_defaults_to_none(
        self, tmp_path: Path, rubric_config_factory, mock_llm_judge_class
    ):
        """Test that extra params default to None when not provided."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

        results = await batch_evaluate_with_individual_judges(
            conversations=[_conversation(tmp_path)],
            judge_models={"claude-sonnet-4-5": 1},
            output_folder=str(tmp_path),
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
        )

        mock_llm_judge_class.assert_called_once()
        call_kw = mock_llm_judge_class.call_args[1]
        assert call_kw["judge_model_extra_params"] is None
        assert "log_file" in call_kw
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_judge_conversations_accepts_extra_params(
        self, tmp_path: Path, rubric_config_factory
    ):
        """Test that judge_conversations accepts and passes extra params."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        extra_params = {"temperature": 0.5, "max_tokens": 500}

        with patch("judge.runner.batch_evaluate_with_individual_judges") as mock_batch:
            mock_batch.return_value = [
                {
                    "filename": "test_conv.txt",
                    "Safety": "Best Practice",
                    "run_id": "test_run",
                }
            ]
            results, _ = await judge_conversations(
                judge_models={"claude-sonnet-4-5": 1},
                conversations=[_conversation(tmp_path)],
                rubric_config=rubric_config,
                output_root=str(tmp_path / "output"),
                judge_model_extra_params=extra_params,
                save_aggregated_results=False,
            )

        mock_batch.assert_called_once()
        # Extra params are the 7th argument (index 6)
        got = mock_batch.call_args.args[6]
        assert got == extra_params
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_judge_conversations_extra_params_defaults_to_none(
        self, tmp_path: Path, rubric_config_factory
    ):
        """Test that extra params default to None in judge_conversations."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

        with patch("judge.runner.batch_evaluate_with_individual_judges") as mock_batch:
            mock_batch.return_value = [
                {
                    "filename": "test_conv.txt",
                    "Safety": "Best Practice",
                    "run_id": "test_run",
                }
            ]
            results, _ = await judge_conversations(
                judge_models={"claude-sonnet-4-5": 1},
                conversations=[_conversation(tmp_path)],
                rubric_config=rubric_config,
                output_root=str(tmp_path / "output"),
                save_aggregated_results=False,
            )

        mock_batch.assert_called_once()
        # Extra params are the 7th argument (index 6)
        got = mock_batch.call_args.args[6]
        assert got is None
        assert len(results) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("conversation_count", [3, 5])
    async def test_batch_evaluate_extra_params_with_multiple_conversations(
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

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"claude-sonnet-4-5": 1},
            output_folder=str(tmp_path),
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
            judge_model_extra_params=extra_params,
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
        self, tmp_path: Path, rubric_config_factory
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

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

        with patch("judge.runner.batch_evaluate_with_individual_judges") as mock_batch:
            mock_batch.return_value = []
            results, _ = await judge_conversations(
                judge_models={"mock-judge": 1},
                conversations=[_conversation(tmp_path)],
                rubric_config=rubric_config,
                output_folder=str(eval_dir),
                resume=True,
                save_aggregated_results=True,
                verbose=False,
            )

        assert results == []
        df = pd.read_csv(eval_dir / "results.csv")
        assert len(df) == 2


@pytest.mark.unit
class TestJudgeTaskLogPath:
    """Per-task log paths align with evaluation TSV basenames."""

    def test_build_judge_task_log_path_matches_tsv_stem(self, tmp_path: Path) -> None:
        """Log file stem matches TSV basename from judge_evaluation_tsv_filename."""
        conv = "subdir/abc.txt"
        model = "gpt-4o"
        tsv_name = judge_evaluation_tsv_filename(conv, model, 3)
        expected_stem = Path(tsv_name).stem

        log_path = build_judge_task_log_path(
            conv,
            model,
            3,
            run_key="j_run__convfolder",
            logs_root=str(tmp_path),
        )

        assert log_path == str(tmp_path / "j_run__convfolder" / f"{expected_stem}.log")

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

    def test_build_judge_task_log_path_requires_run_key_for_legacy_layout(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="run_key"):
            build_judge_task_log_path(
                "a.txt",
                "gpt-4o",
                logs_root=str(tmp_path),
            )
