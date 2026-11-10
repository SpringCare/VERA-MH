"""Judging hands each judge model's extra params to the judge, and skips the
evaluations already on disk."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from tests.mocks.run_record import minimal_run_record
from vera_mh.judge.run import run_judging
from vera_mh.judge.utils import build_judge_task_log_path, judge_evaluation_tsv_filename
from vera_mh.utils.config_schema import JudgingConfig, ModelSpec, RubricFiles


def _conversation_folder(tmp_path: Path, count: int = 1) -> Path:
    """A folder of `count` conversation files, for `run_judging` to load."""
    folder = tmp_path / "conversations"
    folder.mkdir()
    for index in range(count):
        name = "test_conv.txt" if index == 0 else f"conv_{index}.txt"
        (folder / name).write_text(
            f"user: Hello {index}\nchatbot: Hi {index}!", encoding="utf-8"
        )
    return folder


def _judging_config(
    fixtures_dir: Path,
    conversations: Path,
    output: Path,
    judge_model: str = "claude-sonnet-4-5",
    repeats: int = 1,
    extra_params: dict | None = None,
) -> JudgingConfig:
    """A judging config using the `rubric_simple.tsv` fixture rubric."""
    return JudgingConfig(
        models=[
            ModelSpec(
                name=judge_model, repeats=repeats, extra_params=extra_params or {}
            )
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


@pytest.fixture
def mock_llm_judge_class():
    """Patch the `LLMJudge` the `Evaluation` job builds; its evaluation does
    nothing, so no TSV is written."""
    with patch("vera_mh.judge.evaluation.LLMJudge") as mock_class:
        mock_class.return_value.evaluate_conversation_question_flow = AsyncMock()
        yield mock_class


@pytest.mark.unit
class TestRunnerExtraParams:
    """Every evaluation gets its judge model's extra params."""

    @pytest.mark.asyncio
    async def test_run_judging_passes_the_models_extra_params(
        self, tmp_path: Path, fixtures_dir: Path, mock_llm_judge_class
    ):
        extra_params = {"temperature": 0.5, "max_tokens": 500}
        config = _judging_config(
            fixtures_dir,
            _conversation_folder(tmp_path),
            tmp_path / "output",
            extra_params=extra_params,
        )

        await run_judging(config, run_record=minimal_run_record())

        mock_llm_judge_class.assert_called_once()
        call_kw = mock_llm_judge_class.call_args.kwargs
        assert call_kw["judge_model_extra_params"] == extra_params
        assert call_kw["log_file"].endswith(".log")

    @pytest.mark.asyncio
    async def test_run_judging_without_extra_params_passes_an_empty_dict(
        self, tmp_path: Path, fixtures_dir: Path, mock_llm_judge_class
    ):
        config = _judging_config(
            fixtures_dir, _conversation_folder(tmp_path), tmp_path / "output"
        )

        await run_judging(config, run_record=minimal_run_record())

        call_kw = mock_llm_judge_class.call_args.kwargs
        assert call_kw["judge_model_extra_params"] == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("conversation_count", [3, 5])
    async def test_extra_params_reach_every_conversation(
        self,
        tmp_path: Path,
        fixtures_dir: Path,
        mock_llm_judge_class,
        conversation_count: int,
    ):
        extra_params = {"temperature": 0.6, "max_tokens": 2000}
        config = _judging_config(
            fixtures_dir,
            _conversation_folder(tmp_path, conversation_count),
            tmp_path / "output",
            extra_params=extra_params,
        )

        await run_judging(config, run_record=minimal_run_record())

        assert mock_llm_judge_class.call_count == conversation_count
        for call in mock_llm_judge_class.call_args_list:
            assert call.kwargs["judge_model_extra_params"] == extra_params


@pytest.mark.unit
class TestRunnerResumeSkip:
    """`--into` judges only what isn't on disk, and results.csv covers it all."""

    @pytest.mark.asyncio
    async def test_into_skips_evaluations_whose_tsv_exists(
        self, tmp_path: Path, fixtures_dir: Path, mock_llm_judge_class
    ):
        eval_dir = tmp_path / "eval_resume"
        eval_dir.mkdir()
        header = "Dimension\tScore\tReasoning\n"
        for conversation, instance in (("test_conv.txt", 1), ("conv_1.txt", 2)):
            name = judge_evaluation_tsv_filename(conversation, "gpt-4o", instance)
            (eval_dir / name).write_text(header, encoding="utf-8")
        config = _judging_config(
            fixtures_dir,
            _conversation_folder(tmp_path, 2),
            tmp_path / "unused",
            judge_model="gpt-4o",
            repeats=2,
        )

        await run_judging(config, into=str(eval_dir))

        judge = mock_llm_judge_class.return_value
        judged = {
            (call.args[0].metadata["filename"], call.kwargs["judge_instance"])
            for call in judge.evaluate_conversation_question_flow.call_args_list
        }
        # 2 conversations x 2 instances = 4; the 2 on disk are skipped.
        assert judged == {("test_conv.txt", 2), ("conv_1.txt", 1)}

    @pytest.mark.asyncio
    async def test_resume_rebuilds_results_csv_from_all_tsvs(
        self, tmp_path: Path, fixtures_dir: Path, mock_llm_judge_class
    ):
        """results.csv holds every evaluation TSV in the folder, including the
        ones this call skipped."""
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

        folder = await run_judging(config, into=str(eval_dir))

        assert folder == str(eval_dir)
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
