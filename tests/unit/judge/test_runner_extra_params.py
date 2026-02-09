"""Unit tests for judge runner extra parameters functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from judge.rubric_config import ConversationData
from judge.runner import batch_evaluate_with_individual_judges, judge_conversations

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
            judge_models={"claude-3-7-sonnet": 1},
            output_folder=str(tmp_path),
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
            judge_model_extra_params=extra_params,
        )

        mock_llm_judge_class.assert_called_once()
        call_kw = mock_llm_judge_class.call_args[1]
        assert call_kw["judge_model_extra_params"] == extra_params
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_batch_evaluate_extra_params_defaults_to_none(
        self, tmp_path: Path, rubric_config_factory, mock_llm_judge_class
    ):
        """Test that extra params default to None when not provided."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

        results = await batch_evaluate_with_individual_judges(
            conversations=[_conversation(tmp_path)],
            judge_models={"claude-3-7-sonnet": 1},
            output_folder=str(tmp_path),
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
        )

        mock_llm_judge_class.assert_called_once()
        call_kw = mock_llm_judge_class.call_args[1]
        assert call_kw["judge_model_extra_params"] is None
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
                judge_models={"claude-3-7-sonnet": 1},
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
                judge_models={"claude-3-7-sonnet": 1},
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
            judge_models={"claude-3-7-sonnet": 1},
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
