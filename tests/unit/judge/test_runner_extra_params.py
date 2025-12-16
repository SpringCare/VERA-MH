"""Unit tests for judge runner extra parameters functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from judge.runner import batch_evaluate_with_individual_judges, judge_conversations


@pytest.mark.unit
class TestRunnerExtraParams:
    """Test that extra parameters are properly handled in runner functions."""

    @pytest.mark.asyncio
    async def test_batch_evaluate_accepts_extra_params(self, tmp_path: Path):
        """Test that batch_evaluate_with_individual_judges accepts extra params."""
        # Create test conversation file
        conv_file = tmp_path / "test_conv.txt"
        conv_file.write_text("User: Hello\nAssistant: Hi!")

        extra_params = {"temperature": 0.7, "max_tokens": 1000}

        with patch("judge.runner.LLMJudge") as mock_judge_class:
            mock_judge = MagicMock()
            mock_judge.evaluate_conversation_question_flow = AsyncMock(
                return_value={
                    "Safety": {
                        "score": "Best Practice",
                        "reasoning": "Test",
                        "yes_question_id": "",
                        "yes_reasoning": "",
                    }
                }
            )
            mock_judge_class.return_value = mock_judge

            results = await batch_evaluate_with_individual_judges(
                conversation_file_paths=[str(conv_file)],
                judge_models={"claude-3-7-sonnet": 1},
                output_folder=str(tmp_path),
                limit=None,
                max_concurrent=None,
                per_judge=False,
                judge_model_extra_params=extra_params,
            )

            # Verify LLMJudge was created with extra params
            mock_judge_class.assert_called_once()
            call_kwargs = mock_judge_class.call_args[1]
            assert "judge_model_extra_params" in call_kwargs
            assert call_kwargs["judge_model_extra_params"] == extra_params
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_batch_evaluate_extra_params_defaults_to_none(self, tmp_path: Path):
        """Test that extra params default to None when not provided."""
        conv_file = tmp_path / "test_conv.txt"
        conv_file.write_text("User: Hello\nAssistant: Hi!")

        with patch("judge.runner.LLMJudge") as mock_judge_class:
            mock_judge = MagicMock()
            mock_judge.evaluate_conversation_question_flow = AsyncMock(
                return_value={
                    "Safety": {
                        "score": "Best Practice",
                        "reasoning": "Test",
                        "yes_question_id": "",
                        "yes_reasoning": "",
                    }
                }
            )
            mock_judge_class.return_value = mock_judge

            results = await batch_evaluate_with_individual_judges(
                conversation_file_paths=[str(conv_file)],
                judge_models={"claude-3-7-sonnet": 1},
                output_folder=str(tmp_path),
                limit=None,
                max_concurrent=None,
                per_judge=False,
                # No extra params provided
            )

            # Verify LLMJudge was created with None for extra params
            mock_judge_class.assert_called_once()
            call_kwargs = mock_judge_class.call_args[1]
            assert "judge_model_extra_params" in call_kwargs
            assert call_kwargs["judge_model_extra_params"] is None
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_judge_conversations_accepts_extra_params(self, tmp_path: Path):
        """Test that judge_conversations accepts and passes extra params."""
        # Create conversation folder with test file
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        conv_file = conv_folder / "test_conv.txt"
        conv_file.write_text("User: Hello\nAssistant: Hi!")

        extra_params = {"temperature": 0.5, "max_tokens": 500}

        with patch("judge.runner.batch_evaluate_with_individual_judges") as mock_batch:
            mock_batch.return_value = [
                {
                    "filename": "test_conv.txt",
                    "Safety": "Best Practice",
                    "run_id": "test_run",
                }
            ]

            results = await judge_conversations(
                judge_models={"claude-3-7-sonnet": 1},
                conversation_folder=str(conv_folder),
                output_root=str(tmp_path / "output"),
                judge_model_extra_params=extra_params,
                save_aggregated_results=False,
            )

            # Verify batch function was called with extra params
            mock_batch.assert_called_once()
            call_args = mock_batch.call_args[0]
            # Arguments: conversation_file_paths, judge_models, output_folder,
            #            limit, max_concurrent, per_judge, judge_model_extra_params
            assert len(call_args) == 7
            assert call_args[6] == extra_params  # judge_model_extra_params is 7th arg
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_judge_conversations_extra_params_defaults_to_none(
        self, tmp_path: Path
    ):
        """Test that extra params default to None in judge_conversations."""
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        conv_file = conv_folder / "test_conv.txt"
        conv_file.write_text("User: Hello\nAssistant: Hi!")

        with patch("judge.runner.batch_evaluate_with_individual_judges") as mock_batch:
            mock_batch.return_value = [
                {
                    "filename": "test_conv.txt",
                    "Safety": "Best Practice",
                    "run_id": "test_run",
                }
            ]

            results = await judge_conversations(
                judge_models={"claude-3-7-sonnet": 1},
                conversation_folder=str(conv_folder),
                output_root=str(tmp_path / "output"),
                save_aggregated_results=False,
                # No extra params provided
            )

            # Verify batch function was called with None for extra params
            mock_batch.assert_called_once()
            call_args = mock_batch.call_args[0]
            # Arguments: conversation_file_paths, judge_models, output_folder,
            #            limit, max_concurrent, per_judge, judge_model_extra_params
            assert len(call_args) == 7
            assert call_args[6] is None  # judge_model_extra_params is 7th arg
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_multiple_conversations_with_extra_params(self, tmp_path: Path):
        """Test that extra params are used for all conversations in batch."""
        # Create multiple conversation files
        conv_files = []
        for i in range(3):
            conv_file = tmp_path / f"conv_{i}.txt"
            conv_file.write_text(f"User: Hello {i}\nAssistant: Hi {i}!")
            conv_files.append(str(conv_file))

        extra_params = {"temperature": 0.8, "max_tokens": 2000}

        with patch("judge.runner.LLMJudge") as mock_judge_class:
            mock_judge = MagicMock()
            mock_judge.evaluate_conversation_question_flow = AsyncMock(
                return_value={
                    "Safety": {
                        "score": "Best Practice",
                        "reasoning": "Test",
                        "yes_question_id": "",
                        "yes_reasoning": "",
                    }
                }
            )
            mock_judge_class.return_value = mock_judge

            results = await batch_evaluate_with_individual_judges(
                conversation_file_paths=conv_files,
                judge_models={"claude-3-7-sonnet": 1},
                output_folder=str(tmp_path),
                limit=None,
                max_concurrent=None,
                per_judge=False,
                judge_model_extra_params=extra_params,
            )

            # Verify LLMJudge was created 3 times (once per conversation)
            assert mock_judge_class.call_count == 3

            # Verify all calls included extra params
            for call in mock_judge_class.call_args_list:
                call_kwargs = call[1]
                assert "judge_model_extra_params" in call_kwargs
                assert call_kwargs["judge_model_extra_params"] == extra_params

            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_extra_params_with_limit(self, tmp_path: Path):
        """Test that extra params work correctly when limit is applied."""
        # Create multiple conversation files
        conv_files = []
        for i in range(5):
            conv_file = tmp_path / f"conv_{i}.txt"
            conv_file.write_text(f"User: Hello {i}\nAssistant: Hi {i}!")
            conv_files.append(str(conv_file))

        extra_params = {"temperature": 0.6}

        with patch("judge.runner.LLMJudge") as mock_judge_class:
            mock_judge = MagicMock()
            mock_judge.evaluate_conversation_question_flow = AsyncMock(
                return_value={
                    "Safety": {
                        "score": "Best Practice",
                        "reasoning": "Test",
                        "yes_question_id": "",
                        "yes_reasoning": "",
                    }
                }
            )
            mock_judge_class.return_value = mock_judge

            results = await batch_evaluate_with_individual_judges(
                conversation_file_paths=conv_files,
                judge_models={"claude-3-7-sonnet": 1},
                output_folder=str(tmp_path),
                limit=2,  # Only process first 2
                max_concurrent=None,
                per_judge=False,
                judge_model_extra_params=extra_params,
            )

            # Should only create 2 judges (due to limit)
            assert mock_judge_class.call_count == 2

            # Verify all calls included extra params
            for call in mock_judge_class.call_args_list:
                call_kwargs = call[1]
                assert call_kwargs["judge_model_extra_params"] == extra_params

            assert len(results) == 2
