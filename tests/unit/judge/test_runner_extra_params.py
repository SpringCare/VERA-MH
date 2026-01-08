"""Unit tests for judge runner extra parameters functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from judge.rubric_config import ConversationData
from judge.runner import batch_evaluate_with_individual_judges, judge_conversations


@pytest.mark.unit
class TestRunnerExtraParams:
    """Test that extra parameters are properly handled in runner functions."""

    @pytest.mark.asyncio
    async def test_batch_evaluate_accepts_extra_params(
        self, tmp_path: Path, rubric_config_factory
    ):
        """Test that batch_evaluate_with_individual_judges accepts extra params."""
        # Create test conversation
        conversation = ConversationData(
            content="User: Hello\nAssistant: Hi!",
            metadata={
                "filename": "test_conv.txt",
                "run_id": "test",
                "source_path": str(tmp_path / "test_conv.txt"),
            },
        )

        # Load rubric config
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

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
                conversations=[conversation],
                judge_models={"claude-3-7-sonnet": 1},
                output_folder=str(tmp_path),
                rubric_config=rubric_config,
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
    async def test_batch_evaluate_extra_params_defaults_to_none(
        self, tmp_path: Path, rubric_config_factory
    ):
        """Test that extra params default to None when not provided."""
        conversation = ConversationData(
            content="User: Hello\nAssistant: Hi!",
            metadata={
                "filename": "test_conv.txt",
                "run_id": "test",
                "source_path": str(tmp_path / "test_conv.txt"),
            },
        )

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

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
                conversations=[conversation],
                judge_models={"claude-3-7-sonnet": 1},
                output_folder=str(tmp_path),
                rubric_config=rubric_config,
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
    async def test_judge_conversations_accepts_extra_params(
        self, tmp_path: Path, rubric_config_factory
    ):
        """Test that judge_conversations accepts and passes extra params."""
        # Create test conversation
        conversation = ConversationData(
            content="User: Hello\nAssistant: Hi!",
            metadata={
                "filename": "test_conv.txt",
                "run_id": "test",
                "source_path": str(tmp_path / "test_conv.txt"),
            },
        )

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

            results = await judge_conversations(
                judge_models={"claude-3-7-sonnet": 1},
                conversations=[conversation],
                rubric_config=rubric_config,
                output_root=str(tmp_path / "output"),
                judge_model_extra_params=extra_params,
                save_aggregated_results=False,
            )

            # Verify batch function was called with extra params
            mock_batch.assert_called_once()
            # Function called as: batch_evaluate_with_individual_judges(
            #   conversations, judge_models, output_folder, rubric_config,
            #   max_concurrent, per_judge, judge_model_extra_params)
            # extra_params is the 7th positional argument (index 6)
            assert mock_batch.call_args.args[6] == extra_params
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_judge_conversations_extra_params_defaults_to_none(
        self, tmp_path: Path, rubric_config_factory
    ):
        """Test that extra params default to None in judge_conversations."""
        conversation = ConversationData(
            content="User: Hello\nAssistant: Hi!",
            metadata={
                "filename": "test_conv.txt",
                "run_id": "test",
                "source_path": str(tmp_path / "test_conv.txt"),
            },
        )

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

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
                conversations=[conversation],
                rubric_config=rubric_config,
                output_root=str(tmp_path / "output"),
                save_aggregated_results=False,
                # No extra params provided
            )

            # Verify batch function was called with None for extra params
            mock_batch.assert_called_once()
            # Check that judge_model_extra_params defaults to None (7th arg, index 6)
            assert mock_batch.call_args.args[6] is None
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_multiple_conversations_with_extra_params(
        self, tmp_path: Path, rubric_config_factory
    ):
        """Test that extra params are used for all conversations in batch."""
        # Create multiple conversations
        conversations = [
            ConversationData(
                content=f"User: Hello {i}\nAssistant: Hi {i}!",
                metadata={
                    "filename": f"conv_{i}.txt",
                    "run_id": f"test_{i}",
                    "source_path": str(tmp_path / f"conv_{i}.txt"),
                },
            )
            for i in range(3)
        ]

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

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
                conversations=conversations,
                judge_models={"claude-3-7-sonnet": 1},
                output_folder=str(tmp_path),
                rubric_config=rubric_config,
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
    async def test_extra_params_with_multiple_conversations(
        self, tmp_path: Path, rubric_config_factory
    ):
        """Test that extra params work correctly with multiple conversations."""
        # Create 5 conversations but we'll pass all
        conversations = [
            ConversationData(
                content=f"User: Hello {i}\nAssistant: Hi {i}!",
                metadata={
                    "filename": f"conv_{i}.txt",
                    "run_id": f"test_{i}",
                    "source_path": str(tmp_path / f"conv_{i}.txt"),
                },
            )
            for i in range(5)
        ]

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

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
                conversations=conversations,
                judge_models={"claude-3-7-sonnet": 1},
                output_folder=str(tmp_path),
                rubric_config=rubric_config,
                max_concurrent=None,
                per_judge=False,
                judge_model_extra_params=extra_params,
            )

            # Should create 5 judges (one per conversation)
            assert mock_judge_class.call_count == 5

            # Verify all calls included extra params
            for call in mock_judge_class.call_args_list:
                call_kwargs = call[1]
                assert call_kwargs["judge_model_extra_params"] == extra_params

            assert len(results) == 5
