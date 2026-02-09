"""Unit tests for judge model extra parameters functionality."""

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from judge.llm_judge import LLMJudge
from judge.rubric_config import ConversationData, RubricConfig
from utils.utils import parse_key_value_list


def _setup_extra_params_arg(argv: list[str]) -> dict[str, Any]:
    """Parse argv and return args.judge_model_extra_params
    (same type as judge.py CLI --judge-model-extra-params argument)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-model-extra-params",
        "-jep",
        help="Extra parameters for the judge model (key=value, comma-separated)",
        type=parse_key_value_list,
        default={},
    )
    args = parser.parse_args(argv)
    return args.judge_model_extra_params


@pytest.mark.unit
class TestJudgeExtraParams:
    """Test that extra parameters are properly passed through the judge system."""

    async def test_llm_judge_accepts_extra_params(self, rubric_config_factory):
        """Test that LLMJudge accepts judge_model_extra_params parameter."""
        extra_params = _setup_extra_params_arg(
            ["-jep", "temperature=0.7,max_tokens=1000"]
        )

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_config=rubric_config,
        )

        assert judge.judge_model_extra_params == extra_params

    async def test_llm_judge_extra_params_defaults_to_temperature_zero(
        self, rubric_config_factory
    ):
        """Test that judge_model_extra_params defaults to temperature=0."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_config=rubric_config,
        )

        assert judge.judge_model_extra_params == {"temperature": 0}

    async def test_llm_judge_stores_extra_params_correctly(self, rubric_config_factory):
        """Test that LLMJudge stores extra params and makes them available."""
        extra_params = _setup_extra_params_arg(
            ["-jep", "temperature=0.5,max_tokens=500,top_p=0.9"]
        )

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(
            judge_model="claude-3-7-sonnet",
            judge_model_extra_params=extra_params,
            rubric_config=rubric_config,
        )

        # Verify extra params are stored correctly
        assert judge.judge_model_extra_params == extra_params
        assert judge.judge_model_extra_params["temperature"] == 0.5
        assert judge.judge_model_extra_params["max_tokens"] == 500
        assert judge.judge_model_extra_params["top_p"] == 0.9

        # Verify standard model param is still accessible
        assert judge.judge_model == "claude-3-7-sonnet"

    @pytest.mark.asyncio
    async def test_llm_judge_passes_extra_params_in_async_evaluation(
        self, tmp_path: Path, rubric_config_factory, fixtures_dir: Path
    ):
        """Test that extra params are passed to LLMFactory during async evaluation."""
        extra_params = _setup_extra_params_arg(
            ["-jep", "temperature=0.7,max_tokens=1000"]
        )
        captured_kwargs = {}

        # Create a simple rubric prompt file for testing (without persona placeholders)
        test_rubric_dir = tmp_path / "test_rubrics"
        test_rubric_dir.mkdir()

        # Copy the simple rubric file
        import shutil

        shutil.copy(
            fixtures_dir / "rubric_simple.tsv", test_rubric_dir / "rubric_simple.tsv"
        )
        shutil.copy(
            fixtures_dir / "question_prompt.txt",
            test_rubric_dir / "question_prompt.txt",
        )

        # Create a simple rubric prompt that only uses {conversation}
        simple_prompt = (
            "You are evaluating the following conversation:\n\n"
            "{conversation}\n\n"
            "Answer the following questions carefully."
        )
        (test_rubric_dir / "rubric_prompt_beginning.txt").write_text(simple_prompt)

        def create_mock_llm(**kwargs):
            """Capture kwargs and return a MockLLM that supports structured output."""
            # Store all kwargs for verification
            captured_kwargs.update(kwargs)

            # Import MockLLM for proper JudgeLLM implementation
            from tests.mocks.mock_llm import MockLLM

            # Create MockLLM with captured kwargs
            return MockLLM(
                name=kwargs.get("name", "mock-llm"),
                system_prompt=kwargs.get("system_prompt"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
                responses=[
                    '{"answer": "No", "reasoning": "Test reasoning", '
                    '"question_text": "Test question"}'
                ],
            )

        # Patch LLMFactory.create_llm to capture parameters
        with patch(
            "judge.llm_judge.LLMFactory.create_llm", side_effect=create_mock_llm
        ):
            # Load rubric config from test directory
            rubric_config = await RubricConfig.load(
                rubric_folder=str(test_rubric_dir),
                rubric_file="rubric_simple.tsv",
                rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            )

            judge = LLMJudge(
                judge_model="claude-3-7-sonnet",
                judge_model_extra_params=extra_params,
                rubric_config=rubric_config,
            )

            # Create a simple conversation for testing
            conversation = ConversationData(
                content="User: Hello\nAssistant: Hi there!",
                metadata={
                    "filename": "test_conversation.txt",
                    "run_id": "test",
                    "source_path": str(tmp_path / "test_conversation.txt"),
                },
            )

            # Run async evaluation - this will trigger LLM creation
            result = await judge.evaluate_conversation_question_flow(
                conversation,
                output_folder=str(tmp_path),
                auto_save=False,
            )

            # Verify evaluation completed
            assert result is not None
            assert isinstance(result, dict)

            # Verify create_llm was called with extra params
            assert (
                "temperature" in captured_kwargs
            ), f"Expected temperature in {captured_kwargs}"
            assert (
                captured_kwargs["temperature"] == 0.7
            ), f"Expected temperature=0.7, got {captured_kwargs.get('temperature')}"
            assert (
                "max_tokens" in captured_kwargs
            ), f"Expected max_tokens in {captured_kwargs}"
            assert (
                captured_kwargs["max_tokens"] == 1000
            ), f"Expected max_tokens=1000, got {captured_kwargs.get('max_tokens')}"
            assert captured_kwargs["model_name"] == "claude-3-7-sonnet"

    async def test_llm_judge_extra_params_with_none(self, rubric_config_factory):
        """Test that passing None for extra_params sets default temperature=0."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=None,
            rubric_config=rubric_config,
        )

        assert judge.judge_model_extra_params == {"temperature": 0}

    async def test_llm_judge_preserves_standard_params(self, rubric_config_factory):
        """Test that extra params don't interfere with standard parameters."""
        extra_params = _setup_extra_params_arg(["-jep", "temperature=0.8"])

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(
            judge_model="claude-3-7-sonnet",
            judge_model_extra_params=extra_params,
            rubric_config=rubric_config,
        )

        # Standard params should still be accessible
        assert judge.judge_model == "claude-3-7-sonnet"
        assert judge.judge_model_extra_params == extra_params

    async def test_multiple_extra_params_types(self, rubric_config_factory):
        """Test that extra params can contain various types."""
        extra_params = _setup_extra_params_arg(
            ["-jep", "temperature=0.7,max_tokens=1000,top_p=0.95"]
        )

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_config=rubric_config,
        )

        assert judge.judge_model_extra_params == extra_params
        assert isinstance(judge.judge_model_extra_params["temperature"], float)
        assert isinstance(judge.judge_model_extra_params["max_tokens"], int)

    async def test_user_provided_temperature_overrides_default(
        self, rubric_config_factory
    ):
        """Test that user-provided temperature overrides the default of 0."""
        extra_params = _setup_extra_params_arg(
            ["-jep", "temperature=0.9,max_tokens=2000"]
        )

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_config=rubric_config,
        )

        # User-provided temperature should override the default
        assert judge.judge_model_extra_params["temperature"] == 0.9
        assert judge.judge_model_extra_params["max_tokens"] == 2000

    async def test_default_temperature_added_when_other_params_provided(
        self, rubric_config_factory
    ):
        """Test that default temperature=0 is added when user provides other params."""
        extra_params = _setup_extra_params_arg(["-jep", "max_tokens=500,top_p=0.9"])

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_config=rubric_config,
        )

        # Default temperature should be added along with user params
        assert judge.judge_model_extra_params["temperature"] == 0
        assert judge.judge_model_extra_params["max_tokens"] == 500
        assert judge.judge_model_extra_params["top_p"] == 0.9
        # Verify all params are present
        assert len(judge.judge_model_extra_params) == 3

    async def test_temperature_zero_is_preserved_when_explicitly_set(
        self, rubric_config_factory
    ):
        """Test that explicitly setting temperature=0 is preserved (not overridden)."""
        extra_params = _setup_extra_params_arg(
            ["-jep", "temperature=0,max_tokens=1000"]
        )

        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_config=rubric_config,
        )

        # Explicitly set temperature=0 should be preserved
        assert judge.judge_model_extra_params["temperature"] == 0
        assert judge.judge_model_extra_params["max_tokens"] == 1000

    async def test_various_temperature_values_override_default(
        self, rubric_config_factory
    ):
        """Test that various temperature values properly override the default."""
        test_cases = [
            {"temperature": 0.5, "expected": 0.5},
            {"temperature": 1.0, "expected": 1.0},
            {"temperature": 0.1, "expected": 0.1},
            {"temperature": 0.99, "expected": 0.99},
        ]

        for test_case in test_cases:
            extra_params = _setup_extra_params_arg(
                ["-jep", f"temperature={test_case['temperature']}"]
            )
            rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
            judge = LLMJudge(
                judge_model="mock-llm",
                judge_model_extra_params=extra_params,
                rubric_config=rubric_config,
            )
            assert (
                judge.judge_model_extra_params["temperature"] == test_case["expected"]
            ), f"Failed for temperature={test_case['temperature']}"
