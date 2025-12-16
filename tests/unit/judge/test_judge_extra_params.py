"""Unit tests for judge model extra parameters functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest

from judge.llm_judge import LLMJudge


@pytest.mark.unit
class TestJudgeExtraParams:
    """Test that extra parameters are properly passed through the judge system."""

    def test_llm_judge_accepts_extra_params(self, fixtures_dir: Path):
        """Test that LLMJudge accepts judge_model_extra_params parameter."""
        extra_params = {"temperature": 0.7, "max_tokens": 1000}

        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        assert judge.judge_model_extra_params == extra_params

    def test_llm_judge_extra_params_defaults_to_temperature_zero(
        self, fixtures_dir: Path
    ):
        """Test that judge_model_extra_params defaults to temperature=0."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        assert judge.judge_model_extra_params == {"temperature": 0}

    def test_llm_judge_stores_extra_params_correctly(self, fixtures_dir: Path):
        """Test that LLMJudge stores extra params and makes them available."""
        extra_params = {"temperature": 0.5, "max_tokens": 500, "top_p": 0.9}

        judge = LLMJudge(
            judge_model="claude-3-7-sonnet",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
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
        self, tmp_path: Path, fixtures_dir: Path
    ):
        """Test that extra params are passed to LLMFactory during async evaluation."""
        extra_params = {"temperature": 0.7, "max_tokens": 1000}
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
            judge = LLMJudge(
                judge_model="claude-3-7-sonnet",
                judge_model_extra_params=extra_params,
                rubric_folder=str(test_rubric_dir),
                rubric_file="rubric_simple.tsv",
                rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            )

            # Create a simple conversation for testing
            conversation_file = tmp_path / "test_conversation.txt"
            conversation_file.write_text("User: Hello\nAssistant: Hi there!")

            # Run async evaluation - this will trigger LLM creation
            result = await judge.evaluate_conversation_question_flow(
                str(conversation_file),
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

    def test_llm_judge_extra_params_with_none(self, fixtures_dir: Path):
        """Test that passing None for extra_params sets default temperature=0."""
        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=None,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        assert judge.judge_model_extra_params == {"temperature": 0}

    def test_llm_judge_preserves_standard_params(self, fixtures_dir: Path):
        """Test that extra params don't interfere with standard parameters."""
        extra_params = {"temperature": 0.8}

        judge = LLMJudge(
            judge_model="claude-3-7-sonnet",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Standard params should still be accessible
        assert judge.judge_model == "claude-3-7-sonnet"
        assert judge.judge_model_extra_params == extra_params

    def test_multiple_extra_params_types(self, fixtures_dir: Path):
        """Test that extra params can contain various types."""
        extra_params = {
            "temperature": 0.7,  # float
            "max_tokens": 1000,  # int
            "top_p": 0.95,  # float
            "stop_sequences": ["END"],  # list
        }

        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        assert judge.judge_model_extra_params == extra_params
        assert isinstance(judge.judge_model_extra_params["temperature"], float)
        assert isinstance(judge.judge_model_extra_params["max_tokens"], int)
        assert isinstance(judge.judge_model_extra_params["stop_sequences"], list)

    def test_user_provided_temperature_overrides_default(self, fixtures_dir: Path):
        """Test that user-provided temperature overrides the default of 0."""
        extra_params = {"temperature": 0.9, "max_tokens": 2000}

        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # User-provided temperature should override the default
        assert judge.judge_model_extra_params["temperature"] == 0.9
        assert judge.judge_model_extra_params["max_tokens"] == 2000

    def test_default_temperature_added_when_other_params_provided(
        self, fixtures_dir: Path
    ):
        """Test that default temperature=0 is added when user provides other params."""
        extra_params = {"max_tokens": 500, "top_p": 0.9}

        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Default temperature should be added along with user params
        assert judge.judge_model_extra_params["temperature"] == 0
        assert judge.judge_model_extra_params["max_tokens"] == 500
        assert judge.judge_model_extra_params["top_p"] == 0.9
        # Verify all params are present
        assert len(judge.judge_model_extra_params) == 3

    def test_temperature_zero_is_preserved_when_explicitly_set(
        self, fixtures_dir: Path
    ):
        """Test that explicitly setting temperature=0 is preserved (not overridden)."""
        extra_params = {"temperature": 0, "max_tokens": 1000}

        judge = LLMJudge(
            judge_model="mock-llm",
            judge_model_extra_params=extra_params,
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Explicitly set temperature=0 should be preserved
        assert judge.judge_model_extra_params["temperature"] == 0
        assert judge.judge_model_extra_params["max_tokens"] == 1000

    def test_various_temperature_values_override_default(self, fixtures_dir: Path):
        """Test that various temperature values properly override the default."""
        test_cases = [
            {"temperature": 0.5, "expected": 0.5},
            {"temperature": 1.0, "expected": 1.0},
            {"temperature": 0.1, "expected": 0.1},
            {"temperature": 0.99, "expected": 0.99},
        ]

        for test_case in test_cases:
            extra_params = {"temperature": test_case["temperature"]}
            judge = LLMJudge(
                judge_model="mock-llm",
                judge_model_extra_params=extra_params,
                rubric_folder=str(fixtures_dir),
                rubric_file="rubric_simple.tsv",
                rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            )
            assert (
                judge.judge_model_extra_params["temperature"] == test_case["expected"]
            ), f"Failed for temperature={test_case['temperature']}"
