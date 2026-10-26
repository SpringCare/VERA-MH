"""Unit tests for judge utility functions."""

import argparse
from datetime import datetime

import pytest

from vera_mh.judge.utils import (
    build_single_conversation_judge_run_key,
    parse_judge_models,
)


def _setup_judge_model_arg(argv: list[str]) -> list[str]:
    """Parse argv and return args.judge_model (same type as judge.py CLI)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-model",
        "-j",
        nargs="+",
        required=True,
        help="Model(s) to use for judging; format 'model' or 'model:count'",
    )
    args = parser.parse_args(argv)
    return args.judge_model


class TestJudgeModelParsing:
    """Test parsing of --judge-model CLI argument (same nargs='+' list as judge.py)."""

    def test_single_model(self):
        """Test parsing a single model without count."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 1}

    def test_single_model_with_count(self):
        """Test parsing a single model with count."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:3"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 3}

    def test_multiple_different_models(self):
        """Test parsing multiple different models."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o", "claude-sonnet-4-5-20250929"]
        )
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 1, "claude-sonnet-4-5-20250929": 1}

    def test_multiple_models_with_counts(self):
        """Test parsing multiple models with counts."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o:2", "claude-sonnet-4-5-20250929:3"]
        )
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 2, "claude-sonnet-4-5-20250929": 3}

    def test_mixed_models_with_and_without_counts(self):
        """Test parsing mix of models with and without counts."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o", "claude-sonnet-4-5-20250929:2"]
        )
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 1, "claude-sonnet-4-5-20250929": 2}

    def test_model_with_multiple_colons(self):
        """Test parsing ollama-style model with colon in name (e.g. llama:7b:3)."""
        judge_model = _setup_judge_model_arg(["-j", "llama:7b:3"])
        result = parse_judge_models(judge_model)
        assert result == {"llama:7b": 3}

    def test_three_models_mixed(self):
        """Test parsing three models with various count specifications."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o:2", "claude-sonnet-4-5-20250929", "gpt-3.5-turbo:3"]
        )
        result = parse_judge_models(judge_model)
        assert result == {
            "gpt-4o": 2,
            "claude-sonnet-4-5-20250929": 1,
            "gpt-3.5-turbo": 3,
        }

    def test_large_count(self):
        """Test parsing with large instance count."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:100"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 100}

    def test_duplicate_models_last_wins(self):
        """Test that if same model specified twice, last value wins."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:2", "gpt-4o:5"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 5}

    def test_count_zero_raises(self):
        """Count after ':' must be positive; 0 should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:0"])
        with pytest.raises(ValueError, match="must be positive"):
            parse_judge_models(judge_model)

    def test_count_negative_raises(self):
        """Count after ':' must be positive; negative should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:-1"])
        with pytest.raises(ValueError, match="must be positive"):
            parse_judge_models(judge_model)

    def test_count_float_raises(self):
        """Count after ':' must be an integer; float string should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:2.5"])
        with pytest.raises(ValueError, match="must be an integer"):
            parse_judge_models(judge_model)

    def test_count_empty_raises(self):
        """Count after ':' cannot be empty; `model:` should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:"])
        with pytest.raises(ValueError, match="must be an integer"):
            parse_judge_models(judge_model)

    def test_count_non_numeric_raises(self):
        """Count after ':' must be numeric; otherwise should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:abc"])
        with pytest.raises(ValueError, match="must be an integer"):
            parse_judge_models(judge_model)

    def test_count_alphanumeric_raises(self):
        """Count after ':' must be integer only; otherwise should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:2x"])
        with pytest.raises(ValueError, match="must be an integer"):
            parse_judge_models(judge_model)


@pytest.mark.unit
class TestBuildSingleConversationJudgeRunKey:
    """Tests for build_single_conversation_judge_run_key."""

    def test_format_matches_cli_single_mode(self):
        """Stem and truncated microsecond field match prior judge.py inline logic."""
        fixed = datetime(2026, 4, 20, 15, 30, 45, 123456)
        key = build_single_conversation_judge_run_key(
            "/tmp/foo/Amelia_conv.txt",
            now=fixed,
        )
        assert key == "single_20260420_153045_123__Amelia_conv"
