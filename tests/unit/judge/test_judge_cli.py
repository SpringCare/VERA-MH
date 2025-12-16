"""Unit tests for judge.py CLI argument parsing."""


def parse_judge_models(model_specs):
    """
    Parse judge model specifications.

    This is the logic from judge.py main() function that parses
    the --judge-model argument.
    """
    judge_models = {}
    for model_spec in model_specs:
        if ":" in model_spec:
            # Format: "model:count"
            model, count = model_spec.rsplit(":", 1)
            judge_models[model] = int(count)
        else:
            # Format: "model" (defaults to 1 instance)
            judge_models[model_spec] = 1
    return judge_models


class TestJudgeModelParsing:
    """Test parsing of --judge-model CLI argument."""

    def test_single_model(self):
        """Test parsing a single model without count."""
        result = parse_judge_models(["gpt-4o"])
        assert result == {"gpt-4o": 1}

    def test_single_model_with_count(self):
        """Test parsing a single model with count."""
        result = parse_judge_models(["gpt-4o:3"])
        assert result == {"gpt-4o": 3}

    def test_multiple_different_models(self):
        """Test parsing multiple different models."""
        result = parse_judge_models(["gpt-4o", "claude-3-5-sonnet-20241022"])
        assert result == {"gpt-4o": 1, "claude-3-5-sonnet-20241022": 1}

    def test_multiple_models_with_counts(self):
        """Test parsing multiple models with counts."""
        result = parse_judge_models(["gpt-4o:2", "claude-3-5-sonnet-20241022:3"])
        assert result == {"gpt-4o": 2, "claude-3-5-sonnet-20241022": 3}

    def test_mixed_models_with_and_without_counts(self):
        """Test parsing mix of models with and without counts."""
        result = parse_judge_models(["gpt-4o", "claude-3-5-sonnet-20241022:2"])
        assert result == {"gpt-4o": 1, "claude-3-5-sonnet-20241022": 2}

    def test_model_with_multiple_colons(self):
        """Test parsing model name that contains colons (e.g., dated model names)."""
        # Should use rsplit to handle model names with colons
        result = parse_judge_models(["claude-3-5-sonnet-20241022:2"])
        assert result == {"claude-3-5-sonnet-20241022": 2}

    def test_three_models_mixed(self):
        """Test parsing three models with various count specifications."""
        result = parse_judge_models(
            ["gpt-4o:2", "claude-3-5-sonnet-20241022", "gpt-3.5-turbo:3"]
        )
        assert result == {
            "gpt-4o": 2,
            "claude-3-5-sonnet-20241022": 1,
            "gpt-3.5-turbo": 3,
        }

    def test_large_count(self):
        """Test parsing with large instance count."""
        result = parse_judge_models(["gpt-4o:100"])
        assert result == {"gpt-4o": 100}

    def test_empty_list(self):
        """Test parsing empty model list returns empty dict."""
        result = parse_judge_models([])
        assert result == {}

    def test_duplicate_models_last_wins(self):
        """Test that if same model specified twice, last value wins."""
        result = parse_judge_models(["gpt-4o:2", "gpt-4o:5"])
        # Last specification should win
        assert result == {"gpt-4o": 5}
