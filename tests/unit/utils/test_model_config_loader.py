"""Comprehensive tests for model configuration loading."""

import json

import pytest

from utils.model_config_loader import get_model_for_prompt, load_model_config


@pytest.mark.unit
class TestLoadModelConfig:
    """Test load_model_config function."""

    def test_load_model_config_with_valid_file(self, tmp_path):
        """Test loading a valid model configuration file."""
        config_data = {
            "prompt_models": {
                "persona_anxious": "gpt-4o",
                "persona_depressed": "claude-3-opus",
                "chatbot_therapist": "claude-3-5-sonnet",
            },
            "default_model": "claude-sonnet-4-5-20250929",
            "temperature": 0.7,
        }

        config_file = tmp_path / "model_config.json"
        config_file.write_text(json.dumps(config_data))

        result = load_model_config(str(config_file))

        assert result == config_data
        assert result["default_model"] == "claude-sonnet-4-5-20250929"
        assert result["prompt_models"]["persona_anxious"] == "gpt-4o"
        assert result["temperature"] == 0.7

    def test_load_model_config_with_minimal_structure(self, tmp_path):
        """Test loading config with only required fields."""
        config_data = {"prompt_models": {}, "default_model": "gpt-4o"}

        config_file = tmp_path / "minimal_config.json"
        config_file.write_text(json.dumps(config_data))

        result = load_model_config(str(config_file))

        assert result["prompt_models"] == {}
        assert result["default_model"] == "gpt-4o"

    def test_load_model_config_file_not_found(self, tmp_path, capsys):
        """Test handling of non-existent config file."""
        nonexistent_file = tmp_path / "does_not_exist.json"

        result = load_model_config(str(nonexistent_file))

        # Should return default config
        assert result["prompt_models"] == {}
        assert result["default_model"] == "claude-sonnet-4-5-20250929"

        # Should print warning
        captured = capsys.readouterr()
        assert "Warning: Model config file" in captured.out
        assert "not found" in captured.out

    def test_load_model_config_invalid_json_syntax(self, tmp_path, capsys):
        """Test handling of malformed JSON file."""
        config_file = tmp_path / "bad_syntax.json"
        config_file.write_text("{invalid json: missing quotes and comma,}")

        result = load_model_config(str(config_file))

        # Should return default config
        assert result["prompt_models"] == {}
        assert result["default_model"] == "claude-sonnet-4-5-20250929"

        # Should print error
        captured = capsys.readouterr()
        assert "Error loading model config" in captured.out

    def test_load_model_config_empty_file(self, tmp_path, capsys):
        """Test handling of empty JSON file."""
        config_file = tmp_path / "empty.json"
        config_file.write_text("")

        result = load_model_config(str(config_file))

        # Should return default config
        assert result["prompt_models"] == {}
        assert result["default_model"] == "claude-sonnet-4-5-20250929"

        captured = capsys.readouterr()
        assert "Error loading model config" in captured.out

    def test_load_model_config_not_json_object(self, tmp_path):
        """Test handling of JSON file containing non-object (e.g., array)."""
        config_file = tmp_path / "array.json"
        config_file.write_text('["not", "an", "object"]')

        result = load_model_config(str(config_file))

        # Should return the parsed content (even if unexpected type)
        assert result == ["not", "an", "object"]

    def test_load_model_config_with_unicode_characters(self, tmp_path):
        """Test loading config with unicode characters in model names."""
        config_data = {
            "prompt_models": {
                "persona_日本語": "gpt-4o",
                "persona_émotionnel": "claude-3-opus",
            },
            "default_model": "claude-sonnet-4-5-20250929",
        }

        config_file = tmp_path / "unicode_config.json"
        config_file.write_text(
            json.dumps(config_data, ensure_ascii=False), encoding="utf-8"
        )

        result = load_model_config(str(config_file))

        assert "persona_日本語" in result["prompt_models"]
        assert "persona_émotionnel" in result["prompt_models"]

    def test_load_model_config_with_nested_structure(self, tmp_path):
        """Test loading config with nested data structures."""
        config_data = {
            "prompt_models": {"persona_1": "gpt-4o"},
            "default_model": "claude-sonnet-4-5-20250929",
            "model_params": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "nested": {"key": "value"},
            },
        }

        config_file = tmp_path / "nested_config.json"
        config_file.write_text(json.dumps(config_data))

        result = load_model_config(str(config_file))

        assert result["model_params"]["temperature"] == 0.7
        assert result["model_params"]["nested"]["key"] == "value"

    def test_load_model_config_permission_error(self, tmp_path, capsys):
        """Test handling of permission denied error."""
        config_file = tmp_path / "no_permission.json"
        config_file.write_text('{"test": "data"}')

        # Make file unreadable (Unix-like systems)
        import os

        if os.name != "nt":  # Skip on Windows
            config_file.chmod(0o000)

            result = load_model_config(str(config_file))

            # Should return default config
            assert result["prompt_models"] == {}
            assert result["default_model"] == "claude-sonnet-4-5-20250929"

            # Restore permissions for cleanup
            config_file.chmod(0o644)


@pytest.mark.unit
class TestGetModelForPrompt:
    """Test get_model_for_prompt function."""

    def test_get_model_for_prompt_returns_specific_model(self, tmp_path):
        """Test getting model for a prompt that exists in config."""
        config_data = {
            "prompt_models": {
                "persona_anxious": "gpt-4o-turbo",
                "persona_happy": "claude-3-opus",
            },
            "default_model": "claude-sonnet-4-5-20250929",
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        model = get_model_for_prompt("persona_anxious", str(config_file))

        assert model == "gpt-4o-turbo"

    def test_get_model_for_prompt_returns_default_for_unknown(self, tmp_path):
        """Test getting model for prompt not in config returns default."""
        config_data = {
            "prompt_models": {"persona_known": "gpt-4o"},
            "default_model": "claude-sonnet-4-5-20250929",
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        model = get_model_for_prompt("persona_unknown", str(config_file))

        assert model == "claude-sonnet-4-5-20250929"

    def test_get_model_for_prompt_with_empty_prompt_models(self, tmp_path):
        """Test getting model when prompt_models is empty."""
        config_data = {"prompt_models": {}, "default_model": "gpt-4o"}

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        model = get_model_for_prompt("any_prompt", str(config_file))

        assert model == "gpt-4o"

    def test_get_model_for_prompt_with_missing_config_file(self):
        """Test getting model when config file doesn't exist."""
        model = get_model_for_prompt("test_prompt", "nonexistent_file.json")

        # Should return default model from load_model_config fallback
        assert model == "claude-sonnet-4-5-20250929"

    def test_get_model_for_prompt_case_sensitivity(self, tmp_path):
        """Test that prompt name matching is case-sensitive."""
        config_data = {
            "prompt_models": {
                "PersonaAnxious": "gpt-4o",
                "persona_anxious": "claude-3-opus",
            },
            "default_model": "claude-sonnet-4-5-20250929",
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        # Should match exact case
        model1 = get_model_for_prompt("PersonaAnxious", str(config_file))
        model2 = get_model_for_prompt("persona_anxious", str(config_file))
        model3 = get_model_for_prompt("personaanxious", str(config_file))

        assert model1 == "gpt-4o"
        assert model2 == "claude-3-opus"
        assert model3 == "claude-sonnet-4-5-20250929"  # Falls back to default

    def test_get_model_for_prompt_with_special_characters(self, tmp_path):
        """Test prompt names with special characters."""
        config_data = {
            "prompt_models": {
                "persona-with-dashes": "gpt-4o",
                "persona_with_underscores": "claude-3-opus",
                "persona.with.dots": "gpt-3.5-turbo",
            },
            "default_model": "claude-sonnet-4-5-20250929",
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        assert get_model_for_prompt("persona-with-dashes", str(config_file)) == "gpt-4o"
        assert (
            get_model_for_prompt("persona_with_underscores", str(config_file))
            == "claude-3-opus"
        )
        assert (
            get_model_for_prompt("persona.with.dots", str(config_file))
            == "gpt-3.5-turbo"
        )

    def test_get_model_for_prompt_multiple_calls_consistent(self, tmp_path):
        """Test that multiple calls with same prompt return consistent results."""
        config_data = {
            "prompt_models": {"test_prompt": "gpt-4o"},
            "default_model": "claude-sonnet-4-5-20250929",
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        model1 = get_model_for_prompt("test_prompt", str(config_file))
        model2 = get_model_for_prompt("test_prompt", str(config_file))
        model3 = get_model_for_prompt("test_prompt", str(config_file))

        assert model1 == model2 == model3 == "gpt-4o"
