from unittest.mock import patch

import pytest

from llm_clients.config import Config


@pytest.mark.unit
class TestConfig:
    """Unit tests for Config class."""

    def test_api_keys_loaded_from_env(self):
        """Test that API keys are loaded from environment variables."""
        # These should be None or set depending on the test environment
        # We just verify the attributes exist
        assert hasattr(Config, "ANTHROPIC_API_KEY")
        assert hasattr(Config, "OPENAI_API_KEY")
        assert hasattr(Config, "GOOGLE_API_KEY")

    def test_models_config_structure(self):
        """Test that MODELS_CONFIG has expected structure."""
        assert isinstance(Config.MODELS_CONFIG, dict)
        assert len(Config.MODELS_CONFIG) > 0

        # Check a few known models
        assert "claude-3-5-sonnet-20241022" in Config.MODELS_CONFIG
        assert "gpt-4" in Config.MODELS_CONFIG
        assert "gemini-1.5-pro" in Config.MODELS_CONFIG
        assert "llama3:8b" in Config.MODELS_CONFIG

    def test_claude_config_has_required_fields(self):
        """Test that Claude config entries have required fields."""
        claude_config = Config.MODELS_CONFIG["claude-3-5-sonnet-20241022"]
        assert "provider" in claude_config
        assert claude_config["provider"] == "anthropic"
        assert "temperature" in claude_config
        assert "max_tokens" in claude_config

    def test_openai_config_has_required_fields(self):
        """Test that OpenAI config entries have required fields."""
        openai_config = Config.MODELS_CONFIG["gpt-4"]
        assert "provider" in openai_config
        assert openai_config["provider"] == "openai"
        assert "temperature" in openai_config
        assert "max_tokens" in openai_config

    def test_gemini_config_has_required_fields(self):
        """Test that Gemini config entries have required fields."""
        gemini_config = Config.MODELS_CONFIG["gemini-1.5-pro"]
        assert "provider" in gemini_config
        assert gemini_config["provider"] == "google"
        assert "temperature" in gemini_config
        assert "max_tokens" in gemini_config

    def test_llama_config_has_required_fields(self):
        """Test that Llama config entries have required fields."""
        llama_config = Config.MODELS_CONFIG["llama3:8b"]
        assert "provider" in llama_config
        assert llama_config["provider"] == "ollama"
        assert "temperature" in llama_config
        assert "base_url" in llama_config

    def test_get_claude_config(self):
        """Test get_claude_config returns expected structure (line 77)."""
        config = Config.get_claude_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert config["model"] == "claude-3-5-sonnet-20241022"
        assert "temperature" in config
        assert config["temperature"] == 0.7
        assert "max_tokens" in config
        assert config["max_tokens"] == 1000

    def test_get_openai_config(self):
        """Test get_openai_config returns expected structure (line 86)."""
        config = Config.get_openai_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert config["model"] == "gpt-4"
        assert "temperature" in config
        assert config["temperature"] == 0.7
        assert "max_tokens" in config
        assert config["max_tokens"] == 1000

    def test_get_gemini_config(self):
        """Test get_gemini_config returns expected structure."""
        config = Config.get_gemini_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert config["model"] == "gemini-1.5-pro"
        assert "temperature" in config
        assert config["temperature"] == 0.7
        assert "max_tokens" in config
        assert config["max_tokens"] == 1000

    def test_get_llama_config(self):
        """Test get_llama_config returns expected structure."""
        config = Config.get_llama_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert config["model"] == "llama3:8b"
        assert "temperature" in config
        assert config["temperature"] == 0.7
        assert "base_url" in config
        assert config["base_url"] == "http://localhost:11434"

    def test_config_defaults_are_consistent(self):
        """Test that default temperature and max_tokens are consistent."""
        for model_name, model_config in Config.MODELS_CONFIG.items():
            assert "provider" in model_config
            assert "temperature" in model_config
            # Ollama models don't require max_tokens
            if model_config["provider"] != "ollama":
                assert "max_tokens" in model_config

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    def test_anthropic_api_key_from_env(self):
        """Test that ANTHROPIC_API_KEY can be loaded from environment."""
        # Note: This tests the module-level loading behavior
        # The actual key is loaded at import time, so we're just verifying
        # the attribute exists and can be set
        # Reload the module to pick up the patched environment
        import importlib

        from llm_clients import config

        importlib.reload(config)

        assert config.Config.ANTHROPIC_API_KEY == "test-anthropic-key"

        # Reload again to restore original state
        importlib.reload(config)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai-key"})
    def test_openai_api_key_from_env(self):
        """Test that OPENAI_API_KEY can be loaded from environment."""
        import importlib

        from llm_clients import config

        importlib.reload(config)
        assert config.Config.OPENAI_API_KEY == "test-openai-key"

        importlib.reload(config)

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-google-key"})
    def test_google_api_key_from_env(self):
        """Test that GOOGLE_API_KEY can be loaded from environment."""
        import importlib

        from llm_clients import config

        importlib.reload(config)
        assert config.Config.GOOGLE_API_KEY == "test-google-key"

        importlib.reload(config)
