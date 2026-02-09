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

    def test_get_claude_config(self):
        """Test get_claude_config returns expected structure."""
        config = Config.get_claude_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert config["model"] == "claude-sonnet-4-5-20250929"
        # Temperature and max_tokens should NOT be in config
        assert "temperature" not in config
        assert "max_tokens" not in config

    def test_get_openai_config(self):
        """Test get_openai_config returns expected structure."""
        config = Config.get_openai_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert config["model"] == "gpt-5.2"
        # Temperature and max_tokens should NOT be in config
        assert "temperature" not in config
        assert "max_tokens" not in config

    def test_get_gemini_config(self):
        """Test get_gemini_config returns expected structure."""
        config = Config.get_gemini_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert config["model"] == "gemini-1.5-pro"
        # Temperature and max_tokens should NOT be in config
        assert "temperature" not in config
        assert "max_tokens" not in config

    def test_get_ollama_config(self):
        """Test get_ollama_config returns expected structure."""
        config = Config.get_ollama_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert config["model"] == "llama3:8b"
        # Temperature should NOT be in config
        assert "temperature" not in config
        # base_url should be in config
        assert "base_url" in config
        assert config["base_url"] == "http://localhost:11434"

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
