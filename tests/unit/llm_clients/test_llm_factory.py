from unittest.mock import patch

import pytest

from llm_clients import Role
from llm_clients.azure_llm import AzureLLM
from llm_clients.claude_llm import ClaudeLLM
from llm_clients.gemini_llm import GeminiLLM
from llm_clients.llm_factory import LLMFactory
from llm_clients.ollama_llm import OllamaLLM
from llm_clients.openai_llm import OpenAILLM


@pytest.fixture
def mock_all_api_keys():
    """Fixture to patch all API keys for multi-provider tests."""
    with (
        patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key"),
        patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key"),
        patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key"),
        patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key"),
        patch(
            "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
            "https://test.openai.azure.com",
        ),
    ):
        yield


@pytest.mark.unit
class TestLLMFactory:
    """Unit tests for LLMFactory class."""

    @pytest.mark.usefixtures("mock_claude_config", "mock_claude_model")
    def test_create_claude_llm(self):
        """Test that factory correctly creates Claude LLM instance."""
        # Arrange
        model_name = "claude-sonnet-4-5-20250929"
        name = "TestClaude"
        system_prompt = "You are a helpful assistant."

        llm = LLMFactory.create_llm(
            model_name=model_name,
            name=name,
            role=Role.PROVIDER,
            system_prompt=system_prompt,
        )

        assert isinstance(llm, ClaudeLLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == model_name
        assert llm.role == Role.PROVIDER

    @pytest.mark.usefixtures("mock_openai_config", "mock_openai_model")
    def test_create_openai_llm(self):
        """Test that factory correctly creates OpenAI LLM instance."""
        model_name = "gpt-4o"
        name = "TestGPT"
        system_prompt = "You are a test assistant."

        llm = LLMFactory.create_llm(
            model_name=model_name,
            name=name,
            role=Role.PROVIDER,
            system_prompt=system_prompt,
        )

        assert isinstance(llm, OpenAILLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == model_name

    @pytest.mark.usefixtures("mock_gemini_config", "mock_gemini_model")
    def test_create_gemini_llm(self):
        """Test that factory correctly creates Gemini LLM instance."""
        model_name = "gemini-pro"
        name = "TestGemini"
        system_prompt = "You are a Gemini assistant."

        llm = LLMFactory.create_llm(
            model_name=model_name,
            name=name,
            role=Role.PROVIDER,
            system_prompt=system_prompt,
        )

        assert isinstance(llm, GeminiLLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == model_name

    @pytest.mark.usefixtures("mock_ollama_model")
    def test_create_ollama_llm(self):
        """Test that factory correctly creates Ollama LLM instance."""
        model_name = "ollama-llama-3"
        expected_model_name = "llama-3"
        name = "TestOllama"
        system_prompt = "You are an Ollama assistant."

        llm = LLMFactory.create_llm(
            model_name=model_name,
            name=name,
            role=Role.PROVIDER,
            system_prompt=system_prompt,
        )

        assert isinstance(llm, OllamaLLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == expected_model_name

    def test_create_azure_llm(self, mock_azure_config):
        """Test that factory correctly creates Azure LLM instance."""
        model_name = "azure-grok-4"
        name = "TestAzure"
        system_prompt = "You are an Azure assistant."

        llm = LLMFactory.create_llm(
            model_name=model_name,
            name=name,
            role=Role.PROVIDER,
            system_prompt=system_prompt,
        )

        assert isinstance(llm, AzureLLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == "grok-4"  # azure- prefix should be removed

    def test_unsupported_model_raises_error(self):
        """Test that factory raises ValueError for unsupported model names."""
        unsupported_model = "unknown-model-xyz"
        name = "TestUnsupported"

        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create_llm(
                model_name=unsupported_model, name=name, role=Role.PROVIDER
            )

        assert "Unsupported model" in str(exc_info.value)
        assert unsupported_model in str(exc_info.value)

    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    def test_factory_passes_kwargs(self, mock_chat_anthropic):
        """Test that factory correctly forwards kwargs to LLM implementations."""
        # Arrange
        model_name = "claude-sonnet-4-5-20250929"
        name = "TestKwargs"
        temperature = 0.5
        max_tokens = 500

        llm = LLMFactory.create_llm(
            model_name=model_name,
            name=name,
            temperature=temperature,
            max_tokens=max_tokens,
            role=Role.PROVIDER,
        )

        assert isinstance(llm, ClaudeLLM)
        # Verify kwargs were passed to underlying LangChain model
        mock_chat_anthropic.assert_called_once()
        call_kwargs = mock_chat_anthropic.call_args[1]
        assert call_kwargs["temperature"] == temperature
        assert call_kwargs["max_tokens"] == max_tokens

    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    def test_factory_filters_non_model_params(self, mock_chat_openai):
        """Test that factory filters out non-model-specific parameters."""
        model_name = "gpt-4o"
        name = "TestFiltering"
        temperature = 0.7
        # These should be filtered out (model, name, prompt_name, system_prompt)
        extra_params = {
            "prompt_name": "should-be-ignored",
        }

        llm = LLMFactory.create_llm(
            model_name=model_name,
            name=name,
            temperature=temperature,
            role=Role.PROVIDER,
            **extra_params,
        )

        assert isinstance(llm, OpenAILLM)
        assert llm.name == name
        assert llm.model_name == model_name
        # Verify that filtered params were not passed to ChatOpenAI
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["temperature"] == temperature
        assert "prompt_name" not in call_kwargs  # Filtered param should not be present

    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    def test_create_openai_llm_with_openai_prefix(self, mock_chat_openai):
        """Test that factory correctly identifies OpenAI models with 'openai' prefix."""
        model_name = "openai-custom-model"
        name = "TestOpenAIPrefix"

        llm = LLMFactory.create_llm(
            model_name=model_name, name=name, role=Role.PROVIDER
        )

        assert isinstance(llm, OpenAILLM)
        assert llm.model_name == model_name

    @pytest.mark.usefixtures("mock_gemini_config", "mock_gemini_model")
    def test_create_gemini_llm_with_google_prefix(self):
        """Test that factory correctly identifies Gemini models with 'google' prefix."""
        model_name = "google-gemini-ultra"
        name = "TestGooglePrefix"

        llm = LLMFactory.create_llm(
            model_name=model_name, name=name, role=Role.PROVIDER
        )

        assert isinstance(llm, GeminiLLM)
        assert llm.model_name == model_name

    @pytest.mark.usefixtures("mock_ollama_model")
    def test_create_llama_llm_with_ollama_prefix(self):
        """Test that factory correctly identifies Ollama models with 'ollama' prefix."""
        model_name = "ollama-llama-3"
        expected_model_name = "llama-3"
        name = "TestOllamaPrefix"

        llm = LLMFactory.create_llm(
            model_name=model_name, name=name, role=Role.PROVIDER
        )

        assert isinstance(llm, OllamaLLM)
        assert llm.model_name == expected_model_name

    def test_factory_case_insensitive_model_detection(self, mock_all_api_keys):
        """Test that factory detects models regardless of case."""
        with patch(
            "llm_clients.azure_llm.Config.get_azure_config",
            return_value={"model": "azure-gpt-4o"},
        ):
            claude_llm = LLMFactory.create_llm(
                model_name="CLAUDE-3-5", name="Claude", role=Role.PROVIDER
            )
            gpt_llm = LLMFactory.create_llm(
                model_name="gpt-4o-TURBO", name="GPT", role=Role.PROVIDER
            )
            gemini_llm = LLMFactory.create_llm(
                model_name="GEMINI-PRO", name="Gemini", role=Role.PROVIDER
            )
            ollama_llm = LLMFactory.create_llm(
                model_name="OLLAMA-LLAMA-3", name="Ollama", role=Role.PROVIDER
            )
            azure_llm = LLMFactory.create_llm(
                model_name="AZURE-GROK-4", name="Azure", role=Role.PROVIDER
            )

            assert isinstance(claude_llm, ClaudeLLM)
            assert isinstance(gpt_llm, OpenAILLM)
            assert isinstance(gemini_llm, GeminiLLM)
            assert isinstance(ollama_llm, OllamaLLM)
            assert isinstance(azure_llm, AzureLLM)

    @pytest.mark.usefixtures("mock_claude_config", "mock_claude_model")
    def test_create_judge_llm_claude(self):
        """Test that create_judge_llm correctly creates Claude JudgeLLM instance."""
        from llm_clients.llm_interface import JudgeLLM

        model_name = "claude-sonnet-4-5-20250929"
        name = "TestClaudeJudge"
        system_prompt = "You are a helpful judge."

        llm = LLMFactory.create_judge_llm(
            model_name=model_name, name=name, system_prompt=system_prompt
        )

        assert isinstance(llm, JudgeLLM)
        assert isinstance(llm, ClaudeLLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == model_name

    @pytest.mark.usefixtures("mock_openai_config", "mock_openai_model")
    def test_create_judge_llm_openai(self):
        """Test that create_judge_llm correctly creates OpenAI JudgeLLM instance."""
        from llm_clients.llm_interface import JudgeLLM

        model_name = "gpt-4o"
        name = "TestGPTJudge"
        system_prompt = "You are a test judge."

        llm = LLMFactory.create_judge_llm(
            model_name=model_name, name=name, system_prompt=system_prompt
        )

        assert isinstance(llm, JudgeLLM)
        assert isinstance(llm, OpenAILLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == model_name

    @pytest.mark.usefixtures("mock_gemini_config", "mock_gemini_model")
    def test_create_judge_llm_gemini(self):
        """Test that create_judge_llm correctly creates Gemini JudgeLLM instance."""
        from llm_clients.llm_interface import JudgeLLM

        model_name = "gemini-pro"
        name = "TestGeminiJudge"
        system_prompt = "You are a Gemini judge."

        llm = LLMFactory.create_judge_llm(
            model_name=model_name, name=name, system_prompt=system_prompt
        )

        assert isinstance(llm, JudgeLLM)
        assert isinstance(llm, GeminiLLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == model_name

    def test_create_judge_llm_azure(self, mock_azure_config):
        """Test that create_judge_llm correctly creates Azure JudgeLLM instance."""
        from llm_clients.llm_interface import JudgeLLM

        model_name = "azure-grok-4"
        name = "TestAzureJudge"
        system_prompt = "You are an Azure judge."

        llm = LLMFactory.create_judge_llm(
            model_name=model_name, name=name, system_prompt=system_prompt
        )

        assert isinstance(llm, JudgeLLM)
        assert isinstance(llm, AzureLLM)
        assert llm.name == name
        assert llm.system_prompt == system_prompt
        assert llm.model_name == "grok-4"  # azure- prefix should be removed

    def test_create_judge_llm_ollama_raises_error(self):
        """Test that create_judge_llm raises ValueError for Ollama models."""
        model_name = "ollama-llama-3"
        name = "TestOllamaJudge"

        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create_judge_llm(model_name=model_name, name=name)

        assert "does not support structured output" in str(exc_info.value)
        assert model_name in str(exc_info.value)
        assert "Ollama" in str(exc_info.value)

    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    def test_create_judge_llm_passes_kwargs(self, mock_chat_anthropic):
        """Test that create_judge_llm forwards kwargs to LLM implementations."""
        from llm_clients.llm_interface import JudgeLLM

        model_name = "claude-sonnet-4-5-20250929"
        name = "TestKwargsJudge"
        temperature = 0.5
        max_tokens = 500

        llm = LLMFactory.create_judge_llm(
            model_name=model_name,
            name=name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        assert isinstance(llm, JudgeLLM)
        assert isinstance(llm, ClaudeLLM)
        # Verify kwargs were passed to underlying LangChain model
        mock_chat_anthropic.assert_called_once()
        call_kwargs = mock_chat_anthropic.call_args[1]
        assert call_kwargs["temperature"] == temperature
        assert call_kwargs["max_tokens"] == max_tokens

    def test_create_judge_llm_unsupported_model_raises_error(self):
        """Test that create_judge_llm raises ValueError for unsupported model names."""
        unsupported_model = "unknown-model-xyz"
        name = "TestUnsupportedJudge"

        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create_judge_llm(model_name=unsupported_model, name=name)

        # Should raise error from create_llm first
        assert "Unsupported model" in str(exc_info.value)
        assert unsupported_model in str(exc_info.value)
