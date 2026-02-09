"""Comprehensive tests for OllamaLLM implementation."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_clients.llm_interface import Role

from .test_base_llm import TestLLMBase
from .test_helpers import (
    assert_error_metadata,
    assert_error_response,
    assert_iso_timestamp,
    assert_metadata_copy_behavior,
    assert_metadata_structure,
    assert_response_timing,
)


@pytest.mark.unit
@pytest.mark.usefixtures("mock_ollama_model")
class TestOllamaLLM(TestLLMBase):
    """Unit tests for OllamaLLM class.

    OllamaLLM only implements LLMInterface (not JudgeLLM) since it doesn't
    support structured output generation.
    """

    # ============================================================================
    # Factory Methods (Required by TestLLMBase)
    # ============================================================================

    def create_llm(self, role: Role, **kwargs):
        """Create OllamaLLM instance for testing."""
        from llm_clients.ollama_llm import OllamaLLM

        # Provide default name if not specified
        if "name" not in kwargs:
            kwargs["name"] = "test-ollama"

        with patch("llm_clients.ollama_llm.LangChainOllamaLLM") as mock_ollama:
            mock_instance = MagicMock()
            mock_ollama.return_value = mock_instance
            return OllamaLLM(role=role, **kwargs)

    def get_provider_name(self) -> str:
        """Get provider name for metadata validation."""
        return "ollama"

    @contextmanager
    def get_mock_patches(self):
        """Set up mocks for Ollama.

        Note: Actual mocking is handled by class-level fixtures.
        This method provides a no-op context manager for base class compatibility.
        """
        yield

    # ============================================================================
    # Ollama-Specific Tests
    # ============================================================================
    # Note: Ollama uses string-based conversation format instead of LangChain
    # messages, so it has unique behavior that needs specific tests.
    # Some base class tests don't apply due to this difference.
    # ============================================================================

    # Override base class tests that don't work with Ollama's string format
    @pytest.mark.asyncio
    async def test_generate_response_returns_llm_text(
        self, mock_response_factory, mock_llm_factory, mock_system_message
    ):
        """Return LLM output; Ollama's ainvoke returns a string directly."""
        from llm_clients.ollama_llm import OllamaLLM

        expected_text = "Ollama response string"
        with patch("llm_clients.ollama_llm.LangChainOllamaLLM") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(return_value=expected_text)
            mock_ollama.return_value = mock_instance

            llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
            response = await llm.generate_response(
                conversation_history=mock_system_message
            )

            assert response == expected_text

    @pytest.mark.asyncio
    async def test_generate_response_updates_metadata(
        self, mock_response_factory, mock_llm_factory, mock_system_message
    ):
        """Test that generate_response updates metadata - Ollama override."""
        from llm_clients.ollama_llm import OllamaLLM

        with patch("llm_clients.ollama_llm.LangChainOllamaLLM") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(return_value="Response")
            mock_ollama.return_value = mock_instance

            llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
            await llm.generate_response(conversation_history=mock_system_message)

            # Verify metadata structure (Ollama-specific)
            metadata = assert_metadata_structure(
                llm,
                expected_provider=self.get_provider_name(),
                expected_role=Role.PROVIDER,
            )

            assert "timestamp" in metadata
            assert_iso_timestamp(metadata["timestamp"])
            assert_response_timing(metadata)

    @pytest.mark.asyncio
    async def test_generate_response_handles_errors(
        self, mock_llm_factory, mock_system_message
    ):
        """Test that generate_response handles errors - Ollama override."""
        from llm_clients.ollama_llm import OllamaLLM

        with patch("llm_clients.ollama_llm.LangChainOllamaLLM") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(side_effect=Exception("Ollama Error"))
            mock_ollama.return_value = mock_instance

            llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
            response = await llm.generate_response(
                conversation_history=mock_system_message
            )

            assert_error_response(response, "Ollama Error")
            assert_error_metadata(
                llm,
                expected_provider=self.get_provider_name(),
                expected_error_substring="Ollama Error",
            )

    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    def test_init_with_default_config(self, mock_ollama):
        from llm_clients.ollama_llm import OllamaLLM

        OllamaLLM(name="test-ollama", role=Role.PROVIDER)

        # Verify Ollama was initialized with default config
        mock_ollama.assert_called_once()
        call_kwargs = mock_ollama.call_args[1]

        assert "model" in call_kwargs
        # base_url has a hardcoded default for Ollama connectivity
        assert "base_url" in call_kwargs
        assert call_kwargs["base_url"] == "http://localhost:11434"
        # temperature should NOT be set by default
        assert "temperature" not in call_kwargs

    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    def test_init_with_custom_model_name(self, mock_ollama):
        """Test initialization with custom model name."""
        from llm_clients.ollama_llm import OllamaLLM

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER, model_name="llama3:70b")

        call_kwargs = mock_ollama.call_args[1]
        assert call_kwargs["model"] == "llama3:70b"
        assert llm.model_name == "llama3:70b"

    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    def test_init_with_custom_temperature(self, mock_ollama):
        """Test initialization with custom temperature via kwargs."""
        from llm_clients.ollama_llm import OllamaLLM

        OllamaLLM(name="test-ollama", role=Role.PROVIDER, temperature=0.9)

        call_kwargs = mock_ollama.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    def test_init_with_custom_base_url(self, mock_ollama):
        """Test initialization with custom Ollama base URL."""
        from llm_clients.ollama_llm import OllamaLLM

        custom_url = "http://remote-server:11434"
        OllamaLLM(name="test-ollama", role=Role.PROVIDER, base_url=custom_url)

        call_kwargs = mock_ollama.call_args[1]
        assert call_kwargs["base_url"] == custom_url

    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    def test_init_kwargs_override_defaults(self, mock_ollama):
        """Test that kwargs override default config values."""
        from llm_clients.ollama_llm import OllamaLLM

        OllamaLLM(
            name="test-ollama",
            role=Role.PROVIDER,
            temperature=0.1,
            top_p=0.95,
            num_predict=500,
        )

        call_kwargs = mock_ollama.call_args[1]
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["num_predict"] == 500

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_without_system_prompt(
        self, mock_ollama, mock_system_message
    ):
        """Test response without system prompt uses Human/Assistant format."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="This is a test response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        response = await llm.generate_response(conversation_history=mock_system_message)

        # Verify message uses Human/Assistant format even without system prompt
        mock_instance.ainvoke.assert_called_once_with("Human: Test\n\nAssistant:")
        assert response == "This is a test response"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_system_prompt_in_init(
        self, mock_ollama, mock_system_message
    ):
        """Test generating response with system prompt set during initialization."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="I'm doing well, thanks!")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(
            name="test-ollama",
            role=Role.PROVIDER,
            system_prompt="You are a helpful assistant",
        )
        response = await llm.generate_response(conversation_history=mock_system_message)

        # Verify system prompt was included in formatted message
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert "System: You are a helpful assistant" in call_args
        assert "Human: Test" in call_args
        assert "Assistant:" in call_args
        assert response == "I'm doing well, thanks!"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_system_prompt_set_later(
        self, mock_ollama, mock_system_message
    ):
        """Test generating response with system prompt set after initialization."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Sure, I can help with that")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        llm.set_system_prompt("You are a coding expert")
        response = await llm.generate_response(conversation_history=mock_system_message)

        # Verify system prompt was included
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert "System: You are a coding expert" in call_args
        assert "Human: Test" in call_args
        assert response == "Sure, I can help with that"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_handles_ollama_connection_error(
        self, mock_ollama, mock_system_message
    ):
        """Test error handling when Ollama server is unreachable."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(
            side_effect=ConnectionError("Could not connect to Ollama server")
        )
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        response = await llm.generate_response(conversation_history=mock_system_message)

        # Should return error message, not raise exception
        assert_error_response(response, "Could not connect to Ollama server")

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_handles_model_not_found(
        self, mock_ollama, mock_system_message
    ):
        """Test error handling when model doesn't exist."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(
            side_effect=ValueError("Model 'nonexistent:latest' not found")
        )
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(
            name="test-ollama", role=Role.PROVIDER, model_name="nonexistent:latest"
        )
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert_error_response(response, "Model 'nonexistent:latest' not found")

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_handles_timeout_error(
        self, mock_ollama, mock_system_message
    ):
        """Test error handling when request times out."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(
            side_effect=TimeoutError("Request timed out after 30s")
        )
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert_error_response(response, "Request timed out")

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_handles_generic_exception(
        self, mock_ollama, mock_system_message
    ):
        """Test error handling for unexpected exceptions."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(
            side_effect=RuntimeError("Unexpected error occurred")
        )
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert_error_response(response, "Unexpected error occurred")

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_none_message(
        self, mock_ollama, mock_system_message
    ):
        """Test generating response with None message."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Default response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        response = await llm.generate_response(None)

        # Should handle None gracefully - message won't include current message part
        mock_instance.ainvoke.assert_called_once_with("")
        assert response == "Default response"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_empty_string(self, mock_ollama):
        """Test generating response with empty string message."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Response to empty")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": ""}]
        )

        # Empty string gets formatted as "Human: \n\nAssistant:"
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert "Human:" in call_args
        assert "Assistant:" in call_args
        assert response == "Response to empty"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_preserves_multiline_messages(self, mock_ollama):
        """Test that multiline messages are preserved correctly."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Response")
        mock_ollama.return_value = mock_instance

        multiline_msg = "Line 1\nLine 2\nLine 3"
        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER, system_prompt="Helper")
        await llm.generate_response(
            conversation_history=[{"turn": 0, "response": multiline_msg}]
        )

        call_args = mock_instance.ainvoke.call_args[0][0]
        assert "Line 1\nLine 2\nLine 3" in call_args

    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    def test_set_system_prompt_updates_prompt(self, mock_ollama):
        """Test that set_system_prompt updates the system_prompt attribute."""
        from llm_clients.ollama_llm import OllamaLLM

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)

        # Initially empty string (from LLMInterface base class)
        assert llm.system_prompt == ""

        # Set prompt
        llm.set_system_prompt("You are a math tutor")
        assert llm.system_prompt == "You are a math tutor"

        # Update prompt
        llm.set_system_prompt("You are a science tutor")
        assert llm.system_prompt == "You are a science tutor"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_set_system_prompt_affects_subsequent_calls(self, mock_ollama):
        """Test that changing system prompt affects future generate_response calls."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)

        # First call without system prompt
        await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Question 1"}]
        )
        call1 = mock_instance.ainvoke.call_args[0][0]
        assert "System:" not in call1

        # Set system prompt
        llm.set_system_prompt("You are helpful")

        # Second call with system prompt
        mock_instance.ainvoke.reset_mock()
        await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Question 2"}]
        )
        call2 = mock_instance.ainvoke.call_args[0][0]
        assert "System: You are helpful" in call2

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_conversation_history(self, mock_ollama):
        """Test generate_response with conversation_history parameter."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Response with history")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(
            name="test-ollama", role=Role.PROVIDER, system_prompt="You are helpful"
        )

        history = [
            {
                "turn": 1,
                "speaker": "persona",
                "input": "Start",
                "response": "Hello",
            },
            {
                "turn": 2,
                "speaker": "provider",
                "input": "Hello",
                "response": "Hi there",
            },
            {
                "turn": 3,
                "speaker": "persona",
                "input": "Hi there",
                "response": "How are you?",
            },
        ]

        response = await llm.generate_response(conversation_history=history)

        assert response == "Response with history"

        # Verify ainvoke was called with formatted history
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert "System: You are helpful" in call_args
        assert "Human: Hello" in call_args
        assert "Assistant: Hi there" in call_args
        assert "Human: How are you?" in call_args
        assert "Assistant:" in call_args

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_empty_conversation_history(
        self, mock_ollama, mock_system_message
    ):
        """Test generate_response with empty conversation_history list."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)

        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"

        # Should just have current message
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert call_args == "Human: Test\n\nAssistant:"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_none_conversation_history(
        self, mock_ollama, mock_system_message
    ):
        """Test generate_response with None conversation_history."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)

        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"

        # Should just have current message
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert call_args == "Human: Test\n\nAssistant:"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_persona_role_flips_types(self, mock_ollama):
        """Test that persona role flips message types in conversation history."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Persona response")
        mock_ollama.return_value = mock_instance

        # Persona role triggers message type flipping:
        # (persona -> Assistant, provider -> Human)
        persona_prompt = "You are roleplaying as a human user"
        llm = OllamaLLM(
            name="test-ollama", role=Role.PERSONA, system_prompt=persona_prompt
        )

        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "provider", "response": "Hi there"},
            {"turn": 3, "speaker": "persona", "response": "How are you?"},
        ]

        response = await llm.generate_response(conversation_history=history)

        assert response == "Persona response"

        # Verify message types are flipped for persona role
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert "System: You are roleplaying as a human user" in call_args
        # Turn 1 (persona, odd) should be Assistant when persona role
        assert "Assistant: Hello" in call_args
        # Turn 2 (provider, even) should be Human when persona role
        assert "Human: Hi there" in call_args
        # Turn 3 (persona, odd) should be Assistant when persona role
        assert "Assistant: How are you?" in call_args
        assert "Assistant:" in call_args

    def test_get_last_response_metadata_returns_copy(self):
        """Test that get_last_response_metadata returns a copy."""
        from llm_clients.ollama_llm import OllamaLLM

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        assert_metadata_copy_behavior(llm)

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_metadata_populated_after_successful_response(
        self, mock_ollama, mock_system_message
    ):
        """Test that metadata is populated correctly after successful response."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="This is a test response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER, model_name="llama3:8b")
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "This is a test response"

        # Verify metadata was extracted
        metadata = assert_metadata_structure(
            llm, expected_provider="ollama", expected_role=Role.PROVIDER
        )
        assert metadata["response_id"] is None
        assert metadata["model"] == "llama3:8b"
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_metadata_populated_after_error(
        self, mock_ollama, mock_system_message
    ):
        """Test that metadata is populated correctly after error."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(
            side_effect=ConnectionError("Could not connect to Ollama server")
        )
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER, model_name="llama3:8b")
        response = await llm.generate_response(conversation_history=mock_system_message)

        # Should return error message instead of raising
        assert_error_response(response, "Could not connect to Ollama server")

        # Verify error metadata was stored
        assert_error_metadata(llm, "ollama", "Could not connect to Ollama server")

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_metadata_tracks_timing(self, mock_ollama, mock_system_message):
        """Test that response timing is tracked correctly."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Timed response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert_response_timing(metadata)

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_timestamp_format(self, mock_ollama, mock_system_message):
        """Test that timestamp is in ISO format."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Test")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert_iso_timestamp(metadata["timestamp"])

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_metadata_structure_complete(self, mock_ollama, mock_system_message):
        """Test that metadata structure includes all expected fields."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Complete response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER, model_name="mistral:7b")
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()

        # Verify all expected fields are present using helper
        assert_metadata_structure(
            llm, expected_provider="ollama", expected_role=Role.PROVIDER
        )

        # Verify field types
        assert metadata["response_id"] is None
        assert isinstance(metadata["model"], str)
        assert_response_timing(metadata)
        assert isinstance(metadata["usage"], dict)

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_metadata_initialized_empty(self, mock_ollama):
        """Test that metadata is initialized as empty dict."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)

        # Before any response, metadata should be empty
        metadata = llm.get_last_response_metadata()
        assert metadata == {}

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_usage_metadata_always_empty(self, mock_ollama, mock_system_message):
        """Test that Ollama usage metadata is always empty.

        Ollama's BaseLLM.ainvoke returns a plain string without metadata,
        so usage is always an empty dict.
        """
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Test response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert metadata["usage"] == {}
        # Ollama doesn't have these fields
        assert "prompt_tokens" not in metadata["usage"]
        assert "completion_tokens" not in metadata["usage"]
        assert "total_tokens" not in metadata["usage"]

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_no_response_object_in_metadata(
        self, mock_ollama, mock_system_message
    ):
        """Test that Ollama metadata doesn't include response object.

        Ollama returns a plain string, not a response object.
        """
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Test response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        # Ollama doesn't store the response object
        assert "response" not in metadata

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_no_finish_reason_in_metadata(self, mock_ollama, mock_system_message):
        """Test that Ollama metadata doesn't include finish_reason.

        Ollama's simple string response doesn't include finish reason.
        """
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Test response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        # Ollama doesn't have finish_reason
        assert "finish_reason" not in metadata
        assert "stop_reason" not in metadata

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_no_raw_metadata_stored(self, mock_ollama, mock_system_message):
        """Test that Ollama doesn't store raw metadata.

        Ollama's ainvoke returns a plain string without rich metadata.
        """
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Test response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        # Ollama doesn't store raw_metadata or raw_response_metadata
        assert "raw_metadata" not in metadata
        assert "raw_response_metadata" not in metadata
