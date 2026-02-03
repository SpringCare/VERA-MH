"""Comprehensive tests for OllamaLLM implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_clients.llm_interface import Role


@pytest.mark.unit
class TestOllamaLLMInit:
    """Test OllamaLLM initialization."""

    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    def test_init_with_default_config(self, mock_ollama):
        """Test initialization uses default config when no overrides provided."""
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


@pytest.mark.unit
class TestOllamaLLMGenerateResponse:
    """Test OllamaLLM response generation."""

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_without_system_prompt(self, mock_ollama):
        """Test response without system prompt uses Human/Assistant format."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="This is a test response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Hello, how are you?"}
            ]
        )

        # Verify message uses Human/Assistant format even without system prompt
        mock_instance.ainvoke.assert_called_once_with(
            "Human: Hello, how are you?\n\nAssistant:"
        )
        assert response == "This is a test response"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_system_prompt_in_init(self, mock_ollama):
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
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "How are you?"}
            ]
        )

        # Verify system prompt was included in formatted message
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert "System: You are a helpful assistant" in call_args
        assert "Human: How are you?" in call_args
        assert "Assistant:" in call_args
        assert response == "I'm doing well, thanks!"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_system_prompt_set_later(self, mock_ollama):
        """Test generating response with system prompt set after initialization."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Sure, I can help with that")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        llm.set_system_prompt("You are a coding expert")
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Help me debug this code"}
            ]
        )

        # Verify system prompt was included
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert "System: You are a coding expert" in call_args
        assert "Human: Help me debug this code" in call_args
        assert response == "Sure, I can help with that"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_handles_ollama_connection_error(self, mock_ollama):
        """Test error handling when Ollama server is unreachable."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(
            side_effect=ConnectionError("Could not connect to Ollama server")
        )
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Test message"}
            ]
        )

        # Should return error message, not raise exception
        assert "Error generating response" in response
        assert "Could not connect to Ollama server" in response

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_handles_model_not_found(self, mock_ollama):
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
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Test message"}
            ]
        )

        assert "Error generating response" in response
        assert "Model 'nonexistent:latest' not found" in response

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_handles_timeout_error(self, mock_ollama):
        """Test error handling when request times out."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(
            side_effect=TimeoutError("Request timed out after 30s")
        )
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        response = await llm.generate_response(
            conversation_history=[
                {
                    "turn": 0,
                    "speaker": "system",
                    "response": "Long message that times out",
                }
            ]
        )

        assert "Error generating response" in response
        assert "Request timed out" in response

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_handles_generic_exception(self, mock_ollama):
        """Test error handling for unexpected exceptions."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(
            side_effect=RuntimeError("Unexpected error occurred")
        )
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        assert "Error generating response" in response
        assert "Unexpected error occurred" in response

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_none_message(self, mock_ollama):
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
            conversation_history=[{"turn": 0, "speaker": "system", "response": ""}]
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
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": multiline_msg}
            ]
        )

        call_args = mock_instance.ainvoke.call_args[0][0]
        assert "Line 1\nLine 2\nLine 3" in call_args


@pytest.mark.unit
class TestOllamaLLMSystemPrompt:
    """Test system prompt management."""

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
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Question 1"}
            ]
        )
        call1 = mock_instance.ainvoke.call_args[0][0]
        assert "System:" not in call1

        # Set system prompt
        llm.set_system_prompt("You are helpful")

        # Second call with system prompt
        mock_instance.ainvoke.reset_mock()
        await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Question 2"}
            ]
        )
        call2 = mock_instance.ainvoke.call_args[0][0]
        assert "System: You are helpful" in call2


@pytest.mark.unit
class TestOllamaLLMConversationHistory:
    """Test OllamaLLM conversation history support."""

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
    async def test_generate_response_with_empty_conversation_history(self, mock_ollama):
        """Test generate_response with empty conversation_history list."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)

        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Hello"}]
        )

        assert response == "Response"

        # Should just have current message
        call_args = mock_instance.ainvoke.call_args[0][0]
        assert call_args == "Human: Hello\n\nAssistant:"

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_generate_response_with_none_conversation_history(self, mock_ollama):
        """Test generate_response with None conversation_history."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)

        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

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


@pytest.mark.unit
class TestOllamaLLMGetLastResponseMetadata:
    """Test OllamaLLM get_last_response_metadata method."""

    def test_get_last_response_metadata_returns_copy(self):
        """Test that get_last_response_metadata returns a copy."""
        from llm_clients.ollama_llm import OllamaLLM

        with patch("llm_clients.ollama_llm.LangChainOllamaLLM") as mock_ollama:
            mock_instance = MagicMock()
            mock_ollama.return_value = mock_instance

            llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
            llm.last_response_metadata = {"test": "value"}

            metadata1 = llm.get_last_response_metadata()
            metadata2 = llm.get_last_response_metadata()

            # Should be equal but not the same object
            assert metadata1 == metadata2
            assert metadata1 is not metadata2

            # Modifying returned copy shouldn't affect internal state
            metadata1["modified"] = True
            assert "modified" not in llm.last_response_metadata

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_metadata_populated_after_successful_response(self, mock_ollama):
        """Test that metadata is populated correctly after successful response."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="This is a test response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER, model_name="llama3:8b")
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Hello, Ollama!"}
            ]
        )

        assert response == "This is a test response"

        # Verify metadata was extracted
        metadata = llm.get_last_response_metadata()
        assert metadata["response_id"] is None
        assert metadata["model"] == "llama3:8b"
        assert metadata["provider"] == "ollama"
        assert metadata["role"] == "provider"
        assert "timestamp" in metadata
        assert "response_time_seconds" in metadata
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_metadata_populated_after_error(self, mock_ollama):
        """Test that metadata is populated correctly after error."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(
            side_effect=ConnectionError("Could not connect to Ollama server")
        )
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER, model_name="llama3:8b")
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Test message"}
            ]
        )

        # Should return error message instead of raising
        assert "Error generating response" in response
        assert "Could not connect to Ollama server" in response

        # Verify error metadata was stored
        metadata = llm.get_last_response_metadata()
        assert metadata["response_id"] is None
        assert metadata["model"] == "llama3:8b"
        assert metadata["provider"] == "ollama"
        assert metadata["role"] == "provider"
        assert "timestamp" in metadata
        assert "error" in metadata
        assert "Could not connect to Ollama server" in metadata["error"]
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_metadata_tracks_timing(self, mock_ollama):
        """Test that response timing is tracked correctly."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Timed response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        assert "response_time_seconds" in metadata
        assert isinstance(metadata["response_time_seconds"], (int, float))
        assert metadata["response_time_seconds"] >= 0

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_timestamp_format(self, mock_ollama):
        """Test that timestamp is in ISO format."""
        from datetime import datetime

        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Test")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER)
        await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        timestamp = metadata["timestamp"]

        # Verify it's a valid ISO format timestamp
        try:
            datetime.fromisoformat(timestamp)
            timestamp_valid = True
        except ValueError:
            timestamp_valid = False

        assert timestamp_valid

    @pytest.mark.asyncio
    @patch("llm_clients.ollama_llm.LangChainOllamaLLM")
    async def test_metadata_structure_complete(self, mock_ollama):
        """Test that metadata structure includes all expected fields."""
        from llm_clients.ollama_llm import OllamaLLM

        mock_instance = MagicMock()
        mock_instance.ainvoke = AsyncMock(return_value="Complete response")
        mock_ollama.return_value = mock_instance

        llm = OllamaLLM(name="test-ollama", role=Role.PROVIDER, model_name="mistral:7b")
        await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()

        # Verify all expected fields are present
        assert "response_id" in metadata
        assert "model" in metadata
        assert "provider" in metadata
        assert "role" in metadata
        assert "timestamp" in metadata
        assert "response_time_seconds" in metadata
        assert "usage" in metadata

        # Verify field types
        assert metadata["response_id"] is None
        assert isinstance(metadata["model"], str)
        assert metadata["provider"] == "ollama"
        assert metadata["role"] == "provider"
        assert isinstance(metadata["timestamp"], str)
        assert isinstance(metadata["response_time_seconds"], (int, float))
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
