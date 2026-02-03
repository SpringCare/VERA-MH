from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_clients import Role
from llm_clients.openai_llm import OpenAILLM


@pytest.mark.unit
class TestOpenAILLM:
    """Unit tests for OpenAILLM class."""

    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", None)
    def test_init_missing_api_key_raises_error(self):
        """Test that missing OPENAI_API_KEY raises ValueError (line 25)."""
        with pytest.raises(ValueError) as exc_info:
            OpenAILLM(name="TestOpenAI", role=Role.PERSONA)

        assert "OPENAI_API_KEY not found" in str(exc_info.value)

    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    def test_init_with_default_model(self, mock_chat_openai):
        """Test initialization with default model from config."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(
            name="TestOpenAI", role=Role.PERSONA, system_prompt="Test prompt"
        )

        assert llm.name == "TestOpenAI"
        assert llm.system_prompt == "Test prompt"
        assert llm.model_name == "gpt-4"
        assert llm.last_response_metadata == {}

    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    def test_init_with_custom_model(self, mock_chat_openai):
        """Test initialization with custom model name."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA, model_name="gpt-4-turbo")

        assert llm.model_name == "gpt-4-turbo"

    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    def test_init_with_kwargs(self, mock_chat_openai):
        """Test initialization with additional kwargs."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        OpenAILLM(
            name="TestOpenAI",
            role=Role.PERSONA,
            temperature=0.5,
            max_tokens=500,
            top_p=0.9,
        )

        # Verify kwargs were passed to ChatOpenAI
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_success_with_system_prompt(self, mock_chat_openai):
        """Test successful response generation with system prompt."""
        mock_llm = MagicMock()

        # Create mock response with comprehensive metadata
        mock_response = MagicMock()
        mock_response.text = "This is an OpenAI response"
        mock_response.id = "chatcmpl-12345"
        mock_response.additional_kwargs = {"function_call": None}
        mock_response.response_metadata = {
            "model_name": "gpt-4-0613",
            "token_usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
            "finish_reason": "stop",
            "system_fingerprint": "fp_abc123",
            "logprobs": None,
        }
        mock_response.usage_metadata = {
            "input_tokens": 15,
            "output_tokens": 25,
            "total_tokens": 40,
        }

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(
            name="TestOpenAI",
            role=Role.PERSONA,
            system_prompt="You are a helpful assistant.",
        )
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Hello, GPT!"}]
        )

        assert response == "This is an OpenAI response"

        # Verify comprehensive metadata extraction
        metadata = llm.get_last_response_metadata()
        assert metadata["response_id"] == "chatcmpl-12345"
        assert metadata["model"] == "gpt-4-0613"
        assert metadata["provider"] == "openai"
        assert "timestamp" in metadata
        assert "response_time_seconds" in metadata
        assert metadata["usage"]["input_tokens"] == 15
        assert metadata["usage"]["output_tokens"] == 25
        assert metadata["usage"]["total_tokens"] == 40
        assert metadata["finish_reason"] == "stop"
        assert metadata["system_fingerprint"] == "fp_abc123"
        assert metadata["logprobs"] is None
        assert "additional_kwargs" in metadata
        assert "raw_response_metadata" in metadata
        assert "raw_usage_metadata" in metadata

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_without_system_prompt(self, mock_chat_openai):
        """Test response generation without system prompt."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Response without system prompt"
        mock_response.id = "chatcmpl-67890"
        mock_response.response_metadata = {"model_name": "gpt-4"}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)  # No system prompt
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test message"}]
        )

        assert response == "Response without system prompt"

        # Verify ainvoke was called with only HumanMessage
        # (turn 0 message, no SystemMessage)
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].text == "Test message"

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_without_additional_kwargs(self, mock_chat_openai):
        """Test response when additional_kwargs is not available."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-abc"
        del mock_response.additional_kwargs  # Remove attribute

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test"}]
        )

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["additional_kwargs"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_without_response_metadata(self, mock_chat_openai):
        """Test response when response_metadata attribute is missing."""
        mock_llm = MagicMock()

        mock_response = MagicMock(spec=["content", "id"])
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-xyz"

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test"}]
        )

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gpt-4"
        assert metadata["usage"] == {}
        assert metadata["finish_reason"] is None

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_without_usage_metadata(self, mock_chat_openai):
        """Test response when usage_metadata attribute is missing."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-usage"
        mock_response.response_metadata = {
            "model_name": "gpt-4",
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        del mock_response.usage_metadata  # Remove attribute

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test"}]
        )

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        # Should still have usage from token_usage
        assert metadata["usage"]["prompt_tokens"] == 10
        assert metadata["usage"]["completion_tokens"] == 20
        assert metadata["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_api_error(self, mock_chat_openai):
        """Test error handling when API call fails (lines 124-137)."""
        mock_llm = MagicMock()

        # Simulate API error
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API rate limit exceeded"))
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test message"}]
        )

        # Should return error message instead of raising
        assert "Error generating response" in response
        assert "API rate limit exceeded" in response

        # Verify error metadata was stored
        metadata = llm.get_last_response_metadata()
        assert metadata["response_id"] is None
        assert metadata["model"] == "gpt-4"
        assert metadata["provider"] == "openai"
        assert "timestamp" in metadata
        assert "error" in metadata
        assert "API rate limit exceeded" in metadata["error"]
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_tracks_timing(self, mock_chat_openai):
        """Test that response timing is tracked correctly."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Timed response"
        mock_response.id = "chatcmpl-time"
        mock_response.response_metadata = {}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        assert "response_time_seconds" in metadata
        assert isinstance(metadata["response_time_seconds"], (int, float))
        assert metadata["response_time_seconds"] >= 0

    def test_get_last_response_metadata_returns_copy(self):
        """Test that get_last_response_metadata returns a copy (line 141)."""
        with patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key"):
            with patch("llm_clients.openai_llm.ChatOpenAI") as mock_chat:
                mock_llm = MagicMock()
                mock_chat.return_value = mock_llm

                llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
                llm.last_response_metadata = {"test": "value"}

                metadata1 = llm.get_last_response_metadata()
                metadata2 = llm.get_last_response_metadata()

                # Should be equal but not the same object
                assert metadata1 == metadata2
                assert metadata1 is not metadata2

                # Modifying returned copy shouldn't affect internal state
                metadata1["modified"] = True
                assert "modified" not in llm.last_response_metadata

    def test_set_system_prompt(self):
        """Test set_system_prompt method (line 145)."""
        with patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key"):
            with patch("llm_clients.openai_llm.ChatOpenAI") as mock_chat:
                mock_llm = MagicMock()
                mock_chat.return_value = mock_llm

                llm = OpenAILLM(
                    name="TestOpenAI", role=Role.PERSONA, system_prompt="Initial prompt"
                )
                assert llm.system_prompt == "Initial prompt"

                llm.set_system_prompt("Updated prompt")
                assert llm.system_prompt == "Updated prompt"

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_metadata_includes_response_object(self, mock_chat_openai):
        """Test that metadata includes the full response object (line 71)."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.id = "chatcmpl-obj"
        mock_response.response_metadata = {}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        assert "response" in metadata
        assert metadata["response"] == mock_response

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_timestamp_format(self, mock_chat_openai):
        """Test that timestamp is in ISO format (line 64)."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.id = "chatcmpl-ts"
        mock_response.response_metadata = {}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test"}]
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
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_model_name_update_from_metadata(self, mock_chat_openai):
        """Test that model name is updated from response metadata (lines 85-86)."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.id = "chatcmpl-model"
        mock_response.response_metadata = {"model_name": "gpt-4-0613-updated"}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA, model_name="gpt-4")
        await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gpt-4-0613-updated"

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_with_conversation_history(self, mock_chat_openai):
        """Test generate_response with conversation_history parameter."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response with history"
        mock_response.id = "chatcmpl-history"
        mock_response.response_metadata = {
            "model_name": "gpt-4-0613",
            "token_usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
            },
        }

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PROVIDER, system_prompt="Test")

        # Provide conversation history including the current turn
        history = [
            {
                "turn": 1,
                "speaker": Role.PERSONA,
                "input": "Start",
                "response": "Hello",
                "early_termination": False,
                "logging": {},
            },
            {
                "turn": 2,
                "speaker": Role.PROVIDER,
                "input": "Hello",
                "response": "Hi there",
                "early_termination": False,
                "logging": {},
            },
            {
                "turn": 3,
                "speaker": Role.PERSONA,
                "input": "Hi there",
                "response": "How are you?",
                "early_termination": False,
                "logging": {},
            },
        ]

        response = await llm.generate_response(conversation_history=history)

        assert response == "Response with history"

        # Verify ainvoke was called with correct messages
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]

        # Should have: SystemMessage + 3 history messages
        assert len(messages) == 4

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_with_empty_conversation_history(
        self, mock_chat_openai
    ):
        """Test generate_response with empty conversation_history list."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-empty"
        mock_response.response_metadata = {"model_name": "gpt-4"}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA, system_prompt="Test")

        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Hi"}]
        )

        assert response == "Response"

        # Should have: SystemMessage + turn 0 message
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_with_none_conversation_history(
        self, mock_chat_openai
    ):
        """Test generate_response with None conversation_history."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-none"
        mock_response.response_metadata = {"model_name": "gpt-4"}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA, system_prompt="Test")

        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Hi"}]
        )

        assert response == "Response"

        # Should have: SystemMessage + turn 0 message
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_with_persona_role_flips_types(
        self, mock_chat_openai
    ):
        """Test that persona role flips message types in conversation history."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Persona response"
        mock_response.id = "chatcmpl-persona"
        mock_response.response_metadata = {}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        # Persona system prompt should trigger message type flipping
        persona_prompt = "You are roleplaying as a human user"
        llm = OpenAILLM(
            name="TestOpenAI", role=Role.PERSONA, system_prompt=persona_prompt
        )

        history = [
            {
                "turn": 1,
                "speaker": Role.PERSONA,
                "input": "",
                "response": "Hello",
                "early_termination": False,
                "logging": {},
            },
            {
                "turn": 2,
                "speaker": Role.PROVIDER,
                "input": "Hello",
                "response": "Hi there",
                "early_termination": False,
                "logging": {},
            },
            {
                "turn": 3,
                "speaker": Role.PERSONA,
                "input": "Hi there",
                "response": "How are you?",
                "early_termination": False,
                "logging": {},
            },
        ]

        response = await llm.generate_response(conversation_history=history)

        assert response == "Persona response"

        # Verify message types are flipped for persona role
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]

        # Should have: SystemMessage + 3 history messages
        assert len(messages) == 4
        assert isinstance(messages[0], SystemMessage)
        # Turn 1 (persona, odd) should be AIMessage when persona role
        assert isinstance(messages[1], AIMessage)
        assert messages[1].text == "Hello"
        # Turn 2 (provider, even) should be HumanMessage when persona role
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].text == "Hi there"
        # Turn 3 (persona, odd) should be AIMessage when persona role
        assert isinstance(messages[3], AIMessage)
        assert messages[3].text == "How are you?"
