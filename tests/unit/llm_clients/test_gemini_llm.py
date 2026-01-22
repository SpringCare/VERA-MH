from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from llm_clients.gemini_llm import GeminiLLM


@pytest.mark.unit
class TestGeminiLLM:
    """Unit tests for GeminiLLM class."""

    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", None)
    def test_init_missing_api_key_raises_error(self):
        """Test that missing GOOGLE_API_KEY raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            GeminiLLM(name="TestGemini")

        assert "GOOGLE_API_KEY not found" in str(exc_info.value)

    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    def test_init_with_default_model(self, mock_chat_gemini):
        """Test initialization with default model from config."""
        mock_llm = MagicMock()
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini", system_prompt="Test prompt")

        assert llm.name == "TestGemini"
        assert llm.system_prompt == "Test prompt"
        assert llm.model_name == "gemini-1.5-pro"
        assert llm.last_response_metadata == {}

    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    def test_init_with_custom_model(self, mock_chat_gemini):
        """Test initialization with custom model name."""
        mock_llm = MagicMock()
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini", model_name="gemini-1.5-flash")

        assert llm.model_name == "gemini-1.5-flash"

    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    def test_init_with_kwargs(self, mock_chat_gemini):
        """Test initialization with additional kwargs."""
        mock_llm = MagicMock()
        mock_chat_gemini.return_value = mock_llm

        GeminiLLM(name="TestGemini", temperature=0.5, max_tokens=500, top_p=0.9)

        # Verify kwargs were passed to ChatGoogleGenerativeAI
        call_kwargs = mock_chat_gemini.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_success_with_system_prompt(self, mock_chat_gemini):
        """Test successful response generation with system prompt."""
        mock_llm = MagicMock()

        # Create mock response with Gemini-style metadata
        mock_response = MagicMock()
        mock_response.text = "This is a Gemini response"
        mock_response.id = "gemini-12345"

        # Mock response_metadata object with model_name attribute
        mock_metadata_obj = MagicMock()
        mock_metadata_obj.model_name = "gemini-1.5-pro-001"
        mock_response.response_metadata = mock_metadata_obj

        # Add dictionary items for usage extraction
        mock_metadata_obj.__getitem__ = lambda self, key: {
            "usage_metadata": {
                "prompt_token_count": 12,
                "candidates_token_count": 28,
                "total_token_count": 40,
            },
            "finish_reason": "STOP",
        }.get(key)
        mock_metadata_obj.__contains__ = lambda self, key: key in [
            "usage_metadata",
            "finish_reason",
        ]
        mock_metadata_obj.get = lambda key, default=None: {
            "finish_reason": "STOP",
        }.get(key, default)

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini", system_prompt="You are a helpful assistant.")
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Hello, Gemini!"}
            ]
        )

        assert response == "This is a Gemini response"

        # Verify metadata extraction
        metadata = llm.get_last_response_metadata()
        assert metadata["response_id"] == "gemini-12345"
        assert metadata["model"] == "gemini-1.5-pro-001"
        assert metadata["provider"] == "gemini"
        assert "timestamp" in metadata
        assert "response_time_seconds" in metadata
        assert metadata["usage"]["prompt_token_count"] == 12
        assert metadata["usage"]["candidates_token_count"] == 28
        assert metadata["usage"]["total_token_count"] == 40
        assert metadata["finish_reason"] == "STOP"
        assert "raw_metadata" in metadata

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_without_system_prompt(self, mock_chat_gemini):
        """Test response generation without system prompt."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Response without system prompt"
        mock_response.id = "gemini-67890"
        mock_response.response_metadata = {"model_name": "gemini-1.5-pro"}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")  # No system prompt
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Test message"}
            ]
        )

        assert response == "Response without system prompt"

        # Verify ainvoke was called with only HumanMessage (no SystemMessage)
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].content == "Test message"

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_with_fallback_token_usage(self, mock_chat_gemini):
        """Test response with fallback token_usage structure."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Response with fallback"
        mock_response.id = "gemini-fallback"
        mock_response.response_metadata = {
            "model_name": "gemini-1.5-pro",
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        assert response == "Response with fallback"
        metadata = llm.get_last_response_metadata()
        # Should use fallback structure
        assert metadata["usage"]["prompt_tokens"] == 10
        assert metadata["usage"]["completion_tokens"] == 20
        assert metadata["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_without_usage_metadata(self, mock_chat_gemini):
        """Test response when no usage metadata is available."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "gemini-no-usage"
        mock_response.response_metadata = {"model_name": "gemini-1.5-pro"}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_without_response_metadata(self, mock_chat_gemini):
        """Test response when response_metadata attribute is missing."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "gemini-xyz"
        del mock_response.response_metadata  # Remove attribute

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gemini-1.5-pro"
        assert metadata["usage"] == {}
        assert metadata["finish_reason"] is None

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_api_error(self, mock_chat_gemini):
        """Test error handling when API call fails."""
        mock_llm = MagicMock()

        # Simulate API error
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API quota exceeded"))
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Test message"}
            ]
        )

        # Should return error message instead of raising
        assert "Error generating response" in response
        assert "API quota exceeded" in response

        # Verify error metadata was stored
        metadata = llm.get_last_response_metadata()
        assert metadata["response_id"] is None
        assert metadata["model"] == "gemini-1.5-pro"
        assert metadata["provider"] == "gemini"
        assert "timestamp" in metadata
        assert "error" in metadata
        assert "API quota exceeded" in metadata["error"]
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_tracks_timing(self, mock_chat_gemini):
        """Test that response timing is tracked correctly."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Timed response"
        mock_response.id = "gemini-time"
        mock_response.response_metadata = {}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        assert "response_time_seconds" in metadata
        assert isinstance(metadata["response_time_seconds"], (int, float))
        assert metadata["response_time_seconds"] >= 0

    def test_get_last_response_metadata_returns_copy(self):
        """Test that get_last_response_metadata returns a copy."""
        with patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key"):
            with patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI") as mock_chat:
                mock_llm = MagicMock()
                mock_chat.return_value = mock_llm

                llm = GeminiLLM(name="TestGemini")
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
        """Test set_system_prompt method."""
        with patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key"):
            with patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI") as mock_chat:
                mock_llm = MagicMock()
                mock_chat.return_value = mock_llm

                llm = GeminiLLM(name="TestGemini", system_prompt="Initial prompt")
                assert llm.system_prompt == "Initial prompt"

                llm.set_system_prompt("Updated prompt")
                assert llm.system_prompt == "Updated prompt"

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_metadata_includes_response_object(self, mock_chat_gemini):
        """Test that metadata includes the full response object."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.id = "gemini-obj"
        mock_response.response_metadata = {}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        assert "response" in metadata
        assert metadata["response"] == mock_response

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_timestamp_format(self, mock_chat_gemini):
        """Test that timestamp is in ISO format."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.id = "gemini-ts"
        mock_response.response_metadata = {}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
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
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_finish_reason_extraction(self, mock_chat_gemini):
        """Test finish_reason extraction."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Finished response"
        mock_response.id = "gemini-finish"
        mock_response.response_metadata = {
            "model_name": "gemini-1.5-pro",
            "finish_reason": "MAX_TOKENS",
        }

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        assert metadata["finish_reason"] == "MAX_TOKENS"

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_raw_metadata_stored(self, mock_chat_gemini):
        """Test that raw metadata is stored."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.id = "gemini-raw"
        mock_response.response_metadata = {
            "model_name": "gemini-1.5-pro",
            "custom_field": "custom_value",
            "nested": {"key": "value"},
        }

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        assert "raw_metadata" in metadata
        assert metadata["raw_metadata"]["custom_field"] == "custom_value"
        assert metadata["raw_metadata"]["nested"]["key"] == "value"

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_with_conversation_history(self, mock_chat_gemini):
        """Test generate_response with conversation_history parameter."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response with history"
        mock_response.id = "gemini-history"
        mock_response.response_metadata = {
            "model_name": "gemini-1.5-pro",
            "token_usage": {
                "prompt_token_count": 50,
                "candidates_token_count": 20,
                "total_token_count": 70,
            },
        }

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini", system_prompt="Test")

        # Provide conversation history including the current turn
        history = [
            {
                "turn": 1,
                "speaker": "persona",
                "input": "Start",
                "response": "Hello",
                "early_termination": False,
                "logging": {},
            },
            {
                "turn": 2,
                "speaker": "agent",
                "input": "Hello",
                "response": "Hi there",
                "early_termination": False,
                "logging": {},
            },
            {
                "turn": 3,
                "speaker": "persona",
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
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_with_empty_conversation_history(
        self, mock_chat_gemini
    ):
        """Test generate_response with empty conversation_history list."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "gemini-empty"
        mock_response.response_metadata = {"model_name": "gemini-1.5-pro"}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini", system_prompt="Test")

        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Hi"}]
        )

        assert response == "Response"

        # Should have: SystemMessage + current message only
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_with_none_conversation_history(
        self, mock_chat_gemini
    ):
        """Test generate_response with None conversation_history."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "gemini-none"
        mock_response.response_metadata = {"model_name": "gemini-1.5-pro"}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini", system_prompt="Test")

        # Actually pass None to test the default behavior
        response = await llm.generate_response(conversation_history=None)

        assert response == "Response"

        # Should have: SystemMessage only (no history messages)
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_response_with_persona_role_flips_types(
        self, mock_chat_gemini
    ):
        """Test that persona role flips message types in conversation history."""

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Persona response"
        mock_response.id = "gemini-persona"
        mock_response.response_metadata = {}

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_gemini.return_value = mock_llm

        # Persona system prompt should trigger message type flipping
        persona_prompt = "You are roleplaying as a human user"
        llm = GeminiLLM(name="TestGemini", system_prompt=persona_prompt)

        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "provider", "response": "Hi there"},
            {"turn": 3, "speaker": "persona", "response": "How are you?"},
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
        assert messages[1].content == "Hello"
        # Turn 2 (provider, even) should be HumanMessage when persona role
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].content == "Hi there"
        # Turn 3 (persona, odd) should be AIMessage when persona role
        assert isinstance(messages[3], AIMessage)
        assert messages[3].content == "How are you?"

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_structured_response_success_with_system_prompt(
        self, mock_chat_gemini
    ):
        """Test successful structured response generation with system prompt."""

        # Define a simple Pydantic model for testing
        class TestResponse(BaseModel):
            answer: str
            reasoning: str

        mock_llm = MagicMock()

        # Create mock structured LLM that returns a Pydantic model instance
        mock_structured_llm = MagicMock()
        expected_response = TestResponse(
            answer="Yes", reasoning="The response was appropriate."
        )
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini", system_prompt="You are a helpful assistant.")
        response = await llm.generate_structured_response(
            message="Evaluate this conversation.", response_model=TestResponse
        )

        # Verify response is correct type and has expected values
        assert isinstance(response, TestResponse)
        assert response.answer == "Yes"
        assert response.reasoning == "The response was appropriate."

        # Verify with_structured_output was called with correct model
        mock_llm.with_structured_output.assert_called_once_with(TestResponse)

        # Verify ainvoke was called with correct messages
        call_args = mock_structured_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "You are a helpful assistant."
        assert isinstance(call_args[1], HumanMessage)
        assert call_args[1].content == "Evaluate this conversation."

        # Verify metadata was stored
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gemini-1.5-pro"
        assert metadata["provider"] == "gemini"
        assert metadata["structured_output"] is True
        assert "timestamp" in metadata
        assert "response_time_seconds" in metadata
        assert metadata["response_id"] is None
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_structured_response_success_without_system_prompt(
        self, mock_chat_gemini
    ):
        """Test successful structured response generation without system prompt."""

        class TestResponse(BaseModel):
            answer: str
            reasoning: str

        mock_llm = MagicMock()

        mock_structured_llm = MagicMock()
        expected_response = TestResponse(
            answer="No", reasoning="The response was not appropriate."
        )
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")  # No system prompt
        response = await llm.generate_structured_response(
            message="Evaluate this conversation.", response_model=TestResponse
        )

        assert isinstance(response, TestResponse)
        assert response.answer == "No"

        # Verify ainvoke was called with only HumanMessage (no SystemMessage)
        call_args = mock_structured_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], HumanMessage)
        assert call_args[0].content == "Evaluate this conversation."

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_structured_response_with_empty_message(
        self, mock_chat_gemini
    ):
        """Test structured response generation with empty string message."""

        class TestResponse(BaseModel):
            answer: str

        mock_llm = MagicMock()

        mock_structured_llm = MagicMock()
        expected_response = TestResponse(answer="Maybe")
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        response = await llm.generate_structured_response(
            message="", response_model=TestResponse
        )

        assert isinstance(response, TestResponse)
        assert response.answer == "Maybe"

        # Verify empty message was passed as HumanMessage
        call_args = mock_structured_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], HumanMessage)
        assert call_args[0].content == ""

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_structured_response_api_error(self, mock_chat_gemini):
        """Test error handling when structured response API call fails."""

        class TestResponse(BaseModel):
            answer: str

        mock_llm = MagicMock()

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(
            side_effect=Exception("API quota exceeded")
        )
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")

        # Should raise RuntimeError with error message
        with pytest.raises(RuntimeError) as exc_info:
            await llm.generate_structured_response(
                message="Test message", response_model=TestResponse
            )

        assert "Error generating structured response" in str(exc_info.value)
        assert "API quota exceeded" in str(exc_info.value)

        # Verify error metadata was stored
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gemini-1.5-pro"
        assert metadata["provider"] == "gemini"
        assert "timestamp" in metadata
        assert "error" in metadata
        assert "API quota exceeded" in metadata["error"]
        assert metadata["usage"] == {}
        assert "structured_output" not in metadata  # Not set on error

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_structured_response_type_validation_error(
        self, mock_chat_gemini
    ):
        """Test error handling when response is not the correct type."""

        class TestResponse(BaseModel):
            answer: str

        class WrongResponse(BaseModel):
            value: int

        mock_llm = MagicMock()

        mock_structured_llm = MagicMock()
        # Return wrong type (shouldn't happen in practice, but test the validation)
        wrong_response = WrongResponse(value=42)
        mock_structured_llm.ainvoke = AsyncMock(return_value=wrong_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")

        # Should raise RuntimeError due to type validation
        with pytest.raises(RuntimeError) as exc_info:
            await llm.generate_structured_response(
                message="Test message", response_model=TestResponse
            )

        assert "Error generating structured response" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_structured_response_tracks_timing(self, mock_chat_gemini):
        """Test that structured response timing is tracked correctly."""

        class TestResponse(BaseModel):
            answer: str

        mock_llm = MagicMock()

        mock_structured_llm = MagicMock()
        expected_response = TestResponse(answer="Test")
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        await llm.generate_structured_response(
            message="Test message", response_model=TestResponse
        )

        metadata = llm.get_last_response_metadata()
        assert "response_time_seconds" in metadata
        assert isinstance(metadata["response_time_seconds"], (int, float))
        assert metadata["response_time_seconds"] >= 0

    @pytest.mark.asyncio
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_structured_response_timestamp_format(
        self, mock_chat_gemini
    ):
        """Test that structured response timestamp is in ISO format."""

        class TestResponse(BaseModel):
            answer: str

        mock_llm = MagicMock()

        mock_structured_llm = MagicMock()
        expected_response = TestResponse(answer="Test")
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        await llm.generate_structured_response(
            message="Test message", response_model=TestResponse
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
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.ChatGoogleGenerativeAI")
    async def test_generate_structured_response_with_complex_model(
        self, mock_chat_gemini
    ):
        """Test structured response with a more complex Pydantic model."""

        class NestedModel(BaseModel):
            value: int

        class ComplexResponse(BaseModel):
            answer: str
            score: int
            nested: NestedModel
            optional_field: str | None = None

        mock_llm = MagicMock()

        mock_structured_llm = MagicMock()
        expected_response = ComplexResponse(
            answer="Yes",
            score=85,
            nested=NestedModel(value=42),
            optional_field="optional",
        )
        mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        mock_chat_gemini.return_value = mock_llm

        llm = GeminiLLM(name="TestGemini")
        response = await llm.generate_structured_response(
            message="Evaluate this.", response_model=ComplexResponse
        )

        assert isinstance(response, ComplexResponse)
        assert response.answer == "Yes"
        assert response.score == 85
        assert isinstance(response.nested, NestedModel)
        assert response.nested.value == 42
        assert response.optional_field == "optional"
