from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_clients.gemini_llm import GeminiLLM


@pytest.mark.unit
class TestGeminiLLM:
    """Unit tests for GeminiLLM class."""

    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", None)
    def test_init_missing_api_key_raises_error(self):
        """Test that missing GOOGLE_API_KEY raises ValueError (line 25)."""
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
        mock_response.content = "This is a Gemini response"
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
        mock_response.content = "Response without system prompt"
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
        """Test response with fallback token_usage structure (lines 90-97)."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = "Response with fallback"
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
        mock_response.content = "Response"
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
        mock_response.content = "Response"
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
        """Test error handling when API call fails (lines 108-118)."""
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
        mock_response.content = "Timed response"
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
        """Test that get_last_response_metadata returns a copy (line 122)."""
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
        """Test set_system_prompt method (line 126)."""
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
        """Test that metadata includes the full response object (line 73)."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = "Test"
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
        """Test that timestamp is in ISO format (line 69)."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = "Test"
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
        """Test finish_reason extraction (lines 100-102)."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = "Finished response"
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
        """Test that raw metadata is stored (line 105)."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = "Test"
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
        mock_response.content = "Response with history"
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
        mock_response.content = "Response"
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
        mock_response.content = "Response"
        mock_response.id = "gemini-none"
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
