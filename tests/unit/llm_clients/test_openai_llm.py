"""Unit tests for OpenAILLM class."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_clients import Role
from llm_clients.openai_llm import OpenAILLM

from .test_base_llm import TestJudgeLLMBase
from .test_helpers import (
    assert_error_metadata,
    assert_error_response,
    assert_iso_timestamp,
    assert_metadata_copy_behavior,
    assert_metadata_structure,
    assert_response_timing,
    verify_message_types_for_persona,
    verify_no_system_message_in_call,
)


@pytest.mark.unit
@pytest.mark.usefixtures("mock_openai_config", "mock_openai_model")
class TestOpenAILLM(TestJudgeLLMBase):
    """Unit tests for OpenAILLM class."""

    # ============================================================================
    # Factory Methods (Required by TestJudgeLLMBase)
    # ============================================================================

    def create_llm(self, role: Role, **kwargs):
        """Create OpenAILLM instance for testing."""
        # Provide default name if not specified
        if "name" not in kwargs:
            kwargs["name"] = "TestOpenAI"

        with patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key"):
            with patch("llm_clients.openai_llm.ChatOpenAI") as mock_chat:
                mock_llm = MagicMock()
                mock_chat.return_value = mock_llm
                return OpenAILLM(role=role, **kwargs)

    def get_provider_name(self) -> str:
        """Get provider name for metadata validation."""
        return "openai"

    @contextmanager
    def get_mock_patches(self):
        """Set up mocks for OpenAI.

        Note: Actual mocking is handled by class-level fixtures.
        This method provides a no-op context manager for base class compatibility.
        """
        yield

    # ============================================================================
    # OpenAI-Specific Tests
    # ============================================================================

    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", None)
    def test_init_missing_api_key_raises_error(self):
        """Test that missing OPENAI_API_KEY raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OpenAILLM(name="TestOpenAI", role=Role.PERSONA)

        assert "OPENAI_API_KEY not found" in str(exc_info.value)

    def test_init_with_default_model(self):
        """Test initialization with default model from config."""
        llm = OpenAILLM(
            name="TestOpenAI", role=Role.PERSONA, system_prompt="Test prompt"
        )

        assert llm.name == "TestOpenAI"
        assert llm.system_prompt == "Test prompt"
        assert llm.model_name == "gpt-5.2"
        assert llm.last_response_metadata == {}

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA, model_name="gpt-4o-turbo")

        assert llm.model_name == "gpt-4o-turbo"

    def test_init_with_kwargs(self, default_llm_kwargs):
        """Test initialization with additional kwargs."""
        with patch("llm_clients.openai_llm.ChatOpenAI") as mock_chat:
            OpenAILLM(
                name="TestOpenAI",
                role=Role.PERSONA,
                **default_llm_kwargs,
            )

            # Verify kwargs were passed to ChatOpenAI
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 500
            assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_success_with_system_prompt(
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test successful response generation with system prompt."""
        # Create mock response with comprehensive metadata
        mock_response = mock_response_factory(
            text="This is an OpenAI response",
            response_id="chatcmpl-12345",
            provider="openai",
            metadata={
                "model_name": "gpt-4o-0613",
                "token_usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40,
                },
                "finish_reason": "stop",
                "system_fingerprint": "fp_abc123",
                "logprobs": None,
                "additional_kwargs": {"function_call": None},
                "usage_metadata": {
                    "input_tokens": 15,
                    "output_tokens": 25,
                    "total_tokens": 40,
                },
            },
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(
            name="TestOpenAI",
            role=Role.PERSONA,
            system_prompt="You are a helpful assistant.",
        )
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "This is an OpenAI response"

        # Verify comprehensive metadata extraction
        metadata = assert_metadata_structure(
            llm, expected_provider="openai", expected_role=Role.PERSONA
        )
        assert metadata["response_id"] == "chatcmpl-12345"
        assert metadata["model"] == "gpt-4o-0613"
        assert_iso_timestamp(metadata["timestamp"])
        assert_response_timing(metadata)
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
    async def test_generate_response_without_system_prompt(
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test response generation without system prompt."""
        mock_response = mock_response_factory(
            text="Response without system prompt",
            response_id="chatcmpl-67890",
            provider="openai",
            metadata={"model_name": "gpt-4o"},
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)  # No system prompt
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response without system prompt"

        # Verify ainvoke was called with only HumanMessage
        # (turn 0 message, no SystemMessage)
        verify_no_system_message_in_call(mock_llm)

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_without_additional_kwargs(
        self, mock_chat_openai, mock_system_message
    ):
        """Test response when additional_kwargs is not available."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-abc"
        del mock_response.additional_kwargs  # Remove attribute

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["additional_kwargs"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_without_response_metadata(
        self, mock_chat_openai, mock_system_message
    ):
        """Test response when response_metadata attribute is missing."""
        mock_llm = MagicMock()

        mock_response = MagicMock(spec=["content", "id"])
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-xyz"

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gpt-5.2"
        assert metadata["usage"] == {}
        assert metadata["finish_reason"] is None

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_without_usage_metadata(
        self, mock_chat_openai, mock_system_message
    ):
        """Test response when usage_metadata attribute is missing."""
        mock_llm = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-usage"
        mock_response.response_metadata = {
            "model_name": "gpt-4o",
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
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        # Should still have usage from token_usage
        assert metadata["usage"]["prompt_tokens"] == 10
        assert metadata["usage"]["completion_tokens"] == 20
        assert metadata["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_api_error(
        self, mock_chat_openai, mock_llm_factory, mock_system_message
    ):
        """Test error handling when API call fails."""
        mock_llm = mock_llm_factory(side_effect=Exception("API rate limit exceeded"))
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        # Should return error message instead of raising
        assert_error_response(response, "API rate limit exceeded")

        # Verify error metadata was stored
        assert_error_metadata(llm, "openai", "API rate limit exceeded")

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_tracks_timing(
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test that response timing is tracked correctly."""
        mock_response = mock_response_factory(
            text="Timed response",
            response_id="chatcmpl-time",
            provider="openai",
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert_response_timing(metadata)

    def test_get_last_response_metadata_returns_copy(self):
        """Test that get_last_response_metadata returns a copy."""
        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        assert_metadata_copy_behavior(llm)

    def test_set_system_prompt(self):
        """Test set_system_prompt method."""
        llm = OpenAILLM(
            name="TestOpenAI", role=Role.PERSONA, system_prompt="Initial prompt"
        )
        assert llm.system_prompt == "Initial prompt"

        llm.set_system_prompt("Updated prompt")
        assert llm.system_prompt == "Updated prompt"

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_metadata_includes_response_object(
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test that metadata includes the full response object."""
        mock_response = mock_response_factory(
            text="Test", response_id="chatcmpl-obj", provider="openai"
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert "response" in metadata
        assert metadata["response"] == mock_response

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_timestamp_format(
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test that timestamp is in ISO format."""
        mock_response = mock_response_factory(
            text="Test", response_id="chatcmpl-ts", provider="openai"
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert_iso_timestamp(metadata["timestamp"])

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_model_name_update_from_metadata(
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test that model name is updated from response metadata."""
        mock_response = mock_response_factory(
            text="Test",
            response_id="chatcmpl-model",
            provider="openai",
            metadata={"model_name": "gpt-4o-0613-updated"},
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA, model_name="gpt-4o")
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gpt-4o-0613-updated"

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_with_conversation_history(
        self, mock_chat_openai, mock_response_factory, sample_conversation_history
    ):
        """Test generate_response with conversation_history parameter."""
        mock_response = mock_response_factory(
            text="Response with history",
            response_id="chatcmpl-history",
            provider="openai",
            metadata={
                "model_name": "gpt-4o-0613",
                "token_usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "total_tokens": 70,
                },
            },
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PROVIDER, system_prompt="Test")

        response = await llm.generate_response(
            conversation_history=sample_conversation_history
        )

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
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test generate_response with empty conversation_history list."""
        mock_response = mock_response_factory(
            text="Response",
            response_id="chatcmpl-empty",
            provider="openai",
            metadata={"model_name": "gpt-4o"},
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA, system_prompt="Test")

        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"

        # Should have: SystemMessage + turn 0 message
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_with_none_conversation_history(
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test generate_response with None conversation_history."""
        mock_response = mock_response_factory(
            text="Response",
            response_id="chatcmpl-none",
            provider="openai",
            metadata={"model_name": "gpt-4o"},
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA, system_prompt="Test")

        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"

        # Should have: SystemMessage + turn 0 message
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_with_persona_role_flips_types(
        self, mock_chat_openai, mock_response_factory, sample_conversation_history
    ):
        """Test that persona role flips message types in conversation history."""
        mock_response = mock_response_factory(
            text="Persona response",
            response_id="chatcmpl-persona",
            provider="openai",
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        # Persona system prompt should trigger message type flipping
        persona_prompt = "You are roleplaying as a human user"
        llm = OpenAILLM(
            name="TestOpenAI", role=Role.PERSONA, system_prompt=persona_prompt
        )

        response = await llm.generate_response(
            conversation_history=sample_conversation_history
        )

        assert response == "Persona response"

        # Verify message types are flipped for persona role
        verify_message_types_for_persona(mock_llm, expected_message_count=4)

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_generate_response_with_partial_usage_metadata(
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test response with incomplete usage metadata.

        OpenAI LLM gets total_tokens from metadata directly (doesn't calculate it).
        """
        # Response with only prompt_tokens in usage
        mock_response = mock_response_factory(
            text="Partial usage response",
            response_id="chatcmpl-partial",
            provider="openai",
            metadata={
                "model": "gpt-4o",
                "token_usage": {
                    "prompt_tokens": 15
                },  # Missing completion_tokens, total_tokens
            },
        )
        # Remove usage_metadata attribute to test only token_usage handling
        del mock_response.usage_metadata

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Partial usage response"
        metadata = llm.get_last_response_metadata()
        assert metadata["usage"]["prompt_tokens"] == 15
        assert metadata["usage"]["completion_tokens"] == 0  # Default value
        assert (
            metadata["usage"]["total_tokens"] == 0
        )  # Gets from metadata, doesn't calculate

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_metadata_with_finish_reason(
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test metadata extraction of finish_reason."""
        mock_response = mock_response_factory(
            text="Stopped response",
            response_id="chatcmpl-stop",
            provider="openai",
            metadata={"model": "gpt-4o", "finish_reason": "length"},
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert metadata["finish_reason"] == "length"

    @pytest.mark.asyncio
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.ChatOpenAI")
    async def test_raw_metadata_stored(
        self, mock_chat_openai, mock_response_factory, mock_system_message
    ):
        """Test that raw metadata is stored."""
        mock_response = mock_response_factory(
            text="Test",
            response_id="chatcmpl-raw",
            provider="openai",
            metadata={
                "model": "gpt-4o",
                "custom_field": "custom_value",
                "nested": {"key": "value"},
            },
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_openai.return_value = mock_llm

        llm = OpenAILLM(name="TestOpenAI", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        # OpenAI uses 'raw_response_metadata' instead of 'raw_metadata'
        assert "raw_response_metadata" in metadata
        assert metadata["raw_response_metadata"]["custom_field"] == "custom_value"
        assert metadata["raw_response_metadata"]["nested"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_generate_structured_response_success(self, mock_llm_factory):
        """Test successful structured response generation."""
        from pydantic import BaseModel, Field

        with patch("llm_clients.openai_llm.ChatOpenAI") as mock_chat:
            mock_llm = MagicMock()

            # Create a test Pydantic model
            class TestResponse(BaseModel):
                answer: str = Field(description="The answer")
                reasoning: str = Field(description="The reasoning")

            # Mock structured LLM
            mock_structured_llm = MagicMock()
            test_response = TestResponse(answer="Yes", reasoning="Because it's correct")
            mock_structured_llm.ainvoke = AsyncMock(return_value=test_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )

            mock_chat.return_value = mock_llm

            llm = OpenAILLM(
                name="TestOpenAI", role=Role.JUDGE, system_prompt="Test prompt"
            )
            response = await llm.generate_structured_response(
                "What is the answer?", TestResponse
            )

            assert isinstance(response, TestResponse)
            assert response.answer == "Yes"
            assert response.reasoning == "Because it's correct"

            # Verify metadata was stored
            metadata = assert_metadata_structure(
                llm, expected_provider="openai", expected_role=Role.JUDGE
            )
            assert metadata["model"] == "gpt-5.2"
            assert metadata["structured_output"] is True
            assert_response_timing(metadata)

    @pytest.mark.asyncio
    async def test_generate_structured_response_with_complex_model(
        self, mock_llm_factory
    ):
        """Test structured response with nested Pydantic model."""
        from pydantic import BaseModel, Field

        with patch("llm_clients.openai_llm.ChatOpenAI") as mock_chat:
            mock_llm = MagicMock()

            # Define nested Pydantic models
            class SubScore(BaseModel):
                value: int = Field(description="Score value")
                justification: str = Field(description="Justification")

            class ComplexResponse(BaseModel):
                overall_score: int = Field(description="Overall score")
                sub_scores: list[SubScore] = Field(description="Sub scores")
                summary: str = Field(description="Summary")

            # Create test response
            test_response = ComplexResponse(
                overall_score=85,
                sub_scores=[
                    SubScore(value=90, justification="Good quality"),
                    SubScore(value=80, justification="Needs improvement"),
                ],
                summary="Overall good performance",
            )

            # Mock structured LLM
            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(return_value=test_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )

            mock_chat.return_value = mock_llm

            llm = OpenAILLM(name="TestOpenAI", role=Role.JUDGE)
            response = await llm.generate_structured_response(
                "Evaluate this.", ComplexResponse
            )

            # Verify complex structure
            assert isinstance(response, ComplexResponse)
            assert response.overall_score == 85
            assert len(response.sub_scores) == 2
            assert response.sub_scores[0].value == 90
            assert response.summary == "Overall good performance"

    @pytest.mark.asyncio
    async def test_generate_structured_response_error(self):
        """Test error handling in structured response generation."""
        from pydantic import BaseModel

        with patch("llm_clients.openai_llm.ChatOpenAI") as mock_chat:
            mock_llm = MagicMock()

            class TestResponse(BaseModel):
                answer: str

            # Mock structured LLM to raise error
            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(
                side_effect=Exception("Structured output failed")
            )
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )

            mock_chat.return_value = mock_llm

            llm = OpenAILLM(name="TestOpenAI", role=Role.JUDGE)

            with pytest.raises(RuntimeError) as exc_info:
                await llm.generate_structured_response("Test", TestResponse)

            assert "Error generating structured response" in str(exc_info.value)
            assert "Structured output failed" in str(exc_info.value)

            # Verify error metadata was stored
            metadata = llm.get_last_response_metadata()
            assert "error" in metadata
            assert "Structured output failed" in metadata["error"]

    @pytest.mark.asyncio
    async def test_structured_response_metadata_fields(self):
        """Test that structured response metadata includes correct fields."""
        from pydantic import BaseModel

        with patch("llm_clients.openai_llm.ChatOpenAI") as mock_chat:
            mock_llm = MagicMock()

            class SimpleResponse(BaseModel):
                result: str

            test_response = SimpleResponse(result="success")

            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(return_value=test_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )

            mock_chat.return_value = mock_llm

            llm = OpenAILLM(name="TestOpenAI", role=Role.JUDGE)
            await llm.generate_structured_response("Test", SimpleResponse)

            metadata = llm.get_last_response_metadata()

            # Verify required fields
            assert metadata["provider"] == "openai"
            assert metadata["structured_output"] is True
            assert metadata["response_id"] is None
            assert_iso_timestamp(metadata["timestamp"])
            assert_response_timing(metadata)
