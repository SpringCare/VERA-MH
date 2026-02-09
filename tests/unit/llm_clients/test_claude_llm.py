"""Unit tests for ClaudeLLM class."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_clients import Role
from llm_clients.claude_llm import ClaudeLLM

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
@pytest.mark.usefixtures("mock_claude_config", "mock_claude_model")
class TestClaudeLLM(TestJudgeLLMBase):
    """Unit tests for ClaudeLLM class."""

    # ============================================================================
    # Factory Methods (Required by TestJudgeLLMBase)
    # ============================================================================

    def create_llm(self, role: Role, **kwargs):
        """Create ClaudeLLM instance for testing."""
        # Provide default name if not specified
        if "name" not in kwargs:
            kwargs["name"] = "TestClaude"

        with patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key"):
            with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat:
                mock_llm = MagicMock()
                mock_llm.model = kwargs.get("model_name", "claude-sonnet-4-5-20250929")
                mock_chat.return_value = mock_llm
                return ClaudeLLM(role=role, **kwargs)

    def get_provider_name(self) -> str:
        """Get provider name for metadata validation."""
        return "claude"

    @contextmanager
    def get_mock_patches(self):
        """Set up mocks for Claude.

        Note: Actual mocking is handled by class-level fixtures.
        This method provides a no-op context manager for base class compatibility.
        """
        yield

    # ============================================================================
    # Claude-Specific Tests
    # ============================================================================

    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", None)
    def test_init_missing_api_key_raises_error(self):
        """Test that missing ANTHROPIC_API_KEY raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ClaudeLLM(name="TestClaude", role=Role.PERSONA)

        assert "ANTHROPIC_API_KEY not found" in str(exc_info.value)

    def test_init_with_default_model(self):
        """Test initialization with default model from config."""
        llm = ClaudeLLM(
            name="TestClaude", role=Role.PERSONA, system_prompt="Test prompt"
        )

        assert llm.name == "TestClaude"
        assert llm.system_prompt == "Test prompt"
        assert llm.model_name == "claude-sonnet-4-5-20250929"
        assert llm.last_response_metadata == {}

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        llm = ClaudeLLM(
            name="TestClaude", role=Role.PERSONA, model_name="claude-3-opus-20240229"
        )

        assert llm.model_name == "claude-3-opus-20240229"

    def test_init_with_kwargs(self, default_llm_kwargs):
        """Test initialization with additional kwargs."""
        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-sonnet-4-5-20250929"
            mock_chat_anthropic.return_value = mock_llm

            ClaudeLLM(
                name="TestClaude",
                role=Role.PERSONA,
                **default_llm_kwargs,
            )

            # Verify kwargs were passed to ChatAnthropic
            call_kwargs = mock_chat_anthropic.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 500
            assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_success_with_system_prompt(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Test successful response generation with system prompt."""
        # Create mock response with metadata
        mock_response = mock_response_factory(
            text="This is a test response",
            response_id="msg_12345",
            provider="claude",
            metadata={
                "model": "claude-sonnet-4-5-20250929",
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "stop_reason": "end_turn",
            },
        )

        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(
            name="TestClaude",
            role=Role.PERSONA,
            system_prompt="You are a helpful assistant.",
        )
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "This is a test response"

        # Verify metadata was extracted
        metadata = assert_metadata_structure(
            llm, expected_provider="claude", expected_role=Role.PERSONA
        )
        assert metadata["response_id"] == "msg_12345"
        assert metadata["model"] == "claude-sonnet-4-5-20250929"
        assert_iso_timestamp(metadata["timestamp"])
        assert_response_timing(metadata)
        assert metadata["usage"]["input_tokens"] == 10
        assert metadata["usage"]["output_tokens"] == 20
        assert metadata["usage"]["total_tokens"] == 30
        assert metadata["stop_reason"] == "end_turn"
        assert "raw_metadata" in metadata

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_without_system_prompt(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Test response generation without system prompt."""
        mock_response = mock_response_factory(
            text="Response without system prompt",
            response_id="msg_67890",
            provider="claude",
            metadata={"model": "claude-sonnet-4-5-20250929"},
        )

        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)  # No system prompt
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response without system prompt"

        # Verify ainvoke was called with only HumanMessage (no SystemMessage)
        verify_no_system_message_in_call(mock_llm)

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_without_usage_metadata(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Test response when usage metadata is not available."""
        mock_response = mock_response_factory(
            text="Response",
            response_id="msg_abc",
            provider="claude",
            metadata={"model": "claude-sonnet-4-5-20250929"},
        )

        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_without_response_metadata(
        self, mock_chat_anthropic, mock_system_message
    ):
        """Test response when response_metadata attribute is missing."""
        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"

        # Response without response_metadata attribute
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "msg_xyz"
        del mock_response.response_metadata  # Remove attribute

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "claude-sonnet-4-5-20250929"
        assert metadata["usage"] == {}
        assert metadata["stop_reason"] is None

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_api_error(
        self, mock_chat_anthropic, mock_llm_factory, mock_system_message
    ):
        """Test error handling when API call fails."""
        mock_llm = mock_llm_factory(
            side_effect=Exception("API rate limit exceeded"),
            model="claude-sonnet-4-5-20250929",
        )
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        # Should return error message instead of raising
        assert_error_response(response, "API rate limit exceeded")

        # Verify error metadata was stored
        assert_error_metadata(llm, "claude", "API rate limit exceeded")

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_tracks_timing(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Test that response timing is tracked correctly."""
        mock_response = mock_response_factory(
            text="Timed response", response_id="msg_time", provider="claude"
        )

        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert_response_timing(metadata)

    def test_get_last_response_metadata_returns_copy(self):
        """Test that get_last_response_metadata returns a copy."""
        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        assert_metadata_copy_behavior(llm)

    def test_set_system_prompt(self):
        """Test set_system_prompt method."""
        llm = ClaudeLLM(
            name="TestClaude", role=Role.PERSONA, system_prompt="Initial prompt"
        )
        assert llm.system_prompt == "Initial prompt"

        llm.set_system_prompt("Updated prompt")
        assert llm.system_prompt == "Updated prompt"

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_with_partial_usage_metadata(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Test response with incomplete usage metadata."""
        mock_response = mock_response_factory(
            text="Partial usage response",
            response_id="msg_partial",
            provider="claude",
            metadata={
                "model": "claude-sonnet-4-5-20250929",
                "usage": {"input_tokens": 15},  # Missing output_tokens
            },
        )

        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Partial usage response"
        metadata = llm.get_last_response_metadata()
        assert metadata["usage"]["input_tokens"] == 15
        assert metadata["usage"]["output_tokens"] == 0  # Default value
        assert metadata["usage"]["total_tokens"] == 15

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_metadata_includes_response_object(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Test that metadata includes the full response object."""
        mock_response = mock_response_factory(
            text="Test", response_id="msg_obj", provider="claude"
        )

        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert "response" in metadata
        assert metadata["response"] == mock_response

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_timestamp_format(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Test that timestamp is in ISO format."""
        mock_response = mock_response_factory(
            text="Test", response_id="msg_time", provider="claude"
        )

        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert_iso_timestamp(metadata["timestamp"])

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_metadata_with_stop_reason(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Test metadata extraction of stop_reason."""
        mock_response = mock_response_factory(
            text="Stopped response",
            response_id="msg_stop",
            provider="claude",
            metadata={
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "max_tokens",
            },
        )

        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert metadata["stop_reason"] == "max_tokens"

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_raw_metadata_stored(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Test that raw metadata is stored."""
        mock_response = mock_response_factory(
            text="Test",
            response_id="msg_raw",
            provider="claude",
            metadata={
                "model": "claude-sonnet-4-5-20250929",
                "custom_field": "custom_value",
                "nested": {"key": "value"},
            },
        )

        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert "raw_metadata" in metadata
        assert metadata["raw_metadata"]["custom_field"] == "custom_value"
        assert metadata["raw_metadata"]["nested"]["key"] == "value"

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_with_conversation_history(
        self, mock_chat_anthropic, mock_response_factory, sample_conversation_history
    ):
        """Test generate_response with conversation_history parameter."""
        mock_response = mock_response_factory(
            text="Response with history",
            response_id="msg_history",
            provider="claude",
            metadata={
                "model": "claude-sonnet-4-5-20250929",
                "usage": {"input_tokens": 50, "output_tokens": 20},
            },
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", system_prompt="Test", role=Role.PROVIDER)

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
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_with_empty_conversation_history(
        self, mock_chat_anthropic, mock_response_factory
    ):
        """Test generate_response with empty conversation_history."""
        mock_response = mock_response_factory(
            text="Response", response_id="msg_empty", provider="claude"
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA, system_prompt="Test")

        response = await llm.generate_response(conversation_history=[])

        assert response == "Response"

        # Verify ainvoke was called
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]

        # Should have: SystemMessage only (no history messages)
        assert len(messages) == 1
        assert messages[0].text == "Test"

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_with_none_conversation_history(
        self, mock_chat_anthropic, mock_response_factory
    ):
        """Test generate_response with None conversation_history (default)."""
        mock_response = mock_response_factory(
            text="Response", response_id="msg_none", provider="claude"
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA, system_prompt="Test")

        # Test with None explicitly passed
        response = await llm.generate_response(conversation_history=None)

        assert response == "Response"

        # Verify ainvoke was called
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]

        # Should have: SystemMessage only (no history messages)
        assert len(messages) == 1
        assert messages[0].text == "Test"

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_with_persona_role_flips_types(
        self, mock_chat_anthropic, mock_response_factory, sample_conversation_history
    ):
        """Test that persona role flips message types in conversation history."""
        mock_response = mock_response_factory(
            text="Persona response", response_id="msg_persona", provider="claude"
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        # Persona system prompt should trigger message type flipping
        persona_prompt = "You are roleplaying as a human user"
        llm = ClaudeLLM(
            name="TestClaude", system_prompt=persona_prompt, role=Role.PERSONA
        )

        response = await llm.generate_response(
            conversation_history=sample_conversation_history
        )

        assert response == "Persona response"

        # Verify message types are flipped for persona role
        verify_message_types_for_persona(mock_llm, expected_message_count=4)

    @pytest.mark.asyncio
    async def test_generate_structured_response_success(self, mock_llm_factory):
        """Test successful structured response generation."""
        from pydantic import BaseModel, Field

        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-sonnet-4-5-20250929"

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

            mock_chat_anthropic.return_value = mock_llm

            llm = ClaudeLLM(
                name="TestClaude", role=Role.JUDGE, system_prompt="Test prompt"
            )
            response = await llm.generate_structured_response(
                "What is the answer?", TestResponse
            )

            assert isinstance(response, TestResponse)
            assert response.answer == "Yes"
            assert response.reasoning == "Because it's correct"

            # Verify metadata was stored
            metadata = assert_metadata_structure(
                llm, expected_provider="claude", expected_role=Role.JUDGE
            )
            assert metadata["model"] == "claude-sonnet-4-5-20250929"
            assert metadata["structured_output"] is True
            assert_response_timing(metadata)

    @pytest.mark.asyncio
    async def test_generate_structured_response_with_complex_model(
        self, mock_llm_factory
    ):
        """Test structured response with nested Pydantic model."""
        from pydantic import BaseModel, Field

        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-sonnet-4-5-20250929"

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

            mock_chat_anthropic.return_value = mock_llm

            llm = ClaudeLLM(name="TestClaude", role=Role.JUDGE)
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

        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-sonnet-4-5-20250929"

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

            mock_chat_anthropic.return_value = mock_llm

            llm = ClaudeLLM(name="TestClaude", role=Role.JUDGE)

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

        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-sonnet-4-5-20250929"

            class SimpleResponse(BaseModel):
                result: str

            test_response = SimpleResponse(result="success")

            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(return_value=test_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )

            mock_chat_anthropic.return_value = mock_llm

            llm = ClaudeLLM(name="TestClaude", role=Role.JUDGE)
            await llm.generate_structured_response("Test", SimpleResponse)

            metadata = llm.get_last_response_metadata()

            # Verify required fields
            assert metadata["provider"] == "claude"
            assert metadata["structured_output"] is True
            assert metadata["response_id"] is None
            assert_iso_timestamp(metadata["timestamp"])
            assert_response_timing(metadata)
