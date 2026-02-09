"""Unit tests for AzureLLM class."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from llm_clients import Role
from llm_clients.azure_llm import AzureLLM

from .test_base_llm import TestJudgeLLMBase
from .test_helpers import (
    assert_error_metadata,
    assert_error_response,
    assert_iso_timestamp,
    assert_metadata_copy_behavior,
    assert_metadata_structure,
    assert_response_timing,
    verify_message_types_for_persona,
)


# Helper class for mocking response_metadata that supports both dict and
# attribute access
class DictWithAttr(dict):
    """Dict that supports both dict operations and attribute access."""

    def __getattr__(self, key):
        return self.get(key)


@pytest.fixture
def mock_azure_model():
    """Fixture to patch AzureAIChatCompletionsModel."""
    with patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock:
        yield mock


def create_mock_response(
    text="Test response", response_id="chatcmpl-12345", **metadata
):
    """Helper to create a mock Azure response."""
    mock_response = MagicMock()
    mock_response.text = text
    mock_response.id = response_id
    mock_response.response_metadata = DictWithAttr({"model": "gpt-5.2", **metadata})
    return mock_response


@pytest.mark.unit
@pytest.mark.usefixtures("mock_azure_config", "mock_azure_model")
class TestAzureLLM(TestJudgeLLMBase):
    """Unit tests for AzureLLM class."""

    # ============================================================================
    # Factory Methods (Required by TestJudgeLLMBase)
    # ============================================================================

    def create_llm(self, role: Role, **kwargs):
        """Create AzureLLM instance for testing."""
        # Provide default name if not specified
        if "name" not in kwargs:
            kwargs["name"] = "TestAzure"

        with (
            patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key"),
            patch(
                "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
                "https://test.openai.azure.com",
            ),
            patch(
                "llm_clients.azure_llm.Config.get_azure_config",
                return_value={"model": "gpt-5.2"},
            ),
            patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock_model,
        ):
            mock_llm = MagicMock()
            mock_llm.model_name = "gpt-5.2"
            mock_model.return_value = mock_llm
            return AzureLLM(role=role, **kwargs)

    def get_provider_name(self) -> str:
        """Get provider name for metadata validation."""
        return "azure"

    @contextmanager
    def get_mock_patches(self):
        """Set up mocks for Azure.

        Note: Actual mocking is handled by class-level fixtures.
        This method provides a no-op context manager for base class compatibility.
        """
        yield

    # ============================================================================
    # Azure-Specific Tests
    # ============================================================================

    """Unit tests for AzureLLM class."""

    def test_init_missing_api_key_raises_error(self):
        """Test that missing AZURE_API_KEY raises ValueError."""
        with patch("llm_clients.azure_llm.Config.AZURE_API_KEY", None):
            with pytest.raises(ValueError) as exc_info:
                AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert "AZURE_API_KEY not found" in str(exc_info.value)

    def test_init_missing_endpoint_raises_error(self):
        """Test that missing AZURE_ENDPOINT raises ValueError."""
        with (
            patch("llm_clients.azure_llm.Config.AZURE_ENDPOINT", None),
            patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key"),
        ):
            with pytest.raises(ValueError) as exc_info:
                AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert "AZURE_ENDPOINT not found" in str(exc_info.value)

    def test_init_with_default_model(self):
        """Test initialization with default model from config."""
        llm = AzureLLM(name="TestAzure", role=Role.PERSONA, system_prompt="Test prompt")

        assert llm.name == "TestAzure"
        assert llm.system_prompt == "Test prompt"
        assert llm.model_name == "gpt-5.2"
        assert llm.last_response_metadata == {}

    def test_init_with_custom_model(self):
        """Test initialization with custom model name instead of config default."""
        llm = AzureLLM(
            name="TestAzure", role=Role.PERSONA, model_name="azure-some-made-up-model"
        )

        assert llm.model_name == "some-made-up-model"  # azure- prefix should be removed

    def test_init_with_kwargs(self):
        """Test initialization with additional kwargs."""
        with patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock_model:
            AzureLLM(
                name="TestAzure",
                role=Role.PERSONA,
                temperature=0.5,
                max_tokens=500,
                top_p=0.9,
            )

            # Verify kwargs were passed to AzureAIChatCompletionsModel
            call_kwargs = mock_model.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 500
            assert call_kwargs["top_p"] == 0.9

    def test_init_with_api_version(self):
        """Test initialization with API version from config."""
        with (
            patch(
                "llm_clients.azure_llm.Config.AZURE_API_VERSION",
                "2024-05-01-preview",
            ),
            patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock_model,
        ):
            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert llm.api_version == "2024-05-01-preview"
            call_kwargs = mock_model.call_args[1]
            assert call_kwargs["api_version"] == "2024-05-01-preview"

    def test_init_with_default_api_version(self):
        """Test initialization with default API version when not configured."""
        with (
            patch("llm_clients.azure_llm.Config.AZURE_API_VERSION", None),
            patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock_model,
        ):
            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert llm.api_version == AzureLLM.DEFAULT_API_VERSION
            call_kwargs = mock_model.call_args[1]
            assert call_kwargs["api_version"] == AzureLLM.DEFAULT_API_VERSION

    def test_init_strips_endpoint_trailing_slash(self):
        """Test that endpoint trailing slash is removed."""
        with (
            patch(
                "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
                "https://test.openai.azure.com/",
            ),
            patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock_model,
        ):
            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert llm.endpoint == "https://test.openai.azure.com"
            call_kwargs = mock_model.call_args[1]
            assert call_kwargs["endpoint"] == "https://test.openai.azure.com"

    def test_init_adds_models_suffix_for_ai_foundry(self):
        """Test that /models suffix is added for Azure AI Foundry endpoints."""
        with (
            patch(
                "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
                "https://test.services.ai.azure.com",
            ),
            patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock_model,
        ):
            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert llm.endpoint == "https://test.services.ai.azure.com/models"
            call_kwargs = mock_model.call_args[1]
            assert (
                call_kwargs["endpoint"] == "https://test.services.ai.azure.com/models"
            )

    def test_init_does_not_duplicate_models_suffix(self):
        """Test that /models suffix is not duplicated if already present."""
        with (
            patch(
                "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
                "https://test.services.ai.azure.com/models",
            ),
            patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock_model,
        ):
            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert llm.endpoint == "https://test.services.ai.azure.com/models"
            call_kwargs = mock_model.call_args[1]
            assert (
                call_kwargs["endpoint"] == "https://test.services.ai.azure.com/models"
            )

    def test_init_invalid_endpoint_raises_error(self):
        """Test that non-HTTPS endpoint raises ValueError."""
        with (
            patch(
                "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
                "http://test.openai.azure.com",
            ),
            patch("llm_clients.azure_llm.AzureAIChatCompletionsModel"),
        ):
            with pytest.raises(ValueError) as exc_info:
                AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert "must start with 'https://'" in str(exc_info.value)

    def test_init_invalid_endpoint_pattern_raises_error(self):
        """Test that endpoint with unexpected pattern raises ValueError."""
        with (
            patch(
                "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
                "https://test.example.com",
            ),
            patch("llm_clients.azure_llm.AzureAIChatCompletionsModel"),
        ):
            with pytest.raises(ValueError) as exc_info:
                AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert "must match expected patterns" in str(exc_info.value)
            assert ".openai.azure.com" in str(exc_info.value)
            assert ".services.ai.azure.com" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_response_success_with_system_prompt(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test successful response generation with system prompt."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        mock_response = create_mock_response(
            text="This is an Azure response",
            response_id="chatcmpl-12345",
            token_usage={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
            finish_reason="stop",
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(
            name="TestAzure",
            role=Role.PERSONA,
            system_prompt="You are a helpful assistant.",
        )
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "This is an Azure response"

        # Verify metadata was extracted
        metadata = assert_metadata_structure(
            llm, expected_provider="azure", expected_role=Role.PERSONA
        )
        assert metadata["response_id"] == "chatcmpl-12345"
        assert metadata["model"] == "gpt-5.2"
        assert_iso_timestamp(metadata["timestamp"])
        assert_response_timing(metadata)
        assert metadata["usage"]["input_tokens"] == 10
        assert metadata["usage"]["output_tokens"] == 20
        assert metadata["usage"]["total_tokens"] == 30
        assert metadata["finish_reason"] == "stop"
        assert "raw_metadata" in metadata

    @pytest.mark.asyncio
    async def test_generate_response_without_system_prompt(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test response generation without system prompt."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        mock_response = create_mock_response(
            text="Response without system prompt", response_id="chatcmpl-67890"
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)  # No system prompt
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response without system prompt"

        # Verify ainvoke was called with only HumanMessage (no SystemMessage)
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].text == "Test"

    @pytest.mark.asyncio
    async def test_generate_response_without_usage_metadata(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test response when usage metadata is not available."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        # Response without usage in metadata
        mock_response = create_mock_response(
            text="Response", response_id="chatcmpl-abc"
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    async def test_generate_response_without_response_metadata(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test response when response_metadata attribute is missing."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        # Response without response_metadata attribute
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-xyz"
        del mock_response.response_metadata  # Remove attribute

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gpt-5.2"
        assert metadata["usage"] == {}
        assert metadata["finish_reason"] is None

    @pytest.mark.asyncio
    async def test_generate_response_api_error(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test error handling when API call fails."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        # Simulate API error
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API rate limit exceeded"))
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        # Should return error message instead of raising
        assert_error_response(response, "API rate limit exceeded")

        # Verify error metadata was stored
        assert_error_metadata(llm, "azure", "API rate limit exceeded")

    @pytest.mark.asyncio
    async def test_generate_response_404_error_with_helpful_message(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test that 404 errors provide helpful error messages."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        # Simulate 404 error with proper exception class
        class AzureError(Exception):
            def __init__(self, message, status_code=None):
                super().__init__(message)
                self.status_code = status_code
                self.response = MagicMock()
                if status_code:
                    self.response.url = "https://test.openai.azure.com/models/gpt-5.2"

        error = AzureError("404 Resource not found", status_code=404)
        mock_llm.ainvoke = AsyncMock(side_effect=error)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        # Should contain helpful error message
        assert "Error generating response" in response
        assert "404" in response or "Resource not found" in response
        assert "Model name" in response or "deployment name" in response

    @pytest.mark.asyncio
    async def test_generate_response_tracks_timing(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test that response timing is tracked correctly."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        mock_response = create_mock_response(
            text="Timed response", response_id="chatcmpl-time"
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert_response_timing(metadata)

    def test_get_last_response_metadata_returns_copy(self):
        """Test that get_last_response_metadata returns a copy."""
        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        assert_metadata_copy_behavior(llm)

    def test_set_system_prompt(self):
        """Test set_system_prompt method."""
        llm = AzureLLM(
            role=Role.PERSONA,
            model_name="azure-gpt-5.2",
            name="TestAzure",
            system_prompt="Initial prompt",
        )
        assert llm.system_prompt == "Initial prompt"

        llm.set_system_prompt("Updated prompt")
        assert llm.system_prompt == "Updated prompt"

    @pytest.mark.asyncio
    async def test_generate_structured_response_success(self, mock_llm_factory):
        """Test successful structured response generation."""
        with patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock_model:
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

            mock_model.return_value = mock_llm

            llm = AzureLLM(
                name="TestAzure", role=Role.PERSONA, system_prompt="Test prompt"
            )
            response = await llm.generate_structured_response(
                "What is the answer?", TestResponse
            )

            assert isinstance(response, TestResponse)
            assert response.answer == "Yes"
            assert response.reasoning == "Because it's correct"

            # Verify metadata was stored
            metadata = assert_metadata_structure(
                llm, expected_provider="azure", expected_role=Role.PERSONA
            )
            assert metadata["model"] == "gpt-5.2"
            assert metadata["structured_output"] is True
            assert_response_timing(metadata)

    @pytest.mark.asyncio
    async def test_generate_structured_response_error(self):
        """Test error handling in structured response generation."""
        with patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock_model:
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

            mock_model.return_value = mock_llm

            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            with pytest.raises(RuntimeError) as exc_info:
                await llm.generate_structured_response("Test", TestResponse)

            assert "Error generating structured response" in str(exc_info.value)
            assert "Structured output failed" in str(exc_info.value)

            # Verify error metadata was stored
            metadata = llm.get_last_response_metadata()
            assert "error" in metadata
            assert "Structured output failed" in metadata["error"]

    @pytest.mark.asyncio
    async def test_generate_response_with_conversation_history(self):
        """Test generate_response with conversation_history parameter."""
        with patch("llm_clients.azure_llm.AzureAIChatCompletionsModel") as mock_model:
            mock_llm = MagicMock()

            mock_response = create_mock_response(
                text="Response with history",
                response_id="chatcmpl-history",
                token_usage={
                    "input_tokens": 50,
                    "output_tokens": 20,
                },
            )

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_model.return_value = mock_llm

            llm = AzureLLM(name="TestAzure", role=Role.PERSONA, system_prompt="Test")

            # Provide conversation history
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
            ]

            response = await llm.generate_response(conversation_history=history)

            assert response == "Response with history"

            # Verify ainvoke was called with correct messages
            call_args = mock_llm.ainvoke.call_args
            messages = call_args[0][0]

            # Should have: SystemMessage + 2 history messages
            assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_timestamp_format(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test that timestamp is in ISO format."""
        mock_llm = MagicMock()

        mock_response = create_mock_response(text="Test", response_id="chatcmpl-ts")

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert_iso_timestamp(metadata["timestamp"])

    @pytest.mark.asyncio
    async def test_generate_response_with_persona_role_flips_types(
        self, mock_azure_config, mock_azure_model, sample_conversation_history
    ):
        """Test that persona role flips message types in conversation history."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        mock_response = create_mock_response(
            text="Persona response", response_id="chatcmpl-persona"
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        # Persona system prompt should trigger message type flipping
        persona_prompt = "You are roleplaying as a human user"
        llm = AzureLLM(
            name="TestAzure", system_prompt=persona_prompt, role=Role.PERSONA
        )

        response = await llm.generate_response(
            conversation_history=sample_conversation_history
        )

        assert response == "Persona response"

        # Verify message types are flipped for persona role
        verify_message_types_for_persona(mock_llm, expected_message_count=4)

    @pytest.mark.asyncio
    async def test_generate_response_with_partial_usage_metadata(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test response with incomplete usage metadata.

        Azure LLM gets total_tokens from metadata directly (doesn't calculate it).
        """
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        # Response with only input_tokens in usage
        # (missing output_tokens and total_tokens)
        mock_response = create_mock_response(
            text="Partial usage response",
            response_id="chatcmpl-partial",
            token_usage={"input_tokens": 15},  # Missing output_tokens, total_tokens
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "Partial usage response"
        metadata = llm.get_last_response_metadata()
        assert metadata["usage"]["input_tokens"] == 15
        assert metadata["usage"]["output_tokens"] == 0
        assert metadata["usage"]["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_metadata_includes_response_object(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test that metadata includes the full response object."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        mock_response = create_mock_response(text="Test", response_id="chatcmpl-obj")

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert "response" in metadata
        assert metadata["response"] == mock_response

    @pytest.mark.asyncio
    async def test_metadata_with_finish_reason(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test metadata extraction of finish_reason."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        mock_response = create_mock_response(
            text="Stopped response",
            response_id="chatcmpl-stop",
            finish_reason="length",
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert metadata["finish_reason"] == "length"

    @pytest.mark.asyncio
    async def test_raw_metadata_stored(
        self, mock_azure_config, mock_azure_model, mock_system_message
    ):
        """Test that raw metadata is stored."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-5.2"

        # Create response with custom metadata fields
        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.id = "chatcmpl-raw"
        mock_response.response_metadata = DictWithAttr(
            {
                "model": "gpt-5.2",
                "custom_field": "custom_value",
                "nested": {"key": "value"},
            }
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        metadata = llm.get_last_response_metadata()
        assert "raw_metadata" in metadata
        assert metadata["raw_metadata"]["custom_field"] == "custom_value"
        assert metadata["raw_metadata"]["nested"]["key"] == "value"
