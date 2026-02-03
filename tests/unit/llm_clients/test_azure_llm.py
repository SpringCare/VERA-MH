from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from llm_clients import Role
from llm_clients.azure_llm import AzureLLM


# Helper class for mocking response_metadata that supports both dict and
# attribute access
class DictWithAttr(dict):
    """Dict that supports both dict operations and attribute access."""

    def __getattr__(self, key):
        return self.get(key)


@pytest.fixture
def mock_azure_config():
    """Fixture to patch Azure config values."""
    with (
        patch("llm_clients.azure_llm.Config.AZURE_API_KEY", "test-key"),
        patch(
            "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
            "https://test.openai.azure.com",
        ),
        patch(
            "llm_clients.azure_llm.Config.get_azure_config",
            return_value={"model": "gpt-4"},
        ),
    ):
        yield


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
    mock_response.response_metadata = DictWithAttr({"model": "gpt-4", **metadata})
    return mock_response


@pytest.mark.unit
class TestAzureLLM:
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

    def test_init_with_default_model(self, mock_azure_config, mock_azure_model):
        """Test initialization with default model from config."""
        llm = AzureLLM(name="TestAzure", role=Role.PERSONA, system_prompt="Test prompt")

        assert llm.name == "TestAzure"
        assert llm.system_prompt == "Test prompt"
        assert llm.model_name == "gpt-4"
        assert llm.last_response_metadata == {}

    def test_init_with_custom_model(self, mock_azure_config, mock_azure_model):
        """Test initialization with custom model name instead of config default."""
        llm = AzureLLM(
            name="TestAzure", role=Role.PERSONA, model_name="azure-some-made-up-model"
        )

        assert llm.model_name == "some-made-up-model"  # azure- prefix should be removed

    def test_init_with_kwargs(self, mock_azure_config, mock_azure_model):
        """Test initialization with additional kwargs."""
        AzureLLM(
            name="TestAzure",
            role=Role.PERSONA,
            temperature=0.5,
            max_tokens=500,
            top_p=0.9,
        )

        # Verify kwargs were passed to AzureAIChatCompletionsModel
        call_kwargs = mock_azure_model.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.9

    def test_init_with_api_version(self, mock_azure_config, mock_azure_model):
        """Test initialization with API version from config."""
        with patch(
            "llm_clients.azure_llm.Config.AZURE_API_VERSION", "2024-05-01-preview"
        ):
            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert llm.api_version == "2024-05-01-preview"
            call_kwargs = mock_azure_model.call_args[1]
            assert call_kwargs["api_version"] == "2024-05-01-preview"

    def test_init_with_default_api_version(self, mock_azure_config, mock_azure_model):
        """Test initialization with default API version when not configured."""
        with patch("llm_clients.azure_llm.Config.AZURE_API_VERSION", None):
            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert llm.api_version == AzureLLM.DEFAULT_API_VERSION
            call_kwargs = mock_azure_model.call_args[1]
            assert call_kwargs["api_version"] == AzureLLM.DEFAULT_API_VERSION

    def test_init_strips_endpoint_trailing_slash(
        self, mock_azure_config, mock_azure_model
    ):
        """Test that endpoint trailing slash is removed."""
        with patch(
            "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
            "https://test.openai.azure.com/",
        ):
            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert llm.endpoint == "https://test.openai.azure.com"
            call_kwargs = mock_azure_model.call_args[1]
            assert call_kwargs["endpoint"] == "https://test.openai.azure.com"

    def test_init_adds_models_suffix_for_ai_foundry(
        self, mock_azure_config, mock_azure_model
    ):
        """Test that /models suffix is added for Azure AI Foundry endpoints."""
        with patch(
            "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
            "https://test.services.ai.azure.com",
        ):
            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert llm.endpoint == "https://test.services.ai.azure.com/models"
            call_kwargs = mock_azure_model.call_args[1]
            assert (
                call_kwargs["endpoint"] == "https://test.services.ai.azure.com/models"
            )

    def test_init_does_not_duplicate_models_suffix(
        self, mock_azure_config, mock_azure_model
    ):
        """Test that /models suffix is not duplicated if already present."""
        with patch(
            "llm_clients.azure_llm.Config.AZURE_ENDPOINT",
            "https://test.services.ai.azure.com/models",
        ):
            llm = AzureLLM(name="TestAzure", role=Role.PERSONA)

            assert llm.endpoint == "https://test.services.ai.azure.com/models"
            call_kwargs = mock_azure_model.call_args[1]
            assert (
                call_kwargs["endpoint"] == "https://test.services.ai.azure.com/models"
            )

    def test_init_invalid_endpoint_raises_error(self, mock_azure_config):
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

    def test_init_invalid_endpoint_pattern_raises_error(self, mock_azure_config):
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
        self, mock_azure_config, mock_azure_model
    ):
        """Test successful response generation with system prompt."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

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
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Hello, Azure!"}]
        )

        assert response == "This is an Azure response"

        # Verify metadata was extracted
        metadata = llm.get_last_response_metadata()
        assert metadata["response_id"] == "chatcmpl-12345"
        assert metadata["model"] == "gpt-4"
        assert metadata["provider"] == "azure"
        assert "timestamp" in metadata
        assert "response_time_seconds" in metadata
        assert metadata["usage"]["input_tokens"] == 10
        assert metadata["usage"]["output_tokens"] == 20
        assert metadata["usage"]["total_tokens"] == 30
        assert metadata["finish_reason"] == "stop"
        assert "raw_metadata" in metadata

    @pytest.mark.asyncio
    async def test_generate_response_without_system_prompt(
        self, mock_azure_config, mock_azure_model
    ):
        """Test response generation without system prompt."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        mock_response = create_mock_response(
            text="Response without system prompt", response_id="chatcmpl-67890"
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)  # No system prompt
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test message"}]
        )

        assert response == "Response without system prompt"

        # Verify ainvoke was called with only HumanMessage (no SystemMessage)
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].text == "Test message"

    @pytest.mark.asyncio
    async def test_generate_response_without_usage_metadata(
        self, mock_azure_config, mock_azure_model
    ):
        """Test response when usage metadata is not available."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Response without usage in metadata
        mock_response = create_mock_response(
            text="Response", response_id="chatcmpl-abc"
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test"}]
        )

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    async def test_generate_response_without_response_metadata(
        self, mock_azure_config, mock_azure_model
    ):
        """Test response when response_metadata attribute is missing."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Response without response_metadata attribute
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.id = "chatcmpl-xyz"
        del mock_response.response_metadata  # Remove attribute

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test"}]
        )

        assert response == "Response"
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gpt-4"
        assert metadata["usage"] == {}
        assert metadata["finish_reason"] is None

    @pytest.mark.asyncio
    async def test_generate_response_api_error(
        self, mock_azure_config, mock_azure_model
    ):
        """Test error handling when API call fails."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Simulate API error
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API rate limit exceeded"))
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
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
        assert metadata["provider"] == "azure"
        assert "timestamp" in metadata
        assert "error" in metadata
        assert "API rate limit exceeded" in metadata["error"]
        assert metadata["usage"] == {}

    @pytest.mark.asyncio
    async def test_generate_response_404_error_with_helpful_message(
        self, mock_azure_config, mock_azure_model
    ):
        """Test that 404 errors provide helpful error messages."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Simulate 404 error with proper exception class
        class AzureError(Exception):
            def __init__(self, message, status_code=None):
                super().__init__(message)
                self.status_code = status_code
                self.response = MagicMock()
                if status_code:
                    self.response.url = "https://test.openai.azure.com/models/gpt-4"

        error = AzureError("404 Resource not found", status_code=404)
        mock_llm.ainvoke = AsyncMock(side_effect=error)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        response = await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test message"}]
        )

        # Should contain helpful error message
        assert "Error generating response" in response
        assert "404" in response or "Resource not found" in response
        assert "Model name" in response or "deployment name" in response

    @pytest.mark.asyncio
    async def test_generate_response_tracks_timing(
        self, mock_azure_config, mock_azure_model
    ):
        """Test that response timing is tracked correctly."""
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        mock_response = create_mock_response(
            text="Timed response", response_id="chatcmpl-time"
        )

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        await llm.generate_response(
            conversation_history=[{"turn": 0, "response": "Test"}]
        )

        metadata = llm.get_last_response_metadata()
        assert "response_time_seconds" in metadata
        assert isinstance(metadata["response_time_seconds"], (int, float))
        assert metadata["response_time_seconds"] >= 0

    def test_get_last_response_metadata_returns_copy(
        self, mock_azure_config, mock_azure_model
    ):
        """Test that get_last_response_metadata returns a copy."""
        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
        llm.last_response_metadata = {"test": "value"}

        metadata1 = llm.get_last_response_metadata()
        metadata2 = llm.get_last_response_metadata()

        # Should be equal but not the same object
        assert metadata1 == metadata2
        assert metadata1 is not metadata2

        # Modifying returned copy shouldn't affect internal state
        metadata1["modified"] = True
        assert "modified" not in llm.last_response_metadata

    def test_set_system_prompt(self, mock_azure_config, mock_azure_model):
        """Test set_system_prompt method."""
        llm = AzureLLM(
            role=Role.PERSONA,
            model_name="azure-gpt-4",
            name="TestAzure",
            system_prompt="Initial prompt",
        )
        assert llm.system_prompt == "Initial prompt"

        llm.set_system_prompt("Updated prompt")
        assert llm.system_prompt == "Updated prompt"

    @pytest.mark.asyncio
    async def test_generate_structured_response_success(
        self, mock_azure_config, mock_azure_model
    ):
        """Test successful structured response generation."""
        mock_llm = MagicMock()

        # Create a test Pydantic model
        class TestResponse(BaseModel):
            answer: str = Field(description="The answer")
            reasoning: str = Field(description="The reasoning")

        # Mock structured LLM
        mock_structured_llm = MagicMock()
        test_response = TestResponse(answer="Yes", reasoning="Because it's correct")
        mock_structured_llm.ainvoke = AsyncMock(return_value=test_response)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA, system_prompt="Test prompt")
        response = await llm.generate_structured_response(
            "What is the answer?", TestResponse
        )

        assert isinstance(response, TestResponse)
        assert response.answer == "Yes"
        assert response.reasoning == "Because it's correct"

        # Verify metadata was stored
        metadata = llm.get_last_response_metadata()
        assert metadata["model"] == "gpt-4"
        assert metadata["provider"] == "azure"
        assert metadata["structured_output"] is True
        assert "timestamp" in metadata
        assert "response_time_seconds" in metadata

    @pytest.mark.asyncio
    async def test_generate_structured_response_error(
        self, mock_azure_config, mock_azure_model
    ):
        """Test error handling in structured response generation."""
        mock_llm = MagicMock()

        class TestResponse(BaseModel):
            answer: str

        # Mock structured LLM to raise error
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(
            side_effect=Exception("Structured output failed")
        )
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        mock_azure_model.return_value = mock_llm

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
    async def test_generate_response_with_conversation_history(
        self, mock_azure_config, mock_azure_model
    ):
        """Test generate_response with conversation_history parameter."""
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
        mock_azure_model.return_value = mock_llm

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
    async def test_timestamp_format(self, mock_azure_config, mock_azure_model):
        """Test that timestamp is in ISO format."""
        mock_llm = MagicMock()

        mock_response = create_mock_response(text="Test", response_id="chatcmpl-ts")

        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_azure_model.return_value = mock_llm

        llm = AzureLLM(name="TestAzure", role=Role.PERSONA)
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
