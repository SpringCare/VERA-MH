"""Unit tests for ClaudeLLM class."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_clients import Role
from llm_clients.claude_llm import ClaudeLLM
from llm_clients.llm_interface import LLMGenerationFailed

from .test_base_llm import TestJudgeLLMBase
from .test_helpers import (
    assert_iso_timestamp,
    assert_llm_generation_failed,
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

    def test_init_passes_base_url_when_configured(self):
        """A configured base URL reaches ChatAnthropic."""
        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_chat_anthropic.return_value = MagicMock()
            with patch(
                "llm_clients.claude_llm.Config.get_base_url",
                return_value="https://gateway.example.com",
            ):
                ClaudeLLM(name="TestClaude", role=Role.PERSONA)

            call_kwargs = mock_chat_anthropic.call_args[1]
            assert call_kwargs["base_url"] == "https://gateway.example.com"

    def test_init_omits_base_url_when_not_configured(self):
        """Without an override, base_url is left out so ChatAnthropic defaults."""
        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_chat_anthropic.return_value = MagicMock()
            with patch("llm_clients.claude_llm.Config.get_base_url", return_value=None):
                ClaudeLLM(name="TestClaude", role=Role.PERSONA)

            assert "base_url" not in mock_chat_anthropic.call_args[1]

    def test_init_strips_sampling_params_for_opus_4_8(self, default_llm_kwargs):
        """Opus 4.8 rejects temperature/top_p/top_k; none may reach ChatAnthropic."""
        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-opus-4-8"
            mock_chat_anthropic.return_value = mock_llm

            ClaudeLLM(
                name="TestClaude",
                role=Role.JUDGE,
                model_name="claude-opus-4-8",
                **default_llm_kwargs,
            )

            call_kwargs = mock_chat_anthropic.call_args[1]
            assert "temperature" not in call_kwargs
            assert "top_p" not in call_kwargs
            assert call_kwargs["max_tokens"] == 500

    def test_init_strips_sampling_params_for_opus_4_7(self, default_llm_kwargs):
        """Opus 4.7 shares Opus 4.8's sampling-param and manual-thinking 400s."""
        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-opus-4-7"
            mock_chat_anthropic.return_value = mock_llm

            ClaudeLLM(
                name="TestClaude",
                role=Role.JUDGE,
                model_name="claude-opus-4-7",
                **default_llm_kwargs,
            )

            call_kwargs = mock_chat_anthropic.call_args[1]
            assert "temperature" not in call_kwargs
            assert "top_p" not in call_kwargs
            assert call_kwargs["max_tokens"] == 500

    def test_init_strips_sampling_params_for_fable_5(self, default_llm_kwargs):
        """Fable 5 shares the same sampling-param 400s as Opus 4.7/4.8."""
        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-fable-5"
            mock_chat_anthropic.return_value = mock_llm

            ClaudeLLM(
                name="TestClaude",
                role=Role.JUDGE,
                model_name="claude-fable-5",
                **default_llm_kwargs,
            )

            call_kwargs = mock_chat_anthropic.call_args[1]
            assert "temperature" not in call_kwargs
            assert "top_p" not in call_kwargs
            assert call_kwargs["max_tokens"] == 500

    def test_model_supports_param_unsupported_for_opus_4_8(self, default_llm_kwargs):
        with patch("llm_clients.claude_llm.ChatAnthropic"):
            llm = ClaudeLLM(
                name="TestClaude",
                role=Role.JUDGE,
                model_name="claude-opus-4-8",
                **default_llm_kwargs,
            )
        assert not llm._model_supports_param("claude-opus-4-8", "temperature")
        assert not llm._model_supports_param("claude-opus-4-8", "top_p")
        assert not llm._model_supports_param("claude-opus-4-8", "top_k")

    def test_model_supports_param_case_insensitive(self, default_llm_kwargs):
        with patch("llm_clients.claude_llm.ChatAnthropic"):
            llm = ClaudeLLM(
                name="TestClaude",
                role=Role.JUDGE,
                model_name="claude-opus-4-8",
                **default_llm_kwargs,
            )
        assert not llm._model_supports_param("Claude-Opus-4-8", "Temperature")

    def test_filter_supported_params_strips_sampling_params_for_opus_4_8(
        self, default_llm_kwargs
    ):
        with patch("llm_clients.claude_llm.ChatAnthropic"):
            llm = ClaudeLLM(
                name="TestClaude",
                role=Role.JUDGE,
                model_name="claude-opus-4-8",
                **default_llm_kwargs,
            )
        params = {"temperature": 0, "max_tokens": 500, "top_p": 0.9, "top_k": 40}
        filtered = llm._filter_supported_params("claude-opus-4-8", params)
        assert filtered == {"max_tokens": 500}

    # ============================================================================
    # Reasoning / extended-thinking effort translation
    # ============================================================================

    @pytest.mark.parametrize(
        "model_name,expected",
        [
            ("claude-sonnet-5", True),
            ("claude-opus-4-8", True),
            ("claude-opus-4-7", True),
            ("claude-fable-5", True),
            ("claude-haiku-4-5", False),
            ("claude-sonnet-4-5-20250929", False),
            (None, False),
        ],
    )
    def test_is_adaptive_thinking_model(self, model_name, expected):
        assert ClaudeLLM._is_adaptive_thinking_model(model_name) is expected

    def test_apply_thinking_kwargs_no_effort_sonnet_5_omits_thinking(self):
        """No effort means `thinking` is omitted; the model's own default applies."""
        kwargs: dict = {}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-sonnet-5", None)
        assert kwargs == {"max_tokens": 8192}

    def test_apply_thinking_kwargs_no_effort_sonnet_5_keeps_explicit_thinking(self):
        """A caller-supplied `thinking` kwarg must survive untouched."""
        kwargs: dict = {"thinking": {"type": "enabled", "budget_tokens": 5000}}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-sonnet-5", None)
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}

    def test_apply_thinking_kwargs_no_effort_opus_4_8_is_untouched(self):
        kwargs: dict = {}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-opus-4-8", None)
        assert kwargs == {}

    def test_apply_thinking_kwargs_no_effort_opus_5_sets_max_tokens(self):
        """opus-5 also lacks a langchain-anthropic model profile, like sonnet-5."""
        assert ClaudeLLM._is_adaptive_thinking_model("claude-opus-5")
        kwargs: dict = {}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-opus-5", None)
        assert kwargs == {"max_tokens": 8192}

    def test_apply_thinking_kwargs_no_effort_fable_5_is_untouched(self):
        """fable-5 can't disable thinking at all; omitting `thinking` already
        gives the (only) adaptive-always-on behavior, so nothing is forced."""
        kwargs: dict = {}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-fable-5", None)
        assert kwargs == {}

    def test_apply_thinking_kwargs_no_effort_non_adaptive_model_is_untouched(self):
        kwargs: dict = {}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-haiku-4-5", None)
        assert kwargs == {}

    def test_apply_thinking_kwargs_effort_on_sonnet_5_uses_adaptive_and_effort(self):
        kwargs: dict = {}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-sonnet-5", "high")
        assert kwargs == {
            "thinking": {"type": "adaptive"},
            "effort": "high",
            "max_tokens": 8192,
        }

    def test_apply_thinking_kwargs_effort_on_opus_4_8_uses_adaptive_and_effort(self):
        kwargs: dict = {}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-opus-4-8", "high")
        assert kwargs == {"thinking": {"type": "adaptive"}, "effort": "high"}

    def test_apply_thinking_kwargs_effort_on_fable_5_uses_adaptive_and_effort(self):
        kwargs: dict = {}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-fable-5", "xhigh")
        assert kwargs == {"thinking": {"type": "adaptive"}, "effort": "xhigh"}

    def test_apply_thinking_kwargs_effort_on_haiku_uses_budget_tokens(self):
        kwargs: dict = {}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-haiku-4-5", "high")
        assert kwargs == {
            "thinking": {"type": "enabled", "budget_tokens": 16000},
            "max_tokens": 17024,
        }

    def test_apply_thinking_kwargs_unknown_effort_falls_back_to_medium_budget(self):
        kwargs: dict = {}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-haiku-4-5", "not-a-real-level")
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}
        assert kwargs["max_tokens"] == 6024

    def test_apply_thinking_kwargs_does_not_clobber_explicit_effort_or_max_tokens(self):
        """setdefault semantics: caller-supplied overrides win."""
        kwargs = {"effort": "low", "max_tokens": 100}
        ClaudeLLM._apply_thinking_kwargs(kwargs, "claude-sonnet-5", "high")
        assert kwargs["effort"] == "low"
        assert kwargs["max_tokens"] == 100
        assert kwargs["thinking"] == {"type": "adaptive"}

    def test_init_with_thinking_effort_forces_temperature_1_then_stripped_for_sonnet_5(
        self,
    ):
        """Thinking forces temperature=1, but sonnet-5 rejects temperature outright."""
        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-sonnet-5"
            mock_chat_anthropic.return_value = mock_llm

            ClaudeLLM(
                name="TestClaude",
                role=Role.JUDGE,
                model_name="claude-sonnet-5",
                thinking_effort="high",
            )

            call_kwargs = mock_chat_anthropic.call_args[1]
            assert call_kwargs["thinking"] == {"type": "adaptive"}
            assert call_kwargs["effort"] == "high"
            assert "temperature" not in call_kwargs

    def test_init_with_thinking_effort_forces_temperature_1_for_haiku(self):
        """Haiku supports temperature, so the forced temperature=1 survives."""
        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-haiku-4-5"
            mock_chat_anthropic.return_value = mock_llm

            ClaudeLLM(
                name="TestClaude",
                role=Role.JUDGE,
                model_name="claude-haiku-4-5",
                thinking_effort="high",
            )

            call_kwargs = mock_chat_anthropic.call_args[1]
            assert call_kwargs["thinking"] == {
                "type": "enabled",
                "budget_tokens": 16000,
            }
            assert call_kwargs["temperature"] == 1

    def test_unsupported_model_params_covers_all_adaptive_markers(self):
        with patch("llm_clients.claude_llm.ChatAnthropic"):
            llm = ClaudeLLM(name="TestClaude", role=Role.JUDGE)
        unsupported = llm._unsupported_model_params()
        assert unsupported == {
            "opus-4-7": frozenset({"temperature", "top_p", "top_k"}),
            "opus-4-8": frozenset({"temperature", "top_p", "top_k"}),
            "opus-5": frozenset({"temperature", "top_p", "top_k"}),
            "fable-5": frozenset({"temperature", "top_p", "top_k"}),
            "sonnet-5": frozenset({"temperature", "top_p", "top_k"}),
        }

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
    async def test_generate_response_propagates_cache_usage_tokens(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Copy cache_creation_input_tokens and cache_read_input_tokens into usage."""
        mock_response = mock_response_factory(
            text="Cached context response",
            response_id="msg_cache",
            provider="claude",
            metadata={
                "model": "claude-sonnet-4-5-20250929",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 200,
                },
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
        await llm.generate_response(conversation_history=mock_system_message)

        usage = llm.last_response_metadata["usage"]
        assert usage["cache_creation_input_tokens"] == 100
        assert usage["cache_read_input_tokens"] == 200
        assert usage["total_tokens"] == 30

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_caching_false_omits_cache_control_on_invoke(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """When caching=False, ainvoke receives no cache_control kwarg."""
        mock_response = mock_response_factory(
            text="No cache",
            response_id="msg_nocache",
            provider="claude",
            metadata={
                "model": "claude-sonnet-4-5-20250929",
                "usage": {"input_tokens": 1, "output_tokens": 2},
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
            caching=False,
        )
        await llm.generate_response(conversation_history=mock_system_message)

        mock_llm.ainvoke.assert_called_once()
        assert "cache_control" not in mock_llm.ainvoke.call_args.kwargs

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_omits_none_cache_usage_keys(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """None cache token values are not copied into last_response_metadata usage."""
        mock_response = mock_response_factory(
            text="Response",
            response_id="msg_cache_none",
            provider="claude",
            metadata={
                "model": "claude-sonnet-4-5-20250929",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": None,
                    "cache_read_input_tokens": None,
                },
            },
        )

        mock_llm = MagicMock()
        mock_llm.model = "claude-sonnet-4-5-20250929"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA)
        await llm.generate_response(conversation_history=mock_system_message)

        usage = llm.last_response_metadata["usage"]
        assert "cache_creation_input_tokens" not in usage
        assert "cache_read_input_tokens" not in usage

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
        metadata = llm.last_response_metadata
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
        metadata = llm.last_response_metadata
        assert metadata["model"] == "claude-sonnet-4-5-20250929"
        assert metadata["usage"] == {}
        assert metadata["stop_reason"] is None

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_model_name_update_from_metadata(
        self, mock_chat_anthropic, mock_response_factory, mock_system_message
    ):
        """Test that model name is updated from dict response metadata."""
        mock_response = mock_response_factory(
            text="Test",
            response_id="msg-model",
            provider="claude",
            metadata={"model": "claude-3-opus-20240229"},
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(
            name="TestClaude",
            role=Role.PERSONA,
            model_name="claude-sonnet-4-5-20250929",
        )
        await llm.generate_response(conversation_history=mock_system_message)

        assert llm.last_response_metadata["model"] == "claude-3-opus-20240229"

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
        with pytest.raises(LLMGenerationFailed) as exc_info:
            await llm.generate_response(conversation_history=mock_system_message)

        assert_llm_generation_failed(
            exc_info.value,
            "API rate limit exceeded",
            mock_ainvoke=mock_llm.ainvoke,
        )

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

        metadata = llm.last_response_metadata
        assert_response_timing(metadata)

    def test_last_response_metadata_copy_returns_copy(self):
        """Test that last_response_metadata.copy() returns a copy, not the original."""
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
        metadata = llm.last_response_metadata
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

        metadata = llm.last_response_metadata
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

        metadata = llm.last_response_metadata
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

        metadata = llm.last_response_metadata
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

        metadata = llm.last_response_metadata
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
        """Test start_conversation with empty history uses default start_prompt."""
        from llm_clients.llm_interface import DEFAULT_START_PROMPT

        mock_response = mock_response_factory(
            text="Response", response_id="msg_empty", provider="claude"
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA, system_prompt="Test")

        response = await llm.start_conversation()

        assert response == "Response"

        # Empty history: SystemMessage + HumanMessage(default start_prompt)
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0].text == "Test"
        assert messages[1].content == DEFAULT_START_PROMPT

    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.claude_llm.ChatAnthropic")
    async def test_generate_response_with_none_conversation_history(
        self, mock_chat_anthropic, mock_response_factory
    ):
        """Test generate_response with None
        delegates to start_conversation (default start_prompt).
        """
        from llm_clients.llm_interface import DEFAULT_START_PROMPT

        mock_response = mock_response_factory(
            text="Response", response_id="msg_none", provider="claude"
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat_anthropic.return_value = mock_llm

        llm = ClaudeLLM(name="TestClaude", role=Role.PERSONA, system_prompt="Test")

        # None history delegates to start_conversation()
        response = await llm.generate_response(conversation_history=None)

        assert response == "Response"

        # None history: SystemMessage + HumanMessage(default start_prompt)
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0].text == "Test"
        assert messages[1].content == DEFAULT_START_PROMPT

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

            # json_schema is Anthropic's native structured-output mode, and
            # (unlike forced tool calling) works whether or not thinking is
            # enabled. Regression guard against reverting to the default
            # method="function_calling".
            mock_llm.with_structured_output.assert_called_once_with(
                TestResponse, method="json_schema", include_raw=True
            )

    @pytest.mark.asyncio
    async def test_generate_structured_response_unpacks_include_raw_payload(self):
        """Exercise the real `include_raw=True` shape, not a bare instance."""
        from pydantic import BaseModel, Field

        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-sonnet-4-5-20250929"

            class TestResponse(BaseModel):
                answer: str = Field(description="The answer")

            test_response = TestResponse(answer="Yes")
            raw_message = MagicMock()
            raw_message.response_metadata = {"stop_reason": "end_turn"}

            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(
                return_value={
                    "parsed": test_response,
                    "raw": raw_message,
                    "parsing_error": None,
                }
            )
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )
            mock_chat_anthropic.return_value = mock_llm

            llm = ClaudeLLM(name="TestClaude", role=Role.JUDGE)
            response = await llm.generate_structured_response("Test", TestResponse)

            assert response == test_response

    @pytest.mark.asyncio
    async def test_generate_structured_response_raises_with_diagnostics_on_none_parsed(
        self,
    ):
        """When parsing fails, the error message must surface raw/stop_reason info."""
        from pydantic import BaseModel

        with patch("llm_clients.claude_llm.ChatAnthropic") as mock_chat_anthropic:
            mock_llm = MagicMock()
            mock_llm.model = "claude-sonnet-4-5-20250929"

            class TestResponse(BaseModel):
                answer: str

            raw_message = MagicMock()
            raw_message.content = "not valid json"
            raw_message.response_metadata = {"stop_reason": "max_tokens"}

            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(
                return_value={
                    "parsed": None,
                    "raw": raw_message,
                    "parsing_error": ValueError("could not parse"),
                }
            )
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )
            mock_chat_anthropic.return_value = mock_llm

            llm = ClaudeLLM(name="TestClaude", role=Role.JUDGE)

            with pytest.raises(LLMGenerationFailed) as exc_info:
                await llm.generate_structured_response("Test", TestResponse)

            message = str(exc_info.value)
            assert "stop_reason='max_tokens'" in message
            assert "raw_content='not valid json'" in message

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

            with pytest.raises(LLMGenerationFailed) as exc_info:
                await llm.generate_structured_response("Test", TestResponse)

            assert_llm_generation_failed(
                exc_info.value,
                "Structured output failed",
                mock_ainvoke=mock_structured_llm.ainvoke,
            )

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

            metadata = llm.last_response_metadata

            # Verify required fields
            assert metadata["provider"] == "claude"
            assert metadata["structured_output"] is True
            assert metadata["response_id"] is None
            assert_iso_timestamp(metadata["timestamp"])
            assert_response_timing(metadata)
