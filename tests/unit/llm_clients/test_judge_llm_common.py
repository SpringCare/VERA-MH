"""Parameterized tests for common JudgeLLM functionality.

This module tests common functionality across all JudgeLLM implementations
(ClaudeLLM, OpenAILLM, GeminiLLM) to reduce code duplication.

It also includes a test that ensures all JudgeLLM implementations are tested.
"""

from datetime import datetime
from typing import Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from llm_clients.claude_llm import ClaudeLLM
from llm_clients.gemini_llm import GeminiLLM
from llm_clients.llm_interface import JudgeLLM
from llm_clients.openai_llm import OpenAILLM

# Configuration for each JudgeLLM implementation
JUDGE_LLM_CONFIGS = [
    {
        "class": ClaudeLLM,
        "name": "ClaudeLLM",
        "model_name": "claude-3-5-sonnet-20241022",
        "default_model": "claude-3-5-sonnet-20241022",
        "api_key_config": "llm_clients.claude_llm.Config.ANTHROPIC_API_KEY",
        "api_key_name": "ANTHROPIC_API_KEY",
        "langchain_class": "llm_clients.claude_llm.ChatAnthropic",
        "provider": "claude",
        "response_id": "msg_12345",
        "mock_response_metadata": {
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "stop_reason": "end_turn",
        },
    },
    {
        "class": OpenAILLM,
        "name": "OpenAILLM",
        "model_name": "gpt-4",
        "default_model": "gpt-4",
        "api_key_config": "llm_clients.openai_llm.Config.OPENAI_API_KEY",
        "api_key_name": "OPENAI_API_KEY",
        "langchain_class": "llm_clients.openai_llm.ChatOpenAI",
        "provider": "openai",
        "response_id": "chatcmpl-12345",
        "mock_response_metadata": {
            "model_name": "gpt-4-0613",
            "token_usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
            "finish_reason": "stop",
        },
    },
    {
        "class": GeminiLLM,
        "name": "GeminiLLM",
        "model_name": "gemini-1.5-pro",
        "default_model": "gemini-1.5-pro",
        "api_key_config": "llm_clients.gemini_llm.Config.GOOGLE_API_KEY",
        "api_key_name": "GOOGLE_API_KEY",
        "langchain_class": "llm_clients.gemini_llm.ChatGoogleGenerativeAI",
        "provider": "gemini",
        "response_id": "gemini-12345",
        "mock_response_metadata": {
            "model_name": "gemini-1.5-pro-001",
            "usage_metadata": {
                "prompt_token_count": 12,
                "candidates_token_count": 28,
                "total_token_count": 40,
            },
            "finish_reason": "STOP",
        },
    },
]


def get_all_judge_llm_classes() -> list[Type[JudgeLLM]]:
    """Discover all JudgeLLM subclasses in the codebase."""
    judge_llm_classes = []

    # Get all subclasses of JudgeLLM
    for subclass in JudgeLLM.__subclasses__():
        # Exclude test utilities (MockLLM)
        if subclass.__name__ != "MockLLM":
            judge_llm_classes.append(subclass)

    return list(set(judge_llm_classes))  # Remove duplicates


@pytest.mark.unit
class TestJudgeLLMCoverage:
    """Test that ensures all JudgeLLM implementations are tested."""

    def test_all_judge_llm_classes_are_tested(self):
        """Test that fails if a new JudgeLLM is created but not tested."""
        all_judge_llm_classes = get_all_judge_llm_classes()
        tested_classes = {config["class"] for config in JUDGE_LLM_CONFIGS}

        missing_classes = set(all_judge_llm_classes) - tested_classes

        if missing_classes:
            missing_names = [cls.__name__ for cls in missing_classes]
            pytest.fail(
                f"Found {len(missing_classes)} JudgeLLM implementation(s) that are not "
                f"tested in test_judge_llm_common.py: {', '.join(missing_names)}. "
                f"Please add them to JUDGE_LLM_CONFIGS."
            )


@pytest.mark.unit
class TestJudgeLLMCommon:
    """Parameterized tests for common JudgeLLM functionality."""

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    def test_init_missing_api_key_raises_error(self, config):
        """Test that missing API key raises ValueError."""
        with patch(config["api_key_config"], None):
            with pytest.raises(ValueError) as exc_info:
                config["class"](name=f"Test{config['name']}")

            assert config["api_key_name"] in str(exc_info.value)

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    def test_init_with_default_model(self, config):
        """Test initialization with default model from config."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]
            mock_langchain.return_value = mock_llm

            llm = config["class"](
                name=f"Test{config['name']}", system_prompt="Test prompt"
            )

            assert llm.name == f"Test{config['name']}"
            assert llm.system_prompt == "Test prompt"
            assert llm.model_name == config["default_model"]
            assert llm.last_response_metadata == {}

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    def test_init_with_custom_model(self, config):
        """Test initialization with custom model name."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["model_name"]
            mock_langchain.return_value = mock_llm

            llm = config["class"](
                name=f"Test{config['name']}", model_name=config["model_name"]
            )

            assert llm.model_name == config["model_name"]

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    def test_init_with_kwargs(self, config):
        """Test initialization with additional kwargs."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]
            mock_langchain.return_value = mock_llm

            config["class"](
                name=f"Test{config['name']}",
                temperature=0.5,
                max_tokens=500,
                top_p=0.9,
            )

            # Verify kwargs were passed to LangChain class
            call_kwargs = mock_langchain.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 500
            assert call_kwargs["top_p"] == 0.9

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_response_success_with_system_prompt(self, config):
        """Test successful response generation with system prompt."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            # Create mock response with metadata
            mock_response = MagicMock()
            mock_response.text = "This is a test response"
            mock_response.id = config["response_id"]

            # Handle different metadata structures
            if config["provider"] == "openai":
                mock_response.response_metadata = config["mock_response_metadata"]
                mock_response.additional_kwargs = {}
                mock_response.usage_metadata = {
                    "input_tokens": 15,
                    "output_tokens": 25,
                    "total_tokens": 40,
                }
            elif config["provider"] == "gemini":
                # Gemini uses a special metadata object
                mock_metadata_obj = MagicMock()
                mock_metadata_obj.model_name = config["mock_response_metadata"][
                    "model_name"
                ]
                mock_metadata_obj.__getitem__ = lambda self, key: config[
                    "mock_response_metadata"
                ].get(key)
                mock_metadata_obj.__contains__ = (
                    lambda self, key: key in config["mock_response_metadata"]
                )
                mock_metadata_obj.get = lambda key, default=None: config[
                    "mock_response_metadata"
                ].get(key, default)
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = config["mock_response_metadata"]

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](
                name=f"Test{config['name']}",
                system_prompt="You are a helpful assistant.",
            )
            response = await llm.generate_response(
                conversation_history=[
                    {"turn": 0, "speaker": "system", "response": "Hello!"}
                ]
            )

            assert response == "This is a test response"

            # Verify metadata was extracted
            metadata = llm.get_last_response_metadata()
            assert metadata["response_id"] == config["response_id"]
            assert metadata["provider"] == config["provider"]
            assert "timestamp" in metadata
            assert "response_time_seconds" in metadata

            # Verify provider-specific metadata fields
            if config["provider"] == "openai":
                assert "additional_kwargs" in metadata
                assert "system_fingerprint" in metadata
                assert "logprobs" in metadata
                assert "raw_response_metadata" in metadata
                assert "raw_usage_metadata" in metadata
                assert metadata["finish_reason"] == "stop"
            elif config["provider"] == "gemini":
                assert "finish_reason" in metadata
                assert metadata["finish_reason"] == "STOP"
                assert "raw_metadata" in metadata
            else:  # claude
                assert "stop_reason" in metadata
                assert metadata["stop_reason"] == "end_turn"
                assert "raw_metadata" in metadata

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_response_without_system_prompt(self, config):
        """Test response generation without system prompt."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_response = MagicMock()
            mock_response.text = "Response without system prompt"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                mock_response.response_metadata = {
                    "model_name": config["default_model"]
                }
            elif config["provider"] == "gemini":
                mock_metadata_obj = MagicMock()
                mock_metadata_obj.model_name = config["default_model"]
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {"model": config["default_model"]}

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")  # No system prompt
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

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_response_api_error(self, config):
        """Test error handling when API call fails."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            # Simulate API error
            mock_llm.ainvoke = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            response = await llm.generate_response(
                conversation_history=[
                    {"turn": 0, "speaker": "system", "response": "Test message"}
                ]
            )

            # Should return error message instead of raising
            assert "Error generating response" in response
            assert "API rate limit exceeded" in response

            # Verify error metadata was stored
            metadata = llm.get_last_response_metadata()
            assert metadata["response_id"] is None
            assert metadata["model"] == config["default_model"]
            assert metadata["provider"] == config["provider"]
            assert "timestamp" in metadata
            assert "error" in metadata
            assert "API rate limit exceeded" in metadata["error"]
            assert metadata["usage"] == {}

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_response_tracks_timing(self, config):
        """Test that response timing is tracked correctly."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_response = MagicMock()
            mock_response.text = "Timed response"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                mock_response.response_metadata = {}
            elif config["provider"] == "gemini":
                mock_metadata_obj = MagicMock()
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {}

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            await llm.generate_response(
                conversation_history=[
                    {"turn": 0, "speaker": "system", "response": "Test"}
                ]
            )

            metadata = llm.get_last_response_metadata()
            assert "response_time_seconds" in metadata
            assert isinstance(metadata["response_time_seconds"], (int, float))
            assert metadata["response_time_seconds"] >= 0

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    def test_get_last_response_metadata_returns_copy(self, config):
        """Test that get_last_response_metadata returns a copy."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            llm.last_response_metadata = {"test": "value"}

            metadata1 = llm.get_last_response_metadata()
            metadata2 = llm.get_last_response_metadata()

            # Should be equal but not the same object
            assert metadata1 == metadata2
            assert metadata1 is not metadata2

            # Modifying returned copy shouldn't affect internal state
            metadata1["modified"] = True
            assert "modified" not in llm.last_response_metadata

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    def test_set_system_prompt(self, config):
        """Test set_system_prompt method."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]
            mock_langchain.return_value = mock_llm

            llm = config["class"](
                name=f"Test{config['name']}", system_prompt="Initial prompt"
            )
            assert llm.system_prompt == "Initial prompt"

            llm.set_system_prompt("Updated prompt")
            assert llm.system_prompt == "Updated prompt"

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_timestamp_format(self, config):
        """Test that timestamp is in ISO format."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_response = MagicMock()
            mock_response.text = "Test"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                mock_response.response_metadata = {}
            elif config["provider"] == "gemini":
                mock_metadata_obj = MagicMock()
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {}

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            await llm.generate_response(
                conversation_history=[
                    {"turn": 0, "speaker": "system", "response": "Test"}
                ]
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

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_response_with_conversation_history(self, config):
        """Test generate_response with conversation_history parameter."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Response with history"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                mock_response.response_metadata = {
                    "model_name": config["default_model"],
                    "token_usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 20,
                        "total_tokens": 70,
                    },
                }
            elif config["provider"] == "gemini":
                mock_metadata_obj = MagicMock()
                mock_metadata_obj.model_name = config["default_model"]
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {
                    "model": config["default_model"],
                    "usage": {"input_tokens": 50, "output_tokens": 20},
                }

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}", system_prompt="Test")

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

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_response_with_none_conversation_history(self, config):
        """Test generate_response with None conversation_history."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                mock_response.response_metadata = {
                    "model_name": config["default_model"]
                }
            elif config["provider"] == "gemini":
                mock_metadata_obj = MagicMock()
                mock_metadata_obj.model_name = config["default_model"]
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {"model": config["default_model"]}

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}", system_prompt="Test")

            # Actually pass None to test the default behavior
            response = await llm.generate_response(conversation_history=None)

            assert response == "Response"

            # Verify ainvoke was called
            call_args = mock_llm.ainvoke.call_args
            messages = call_args[0][0]

            # Should have: SystemMessage only (no history messages)
            assert len(messages) == 1
            assert isinstance(messages[0], SystemMessage)

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_response_with_persona_role_flips_types(self, config):
        """Test that persona role flips message types in conversation history."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Persona response"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                mock_response.response_metadata = {}
            elif config["provider"] == "gemini":
                mock_metadata_obj = MagicMock()
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {}

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            # Persona system prompt should trigger message type flipping
            persona_prompt = "You are roleplaying as a human user"
            llm = config["class"](
                name=f"Test{config['name']}", system_prompt=persona_prompt
            )

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

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_response_with_partial_usage_metadata(self, config):
        """Test response with incomplete/partial usage metadata."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_response = MagicMock()
            mock_response.text = "Partial usage response"
            mock_response.id = config["response_id"]

            # Set up partial usage metadata based on provider
            if config["provider"] == "openai":
                mock_response.response_metadata = {
                    "model_name": config["default_model"],
                    "token_usage": {"prompt_tokens": 15},  # Missing completion_tokens
                }
                mock_response.additional_kwargs = {}
            elif config["provider"] == "gemini":
                mock_metadata_obj = MagicMock()
                mock_metadata_obj.model_name = config["default_model"]
                mock_metadata_obj.__getitem__ = lambda self, key: {
                    "usage_metadata": {"prompt_token_count": 12}
                }.get(key)
                mock_metadata_obj.__contains__ = lambda self, key: key in [
                    "usage_metadata"
                ]
                mock_metadata_obj.get = lambda key, default=None: {
                    "usage_metadata": {"prompt_token_count": 12}
                }.get(key, default)
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {
                    "model": config["default_model"],
                    "usage": {"input_tokens": 15},  # Missing output_tokens
                }

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            response = await llm.generate_response(
                conversation_history=[
                    {"turn": 0, "speaker": "system", "response": "Test"}
                ]
            )

            assert response == "Partial usage response"
            metadata = llm.get_last_response_metadata()

            # Verify partial usage is handled (missing fields default to 0)
            if config["provider"] == "openai":
                assert metadata["usage"]["prompt_tokens"] == 15
                assert metadata["usage"]["completion_tokens"] == 0
            elif config["provider"] == "gemini":
                assert metadata["usage"]["prompt_token_count"] == 12
                assert metadata["usage"]["candidates_token_count"] == 0
            else:  # claude
                assert metadata["usage"]["input_tokens"] == 15
                assert metadata["usage"]["output_tokens"] == 0
                assert metadata["usage"]["total_tokens"] == 15

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_response_without_additional_kwargs(self, config):
        """Test response when additional_kwargs is missing (OpenAI-specific)."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                mock_response.response_metadata = {
                    "model_name": config["default_model"]
                }
                del mock_response.additional_kwargs  # Remove attribute
            elif config["provider"] == "gemini":
                mock_metadata_obj = MagicMock()
                mock_metadata_obj.model_name = config["default_model"]
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {"model": config["default_model"]}

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            response = await llm.generate_response(
                conversation_history=[
                    {"turn": 0, "speaker": "system", "response": "Test"}
                ]
            )

            assert response == "Response"
            metadata = llm.get_last_response_metadata()

            # OpenAI should default additional_kwargs to empty dict
            if config["provider"] == "openai":
                assert metadata["additional_kwargs"] == {}

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_response_with_fallback_token_usage(self, config):
        """Test response with fallback token_usage structure (Gemini-specific)."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_response = MagicMock()
            mock_response.text = "Response with fallback"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                mock_response.response_metadata = {
                    "model_name": config["default_model"],
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                }
                mock_response.additional_kwargs = {}
            elif config["provider"] == "gemini":
                # Gemini fallback: use token_usage when usage_metadata is missing
                mock_metadata_obj = MagicMock()
                mock_metadata_obj.model_name = config["default_model"]
                mock_metadata_obj.__getitem__ = lambda self, key: {
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    }
                }.get(key)
                mock_metadata_obj.__contains__ = lambda self, key: key in [
                    "token_usage"
                ]
                mock_metadata_obj.get = lambda key, default=None: {
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    }
                }.get(key, default)
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {
                    "model": config["default_model"],
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                }

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            response = await llm.generate_response(
                conversation_history=[
                    {"turn": 0, "speaker": "system", "response": "Test"}
                ]
            )

            assert response == "Response with fallback"
            metadata = llm.get_last_response_metadata()

            # Gemini should use fallback token_usage structure
            if config["provider"] == "gemini":
                assert metadata["usage"]["prompt_tokens"] == 10
                assert metadata["usage"]["completion_tokens"] == 20
                assert metadata["usage"]["total_tokens"] == 30

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_model_name_update_from_metadata(self, config):
        """Test model name update from response metadata (OpenAI-specific)."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_response = MagicMock()
            mock_response.text = "Test"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                # OpenAI updates model name from response_metadata.model_name
                updated_model = "gpt-4-0613-updated"
                mock_response.response_metadata = {"model_name": updated_model}
                mock_response.additional_kwargs = {}
            elif config["provider"] == "gemini":
                updated_model = "gemini-1.5-pro-002"
                mock_metadata_obj = MagicMock()
                mock_metadata_obj.model_name = updated_model
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                # Claude doesn't update model name from metadata
                updated_model = config["default_model"]
                mock_response.response_metadata = {"model": updated_model}

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](
                name=f"Test{config['name']}", model_name=config["default_model"]
            )
            await llm.generate_response(
                conversation_history=[
                    {"turn": 0, "speaker": "system", "response": "Test"}
                ]
            )

            metadata = llm.get_last_response_metadata()

            # OpenAI and Gemini update model name from metadata
            if config["provider"] in ["openai", "gemini"]:
                assert metadata["model"] == updated_model
            else:  # claude
                # Claude uses model from response_metadata.model if available
                assert metadata["model"] == updated_model

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_finish_reason_stop_reason_extraction(self, config):
        """Test finish_reason/stop_reason extraction based on provider."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_response = MagicMock()
            mock_response.text = "Finished response"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                mock_response.response_metadata = {
                    "model_name": config["default_model"],
                    "finish_reason": "max_tokens",
                }
                mock_response.additional_kwargs = {}
            elif config["provider"] == "gemini":
                mock_metadata_obj = MagicMock()
                mock_metadata_obj.model_name = config["default_model"]
                mock_metadata_obj.__getitem__ = lambda self, key: {
                    "finish_reason": "MAX_TOKENS"
                }.get(key)
                mock_metadata_obj.__contains__ = lambda self, key: key in [
                    "finish_reason"
                ]
                mock_metadata_obj.get = lambda key, default=None: {
                    "finish_reason": "MAX_TOKENS"
                }.get(key, default)
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {
                    "model": config["default_model"],
                    "stop_reason": "max_tokens",
                }

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            await llm.generate_response(
                conversation_history=[
                    {"turn": 0, "speaker": "system", "response": "Test"}
                ]
            )

            metadata = llm.get_last_response_metadata()

            # Verify correct field name based on provider
            if config["provider"] == "openai":
                assert metadata["finish_reason"] == "max_tokens"
                assert "stop_reason" not in metadata
            elif config["provider"] == "gemini":
                assert metadata["finish_reason"] == "MAX_TOKENS"
                assert "stop_reason" not in metadata
            else:  # claude
                assert metadata["stop_reason"] == "max_tokens"
                assert "finish_reason" not in metadata

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_metadata_includes_response_object(self, config):
        """Test that metadata includes the full response object."""
        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_response = MagicMock()
            mock_response.text = "Test"
            mock_response.id = config["response_id"]

            if config["provider"] == "openai":
                mock_response.response_metadata = {}
                mock_response.additional_kwargs = {}
            elif config["provider"] == "gemini":
                mock_metadata_obj = MagicMock()
                mock_response.response_metadata = mock_metadata_obj
            else:  # claude
                mock_response.response_metadata = {}

            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            await llm.generate_response(
                conversation_history=[
                    {"turn": 0, "speaker": "system", "response": "Test"}
                ]
            )

            metadata = llm.get_last_response_metadata()
            assert "response" in metadata
            assert metadata["response"] == mock_response

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_structured_response_success_with_system_prompt(
        self, config
    ):
        """Test successful structured response generation with system prompt."""

        # Define a simple Pydantic model for testing
        class TestResponse(BaseModel):
            answer: str
            reasoning: str

        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            # Create mock structured LLM that returns a Pydantic model instance
            mock_structured_llm = MagicMock()
            expected_response = TestResponse(
                answer="Yes", reasoning="The response was appropriate."
            )
            mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )
            mock_langchain.return_value = mock_llm

            llm = config["class"](
                name=f"Test{config['name']}",
                system_prompt="You are a helpful assistant.",
            )
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
            assert metadata["model"] == config["default_model"]
            assert metadata["provider"] == config["provider"]
            assert metadata["structured_output"] is True
            assert "timestamp" in metadata
            assert "response_time_seconds" in metadata
            assert metadata["response_id"] is None
            assert metadata["usage"] == {}

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_structured_response_success_without_system_prompt(
        self, config
    ):
        """Test successful structured response generation without system prompt."""

        class TestResponse(BaseModel):
            answer: str
            reasoning: str

        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_structured_llm = MagicMock()
            expected_response = TestResponse(
                answer="No", reasoning="The response was not appropriate."
            )
            mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")  # No system prompt
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

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_structured_response_with_empty_message(self, config):
        """Test structured response generation with empty string message."""

        class TestResponse(BaseModel):
            answer: str

        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_structured_llm = MagicMock()
            expected_response = TestResponse(answer="Maybe")
            mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
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

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_structured_response_api_error(self, config):
        """Test error handling when structured response API call fails."""

        class TestResponse(BaseModel):
            answer: str

        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")

            # Should raise RuntimeError with error message
            with pytest.raises(RuntimeError) as exc_info:
                await llm.generate_structured_response(
                    message="Test message", response_model=TestResponse
                )

            assert "Error generating structured response" in str(exc_info.value)
            assert "API rate limit exceeded" in str(exc_info.value)

            # Verify error metadata was stored
            metadata = llm.get_last_response_metadata()
            assert metadata["model"] == config["default_model"]
            assert metadata["provider"] == config["provider"]
            assert "timestamp" in metadata
            assert "error" in metadata
            assert "API rate limit exceeded" in metadata["error"]
            assert metadata["usage"] == {}
            assert "structured_output" not in metadata  # Not set on error

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_structured_response_type_validation_error(self, config):
        """Test error handling when response is not the correct type."""

        class TestResponse(BaseModel):
            answer: str

        class WrongResponse(BaseModel):
            value: int

        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_structured_llm = MagicMock()
            # Return wrong type (shouldn't happen in practice, but test the validation)
            wrong_response = WrongResponse(value=42)
            mock_structured_llm.ainvoke = AsyncMock(return_value=wrong_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")

            # Should raise RuntimeError due to type validation
            with pytest.raises(RuntimeError) as exc_info:
                await llm.generate_structured_response(
                    message="Test message", response_model=TestResponse
                )

            assert "Error generating structured response" in str(exc_info.value)

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_structured_response_tracks_timing(self, config):
        """Test that structured response timing is tracked correctly."""

        class TestResponse(BaseModel):
            answer: str

        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_structured_llm = MagicMock()
            expected_response = TestResponse(answer="Test")
            mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            await llm.generate_structured_response(
                message="Test message", response_model=TestResponse
            )

            metadata = llm.get_last_response_metadata()
            assert "response_time_seconds" in metadata
            assert isinstance(metadata["response_time_seconds"], (int, float))
            assert metadata["response_time_seconds"] >= 0

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_structured_response_timestamp_format(self, config):
        """Test that structured response timestamp is in ISO format."""

        class TestResponse(BaseModel):
            answer: str

        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_structured_llm = MagicMock()
            expected_response = TestResponse(answer="Test")
            mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
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

    @pytest.mark.parametrize("config", JUDGE_LLM_CONFIGS)
    @pytest.mark.asyncio
    @patch("llm_clients.claude_llm.Config.ANTHROPIC_API_KEY", "test-key")
    @patch("llm_clients.openai_llm.Config.OPENAI_API_KEY", "test-key")
    @patch("llm_clients.gemini_llm.Config.GOOGLE_API_KEY", "test-key")
    async def test_generate_structured_response_with_complex_model(self, config):
        """Test structured response with a more complex Pydantic model."""

        class NestedModel(BaseModel):
            value: int

        class ComplexResponse(BaseModel):
            answer: str
            score: int
            nested: NestedModel
            optional_field: str | None = None

        with patch(config["langchain_class"]) as mock_langchain:
            mock_llm = MagicMock()
            mock_llm.model = config["default_model"]

            mock_structured_llm = MagicMock()
            expected_response = ComplexResponse(
                answer="Yes",
                score=85,
                nested=NestedModel(value=42),
                optional_field="optional",
            )
            mock_structured_llm.ainvoke = AsyncMock(return_value=expected_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )
            mock_langchain.return_value = mock_llm

            llm = config["class"](name=f"Test{config['name']}")
            response = await llm.generate_structured_response(
                message="Evaluate this.", response_model=ComplexResponse
            )

            assert isinstance(response, ComplexResponse)
            assert response.answer == "Yes"
            assert response.score == 85
            assert isinstance(response.nested, NestedModel)
            assert response.nested.value == 42
            assert response.optional_field == "optional"
