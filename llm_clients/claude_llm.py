import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from utils.conversation_utils import build_langchain_messages
from utils.debug import debug_print

from .config import Config
from .llm_interface import (
    JudgeLLM,
    Role,
    ensure_pydantic_response,
    extract_structured_invoke_result,
)

T = TypeVar("T", bound=BaseModel)

# Anthropic-only: prompt caching is opt-in per request (not `cache_control` elsewhere).
# Default ephemeral TTL is 5m (no `ttl` key).
# TTL is the time after which the stored context will expire
# and be removed from memory if it is not used.
_DEFAULT_ANTHROPIC_CACHE_CONTROL: Dict[str, Any] = {"type": "ephemeral"}


class ClaudeLLM(JudgeLLM):
    """Claude implementation using LangChain.

    Prompt caching uses Anthropic's per-request ``cache_control`` (see ``caching`` and
    ``anthropic_cache_control`` constructor args).
    """

    def _no_retry_substrings(self) -> tuple[str, ...]:
        # Anthropic API / Messages API (see https://docs.anthropic.com/en/api/errors)
        return (
            "credit balance is too low",
            "insufficient_quota",
            "invalid x-api-key",
            "invalid_api_key",
            "authentication_error",
        )

    def __init__(
        self,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        caching: bool = True,
        **kwargs,
    ):
        first_message = kwargs.pop("first_message", None)
        start_prompt = kwargs.pop("start_prompt", None)
        cache_control_arg: Optional[Dict[str, Any]] = kwargs.pop(
            "anthropic_cache_control", dict(_DEFAULT_ANTHROPIC_CACHE_CONTROL)
        )
        if not caching:
            self._anthropic_cache_control: Optional[Dict[str, Any]] = None
        else:
            self._anthropic_cache_control = cache_control_arg
        super().__init__(
            name,
            role,
            system_prompt,
            first_message=first_message,
            start_prompt=start_prompt,
        )

        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_claude_config()["model"]

        # Get default config and allow kwargs to override
        llm_params = {
            "anthropic_api_key": Config.ANTHROPIC_API_KEY,
            "model": self.model_name,
        }

        # Override with any provided kwargs
        llm_params.update(kwargs)

        # Print configuration before creating LLM
        print("Creating Claude LLM with parameters:")
        print(f"  Model: {llm_params['model']}")
        print(f"  Temperature: {llm_params.get('temperature', 'default')}")
        print(f"  Max tokens: {llm_params.get('max_tokens', 'default')}")
        extra_params = {
            k: v
            for k, v in llm_params.items()
            if k not in ["model", "anthropic_api_key"]
        }
        if extra_params:
            print(f"  Extra parameters: {extra_params}")

        self.llm = ChatAnthropic(**llm_params)

        print(f"Using Claude model: {self.llm.model}")

        # Store configuration parameters for logging
        self.temperature = getattr(self.llm, "temperature", None)
        self.max_tokens = getattr(self.llm, "max_tokens", None)

    def _enrich_claude_message_metadata(self, response: Any) -> None:
        """Merge token usage and response fields from a LangChain AIMessage."""
        if response is None:
            return

        response_id = getattr(response, "id", None)
        if response_id is not None:
            self._last_response_metadata["response_id"] = response_id

        model = (
            getattr(response.response_metadata, "model", self.model_name)
            if hasattr(response, "response_metadata")
            else self.model_name
        )
        if model is not None:
            self._last_response_metadata["model"] = model

        if hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata
            if "usage" in metadata:
                usage = metadata["usage"]
                self._last_response_metadata["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                }
                for ck in (
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ):
                    if usage.get(ck) is not None:
                        self._last_response_metadata["usage"][ck] = usage[ck]
            self._last_response_metadata["stop_reason"] = metadata.get("stop_reason")
            self._last_response_metadata["raw_metadata"] = dict(metadata)

    async def start_conversation(self) -> str:
        """Produce the first response:
        - static first_message if set, or
        - LLM with start_prompt if first_message is not set.
        """
        if self.first_message is not None:
            self._set_response_metadata("claude", static_first_message=True)
            return self.first_message
        return await self.generate_response(self.get_initial_prompt_turns())

    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response based on conversation history.

        Args:
            conversation_history: Optional list of previous conversation turns
        """
        if not conversation_history or len(conversation_history) == 0:
            return await self.start_conversation()

        messages = []

        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        messages.extend(build_langchain_messages(self.role, conversation_history))

        # Debug: Print messages being sent to LLM
        debug_print(f"\n[DEBUG {self.name} - {self.role.value}] Messages sent to LLM:")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            preview = msg.text[:100]
            content_preview = preview + "..." if len(msg.text) > 100 else msg.text
            debug_print(f"  {i + 1}. {msg_type}: {content_preview}")

        async def _invoke() -> str:
            start_time = time.time()
            invoke_kw: Dict[str, Any] = {}
            if self._anthropic_cache_control is not None:
                invoke_kw["cache_control"] = self._anthropic_cache_control
            response = await self.llm.ainvoke(messages, **invoke_kw)
            end_time = time.time()

            model = (
                getattr(response.response_metadata, "model", self.model_name)
                if hasattr(response, "response_metadata")
                else self.model_name
            )
            self._set_response_metadata(
                "claude",
                response_id=getattr(response, "id", None),
                model=model,
                response_time_seconds=round(end_time - start_time, 3),
                stop_reason=None,
                response=response,
            )
            self._enrich_claude_message_metadata(response)

            return response.text

        return await self._run_with_retry(_invoke, provider="claude")

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        """Generate a structured response using Pydantic model.

        Args:
            message: The prompt message
            response_model: Pydantic model class to structure the response

        Returns:
            Instance of the response_model with structured data
        """
        messages = []

        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        messages.append(HumanMessage(content=message))

        async def _invoke() -> T:
            structured_llm = self.llm.with_structured_output(
                response_model,
                include_raw=True,
            )

            start_time = time.time()
            invoke_kw: Dict[str, Any] = {}
            if self._anthropic_cache_control is not None:
                invoke_kw["cache_control"] = self._anthropic_cache_control
            invoke_result = await structured_llm.ainvoke(messages, **invoke_kw)
            end_time = time.time()

            parsed, raw, _ = extract_structured_invoke_result(invoke_result)
            response = ensure_pydantic_response(parsed, response_model)

            self._set_response_metadata(
                "claude",
                response_time_seconds=round(end_time - start_time, 3),
                structured_output=True,
            )
            self._enrich_claude_message_metadata(raw)

            if response is None:
                raise ValueError(
                    f"Response is not an instance of {response_model.__name__}"
                )

            return response

        return await self._run_with_retry(_invoke, provider="claude")

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
