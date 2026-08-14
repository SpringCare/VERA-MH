import re
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
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

# gpt-5+ (non-chat) reasoning models reject these sampling params outright
# once reasoning is active. langchain_openai only silently strips
# `temperature` for these models (see `validate_temperature` in
# langchain_openai.chat_models.base) - top_p/penalties/logprobs reach the
# API untouched and the request 400s. We filter all of them ourselves via
# `_model_supports_param` so upstream defaults (e.g. the judge's
# temperature=0) don't crash reasoning runs.
#
# Gated on major version (5+), not a fixed "gpt-5" prefix: OpenAI's reasoning
# line is expected to continue, so we cover gpt-6+ without editing here. The
# `*-chat` variants stay on the classic sampling params and are excluded.
_REASONING_ONLY_GPT_MAJOR = 5
_GPT_VERSION_RE = re.compile(r"gpt-(\d+)")
_UNSUPPORTED_REASONING_PARAMS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
    }
)


def _gpt_major_version(model_name: str) -> Optional[int]:
    """Major version parsed from a GPT model name (``gpt-5.4-mini`` -> 5)."""
    match = _GPT_VERSION_RE.search(model_name.lower())
    return int(match.group(1)) if match else None


class OpenAILLM(JudgeLLM):
    """OpenAI implementation using LangChain.

    Prompt caching is automatic for eligible models/prefixes; we pass
    ``prompt_cache_key`` (per conversation) on each call so routing can improve
    cache hits.
    Since this is automatic, we do not require something like
    Anthropic's ``cache_control``.
    """

    def _model_supports_param(self, model_name: str, param_name: str) -> bool:
        """gpt-5+ (excluding *-chat) reject sampling params when reasoning."""
        model_lower = model_name.lower()
        major = _gpt_major_version(model_lower)
        is_reasoning = (
            major is not None
            and major >= _REASONING_ONLY_GPT_MAJOR
            and "chat" not in model_lower
        )
        if is_reasoning and param_name.lower() in _UNSUPPORTED_REASONING_PARAMS:
            return False
        return super()._model_supports_param(model_name, param_name)

    def _no_retry_substrings(self) -> tuple[str, ...]:
        # https://platform.openai.com/docs/guides/error-codes
        return (
            "insufficient_quota",
            "billing_hard_limit",
            "Your account is not active",
            "invalid_api_key",
            "account_deactivated",
        )

    def __init__(
        self,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        first_message = kwargs.pop("first_message", None)
        start_prompt = kwargs.pop("start_prompt", None)
        super().__init__(
            name,
            role,
            system_prompt,
            first_message=first_message,
            start_prompt=start_prompt,
        )

        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_openai_config()["model"]

        # Get default config and allow kwargs to override
        llm_params = {
            "openai_api_key": Config.OPENAI_API_KEY,
            "model": self.model_name,
        }

        if Config.OPENAI_BASE_URL:
            llm_params["base_url"] = Config.OPENAI_BASE_URL

        # Override with any provided kwargs
        llm_params.update(kwargs)
        filtered_params = self._filter_supported_params(self.model_name, llm_params)
        dropped_params = sorted(set(llm_params) - set(filtered_params))
        llm_params = filtered_params

        # Print configuration before creating LLM
        print("Creating OpenAI LLM with parameters:")
        print(f"  Model: {llm_params['model']}")
        print(f"  Temperature: {llm_params.get('temperature', 'default')}")
        print(f"  Max tokens: {llm_params.get('max_tokens', 'default')}")
        if dropped_params:
            print(
                f"  Dropped (unsupported for {self.model_name}): "
                f"{', '.join(dropped_params)}"
            )
        extra_params = {
            k: v for k, v in llm_params.items() if k not in ["model", "openai_api_key"]
        }
        if extra_params:
            print(f"  Extra parameters: {extra_params}")

        self.llm = ChatOpenAI(**llm_params)

    def _enrich_openai_message_metadata(self, response: Any) -> None:
        """Merge token usage and response fields from a LangChain AIMessage."""
        if response is None:
            return

        response_id = getattr(response, "id", None)
        if response_id is not None:
            self._last_response_metadata["response_id"] = response_id

        if hasattr(response, "additional_kwargs") and response.additional_kwargs:
            self._last_response_metadata["additional_kwargs"] = dict(
                response.additional_kwargs
            )

        self._last_response_metadata["model"] = self._extract_response_model(response)

        if hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata

            if "token_usage" in metadata:
                token_usage = metadata["token_usage"]
                self._last_response_metadata["usage"] = {
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }

            self._last_response_metadata["finish_reason"] = metadata.get(
                "finish_reason"
            )
            self._last_response_metadata["system_fingerprint"] = metadata.get(
                "system_fingerprint"
            )
            self._last_response_metadata["logprobs"] = metadata.get("logprobs")
            self._last_response_metadata["raw_response_metadata"] = dict(metadata)

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage_meta = response.usage_metadata
            self._last_response_metadata.setdefault("usage", {})
            self._last_response_metadata["usage"].update(
                {
                    "input_tokens": usage_meta.get("input_tokens", 0),
                    "output_tokens": usage_meta.get("output_tokens", 0),
                    "total_tokens": usage_meta.get("total_tokens", 0),
                }
            )
            self._last_response_metadata["raw_usage_metadata"] = dict(usage_meta)

    async def start_conversation(self) -> str:
        """Produce the first response:
        - static first_message if set, or
        - LLM with start_prompt if first_message is not set.
        """
        if self.first_message is not None:
            self._set_response_metadata("openai", static_first_message=True)
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

        # Debug: Print input parameters
        debug_print(f"\n[DEBUG {self.name} - {self.role.value}] Input parameters:")
        hist_len = len(conversation_history) if conversation_history else 0
        debug_print(f"  - conversation_history length: {hist_len}")

        # Build messages from history
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
            response = await self.llm.ainvoke(
                messages,
                prompt_cache_key=self.conversation_id,
            )
            end_time = time.time()

            self._set_response_metadata(
                "openai",
                response_id=getattr(response, "id", None),
                response_time_seconds=round(end_time - start_time, 3),
                finish_reason=None,
                additional_kwargs={},
                system_fingerprint=None,
                logprobs=None,
                response=response,
            )
            self._enrich_openai_message_metadata(response)

            return response.text

        return await self._run_with_retry(_invoke, provider="openai")

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
            invoke_result = await structured_llm.ainvoke(
                messages,
                prompt_cache_key=self.conversation_id,
            )
            end_time = time.time()

            parsed, raw, _ = extract_structured_invoke_result(invoke_result)
            response = ensure_pydantic_response(parsed, response_model)

            self._set_response_metadata(
                "openai",
                response_time_seconds=round(end_time - start_time, 3),
                structured_output=True,
            )
            self._enrich_openai_message_metadata(raw)

            if response is None:
                raise ValueError(
                    f"Response is not an instance of {response_model.__name__}"
                )

            return response

        return await self._run_with_retry(_invoke, provider="openai")

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
