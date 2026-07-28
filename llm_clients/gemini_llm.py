import json
import re
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
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

_STRUCTURED_OUTPUT_MAX_LOG_CHARS = 4000

# Gemini 3.x: Google's migration guidance says temperature/top_p/top_k are "no
# longer recommended" and forcing low values (e.g. this judge's temperature=0
# default) "can cause potential looping issues or performance degradation."
# Unlike OpenAI's gpt-5.x, the API doesn't reject these outright -
# langchain_google_genai passes them straight through untouched - so we drop
# them ourselves rather than let an inherited default silently hurt judge
# quality. See:
# https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate
#
# Gated on major version (3+), not a fixed "gemini-3" marker: this is Google's
# steer-via-prompt direction for the model line, so we expect gemini-4+ to keep
# rejecting these params until Google says otherwise.
_REASONING_ONLY_GEMINI_MAJOR = 3
_GEMINI_VERSION_RE = re.compile(r"gemini-(\d+)")
_UNSUPPORTED_SAMPLING_PARAMS: frozenset[str] = frozenset(
    {"temperature", "top_p", "top_k"}
)


def _gemini_major_version(model_name: str) -> Optional[int]:
    """Major version parsed from a Gemini model name (``gemini-3-pro`` -> 3)."""
    match = _GEMINI_VERSION_RE.search(model_name.lower())
    return int(match.group(1)) if match else None


def _truncate_for_log(
    value: Any, max_chars: int = _STRUCTURED_OUTPUT_MAX_LOG_CHARS
) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated, {len(text)} chars total]"


def _raw_message_content_for_log(raw: Any) -> str:
    """Bounded text preview of the raw model message body."""
    if raw is None:
        return ""
    if isinstance(raw, AIMessage):
        return _truncate_for_log(raw.content)
    return _truncate_for_log(raw)


def _serialize_raw_message(raw: Any) -> Dict[str, Any]:
    """Serialize an AIMessage (or similar) for failure diagnostics.

    Returns the structural ``raw`` sub-dict (type, tool_calls, response_metadata,
    etc.) used inside :meth:`_build_structured_output_failure_debug`. It
    deliberately omits ``AIMessage.content`` here so the body is not duplicated
    inside nested metadata. The caller adds a separate top-level ``raw_content``
    key with a truncated text preview — one bounded copy of the response text
    when the whole failure debug dict is logged.
    """
    if raw is None:
        return {}
    if isinstance(raw, AIMessage):
        payload: Dict[str, Any] = {
            "type": "AIMessage",
            "tool_calls": raw.tool_calls,
        }
        if raw.response_metadata:
            payload["response_metadata"] = dict(raw.response_metadata)
        if raw.additional_kwargs:
            payload["additional_kwargs"] = dict(raw.additional_kwargs)
        return payload
    return {"type": type(raw).__name__, "repr": _truncate_for_log(raw)}


class GeminiLLM(JudgeLLM):
    """Gemini implementation using LangChain.

    Newer Gemini models may use the API's **implicit** prompt caching automatically
    (shared-prefix requests); this client does not add special flags for that.

    **Explicit** Context Caching (``cached_content`` resource names) is not wired here;
    it needs a separate cache create/update lifecycle. There is no Anthropic-style
    ``cache_control`` on ``ChatGoogleGenerativeAI``.

    Reasoning/thinking is controlled via two flat, native ``ChatGoogleGenerativeAI``
    fields that pass straight through kwargs (e.g. ``-jep thinking_level=high``):
    ``thinking_level`` (``"low"``/``"high"``, Gemini 3+, takes precedence) and
    ``thinking_budget`` (int token budget, older/Gemini 2.5 models). No dict
    wrapping is needed here, unlike Claude's ``thinking`` param.

    Gemini 3.x and later drop ``temperature``/``top_p``/``top_k`` (see
    ``_model_supports_param``); Google's own guidance says these are no longer
    recommended and can degrade reasoning quality if forced.
    """

    def _model_supports_param(self, model_name: str, param_name: str) -> bool:
        """Filter out sampling params Gemini 3+ no longer wants."""
        major = _gemini_major_version(model_name)
        if major is not None and major >= _REASONING_ONLY_GEMINI_MAJOR:
            if param_name.lower() in _UNSUPPORTED_SAMPLING_PARAMS:
                return False
        return super()._model_supports_param(model_name, param_name)

    def _no_retry_substrings(self) -> tuple[str, ...]:
        # Google AI / Gemini API (ai.google.dev); LangChain may wrap HTTP/GRPC text.
        return (
            "API_KEY_INVALID",
            "API key not valid",
            "PERMISSION_DENIED",
            "BILLING_NOT_ENABLED",
            "billing has not been enabled",
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

        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_gemini_config()["model"]

        # Get default config and allow kwargs to override
        llm_params = {
            "google_api_key": Config.GOOGLE_API_KEY,
            "model": self.model_name,
        }

        # Override with any provided kwargs
        llm_params.update(kwargs)
        filtered_params = self._filter_supported_params(self.model_name, llm_params)
        dropped_params = sorted(set(llm_params) - set(filtered_params))
        llm_params = filtered_params

        # Print configuration before creating LLM
        print("Creating Gemini LLM with parameters:")
        print(f"  Model: {llm_params['model']}")
        print(f"  Temperature: {llm_params.get('temperature', 'default')}")
        print(f"  Max tokens: {llm_params.get('max_tokens', 'default')}")
        if dropped_params:
            print(
                f"  Dropped (unsupported for {self.model_name}): "
                f"{', '.join(dropped_params)}"
            )
        extra_params = {
            k: v for k, v in llm_params.items() if k not in ["model", "google_api_key"]
        }
        if extra_params:
            print(f"  Extra parameters: {extra_params}")

        self.llm = ChatGoogleGenerativeAI(**llm_params)

        print(f"Using Gemini model: {self.llm.model}")

        # Store configuration parameters for logging
        self.temperature = getattr(self.llm, "temperature", None)
        self.max_tokens = getattr(self.llm, "max_tokens", None)

    def _enrich_gemini_message_metadata(self, response: Any) -> None:
        """Merge token usage and response fields from a LangChain AIMessage."""
        if response is None:
            return

        response_id = getattr(response, "id", None)
        if response_id is not None:
            self._last_response_metadata["response_id"] = response_id

        self._last_response_metadata["model"] = self._extract_response_model(response)

        if hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata

            if "usage_metadata" in metadata:
                usage = metadata["usage_metadata"]
                self._last_response_metadata["usage"] = {
                    "prompt_token_count": usage.get("prompt_token_count", 0),
                    "candidates_token_count": usage.get("candidates_token_count", 0),
                    "total_token_count": usage.get("total_token_count", 0),
                }
            elif "token_usage" in metadata:
                usage = metadata["token_usage"]
                self._last_response_metadata["usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            self._last_response_metadata["finish_reason"] = metadata.get(
                "finish_reason"
            )
            self._last_response_metadata["raw_metadata"] = dict(metadata)

    async def start_conversation(self) -> str:
        """Produce the first response:
        - static first_message if set, or
        - LLM with start_prompt if first_message is not set.
        """
        if self.first_message is not None:
            self._set_response_metadata("gemini", static_first_message=True)
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
            response = await self.llm.ainvoke(messages)
            end_time = time.time()

            model = self._extract_response_model(response)
            self._set_response_metadata(
                "gemini",
                response_id=getattr(response, "id", None),
                model=model,
                response_time_seconds=round(end_time - start_time, 3),
                finish_reason=None,
                response=response,
            )
            self._enrich_gemini_message_metadata(response)

            return response.text

        return await self._run_with_retry(_invoke, provider="gemini")

    def _build_structured_output_failure_debug(
        self,
        *,
        response_model: Type[T],
        parsed: Any,
        raw: Any,
        parsing_error: Any,
    ) -> Dict[str, Any]:
        raw_payload = _serialize_raw_message(raw)
        return {
            "structured_output_method": "json_schema",
            "expected_model": response_model.__name__,
            "parsed_type": type(parsed).__name__,
            "parsed_repr": _truncate_for_log(parsed),
            "parsing_error": _truncate_for_log(parsing_error)
            if parsing_error is not None
            else None,
            "raw": raw_payload,
            "raw_content": _raw_message_content_for_log(raw),
            "raw_tool_calls": raw_payload.get("tool_calls"),
        }

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        """Generate a structured response using Pydantic model.

        Uses Gemini native JSON schema (via LangChain) and captures raw responses
        when parsing fails so judge logs can show what the model returned.

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

        failure_debug: Dict[str, Any] = {}

        async def _invoke() -> T:
            failure_debug.clear()
            structured_llm = self.llm.with_structured_output(
                response_model,
                method="json_schema",
                include_raw=True,
            )

            start_time = time.time()
            invoke_result = await structured_llm.ainvoke(messages)
            end_time = time.time()

            parsed, raw, parsing_error = extract_structured_invoke_result(invoke_result)
            response = ensure_pydantic_response(parsed, response_model)

            self._set_response_metadata(
                "gemini",
                response_time_seconds=round(end_time - start_time, 3),
                structured_output=True,
                structured_output_method="json_schema",
            )
            self._enrich_gemini_message_metadata(raw)

            if response is not None:
                failure_debug.clear()
                debug_print(
                    f"[DEBUG {self.name}] Structured {response_model.__name__}: "
                    f"{response!r}"
                )
                return response

            failure_debug.clear()
            failure_debug.update(
                self._build_structured_output_failure_debug(
                    response_model=response_model,
                    parsed=parsed,
                    raw=raw,
                    parsing_error=parsing_error,
                )
            )
            debug_print(
                f"[DEBUG {self.name}] Structured output failure:\n"
                f"{json.dumps(failure_debug, indent=2, default=str)}"
            )
            raise ValueError(
                f"Expected {response_model.__name__}, got {type(parsed).__name__}: "
                f"{_truncate_for_log(parsed)}. "
                f"raw_content={failure_debug.get('raw_content')!r}"
            )

        def _on_structured_error(
            error: BaseException,
            attempt_number: int,
            max_attempts: int,
            retryable: bool,
            will_retry: bool,
        ) -> Optional[Dict[str, Any]]:
            _ = (error, attempt_number, max_attempts, retryable, will_retry)
            if failure_debug:
                return {"structured_output_debug": dict(failure_debug)}
            return None

        return await self._run_with_retry(
            _invoke, provider="gemini", on_error=_on_structured_error
        )

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
