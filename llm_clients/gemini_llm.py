import json
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

from utils.conversation_utils import build_langchain_messages
from utils.debug import debug_print

from .config import Config
from .llm_interface import JudgeLLM, Role

T = TypeVar("T", bound=BaseModel)

_STRUCTURED_OUTPUT_MAX_LOG_CHARS = 4000


def _truncate_for_log(
    value: Any, max_chars: int = _STRUCTURED_OUTPUT_MAX_LOG_CHARS
) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated, {len(text)} chars total]"


def _serialize_raw_message(raw: Any) -> Dict[str, Any]:
    """Serialize an AIMessage (or similar) for failure diagnostics."""
    if raw is None:
        return {}
    if isinstance(raw, AIMessage):
        payload: Dict[str, Any] = {
            "type": "AIMessage",
            "content": raw.content,
            "tool_calls": raw.tool_calls,
        }
        if raw.response_metadata:
            payload["response_metadata"] = dict(raw.response_metadata)
        if raw.additional_kwargs:
            payload["additional_kwargs"] = dict(raw.additional_kwargs)
        return payload
    return {"type": type(raw).__name__, "repr": _truncate_for_log(raw)}


def _extract_include_raw_result(
    invoke_result: Any,
) -> tuple[Any, Any, Any]:
    """Unpack LangChain include_raw=True structured output payloads."""
    if isinstance(invoke_result, dict) and "parsed" in invoke_result:
        return (
            invoke_result.get("parsed"),
            invoke_result.get("raw"),
            invoke_result.get("parsing_error"),
        )
    return invoke_result, None, None


def _coerce_structured_response(response: Any, response_model: Type[T]) -> Optional[T]:
    """Normalize structured LLM output to a Pydantic model instance."""
    if response is None:
        return None
    if isinstance(response, response_model):
        return response
    if isinstance(response, dict):
        try:
            return response_model.model_validate(response)
        except ValidationError:
            return None
    return None


class GeminiLLM(JudgeLLM):
    """Gemini implementation using LangChain.

    Newer Gemini models may use the API's **implicit** prompt caching automatically
    (shared-prefix requests); this client does not add special flags for that.

    **Explicit** Context Caching (``cached_content`` resource names) is not wired here;
    it needs a separate cache create/update lifecycle. There is no Anthropic-style
    ``cache_control`` on ``ChatGoogleGenerativeAI``.
    """

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

        # Print configuration before creating LLM
        print("Creating Gemini LLM with parameters:")
        print(f"  Model: {llm_params['model']}")
        print(f"  Temperature: {llm_params.get('temperature', 'default')}")
        print(f"  Max tokens: {llm_params.get('max_tokens', 'default')}")
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

            model = (
                getattr(response.response_metadata, "model_name", self.model_name)
                if hasattr(response, "response_metadata")
                else self.model_name
            )
            self._set_response_metadata(
                "gemini",
                response_id=getattr(response, "id", None),
                model=model,
                response_time_seconds=round(end_time - start_time, 3),
                finish_reason=None,
                response=response,
            )

            if hasattr(response, "response_metadata") and response.response_metadata:
                metadata = response.response_metadata

                if "usage_metadata" in metadata:
                    usage = metadata["usage_metadata"]
                    self._last_response_metadata["usage"] = {
                        "prompt_token_count": usage.get("prompt_token_count", 0),
                        "candidates_token_count": usage.get(
                            "candidates_token_count", 0
                        ),
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
            "raw_content": _truncate_for_log(raw_payload.get("content", "")),
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
            structured_llm = self.llm.with_structured_output(
                response_model,
                method="json_schema",
                include_raw=True,
            )

            start_time = time.time()
            invoke_result = await structured_llm.ainvoke(messages)
            end_time = time.time()

            parsed, raw, parsing_error = _extract_include_raw_result(invoke_result)
            coerced = _coerce_structured_response(parsed, response_model)

            self._set_response_metadata(
                "gemini",
                response_time_seconds=round(end_time - start_time, 3),
                structured_output=True,
                structured_output_method="json_schema",
            )

            if coerced is not None:
                failure_debug.clear()
                debug_print(
                    f"[DEBUG {self.name}] Structured {response_model.__name__}: "
                    f"{coerced!r}"
                )
                return coerced

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
