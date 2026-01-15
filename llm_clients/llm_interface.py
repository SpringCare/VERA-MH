import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMInterface(ABC):
    """Abstract base class for LLM implementations.

    Provides basic text generation capabilities. All LLM implementations
    must support basic text generation and system prompt management.
    """

    def __init__(
        self, name: str, system_prompt: Optional[str] = None, max_retries: int = 10
    ):
        self.name = name
        self.system_prompt = system_prompt or ""
        self.max_retries = max_retries

    @abstractmethod
    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response based on conversation history.

        Args:
            conversation_history: List of previous conversation turns.
                Each turn is a dict with keys: 'turn', 'speaker', 'response'.
                On the first turn (turn 0), conversation_history will contain
                a single entry with turn=0, speaker="system", and the initial
                message in the 'response' field. This provides context for
                starting the conversation.

        Returns:
            str: The response text. Metadata available via
                get_last_response_metadata()
        """
        pass

    @abstractmethod
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        pass

    def get_name(self) -> str:
        """Get the name of this LLM instance."""
        return self.name

    def _extract_http_status_code(self, exception: Exception) -> Optional[int]:
        """Extract HTTP status code from exception if available.

        LangChain and various HTTP libraries wrap HTTP errors differently.
        This method attempts to extract the status code from common
        exception types.
        """
        # Check for status_code attribute (common in HTTPException)
        if hasattr(exception, "status_code"):
            status_code = getattr(exception, "status_code")
            if status_code is not None:
                return int(status_code)

        # Check for response attribute with status_code
        if hasattr(exception, "response"):
            response = getattr(exception, "response")
            if hasattr(response, "status_code"):
                status_code = getattr(response, "status_code")
                if status_code is not None:
                    return int(status_code)
            if hasattr(response, "status"):
                status = getattr(response, "status")
                if status is not None:
                    return int(status)

        # Check for status attribute directly
        if hasattr(exception, "status"):
            status = getattr(exception, "status")
            if status is not None:
                return int(status)

        # Check exception message for status codes (fallback)
        error_str = str(exception).lower()
        for code in [429, 500, 502, 503, 504, 529]:
            if f"status {code}" in error_str or f"status_code {code}" in error_str:
                return code

        return None

    def _extract_retry_after(self, exception: Exception) -> Optional[int]:
        """Extract Retry-After header value from exception if available."""
        if hasattr(exception, "response"):
            response = getattr(exception, "response")
            if hasattr(response, "headers"):
                headers = getattr(response, "headers")
                retry_after = headers.get("Retry-After") or headers.get("retry-after")
                if retry_after:
                    try:
                        return int(retry_after)
                    except (ValueError, TypeError):
                        pass
        return None

    async def _retry_with_backoff(
        self,
        func: Callable[[], Any],
        operation_name: str = "operation",
    ) -> Any:
        """Execute a function with retry logic for transient HTTP errors.

        Handles the following HTTP status codes:
        - 429 (Too Many Requests): Respects Retry-After header,
          otherwise exponential backoff
        - 500 (Internal Server Error): Retry 1-3 times with
          exponential backoff
        - 502 (Bad Gateway): Retry 1-3 times with exponential backoff
        - 503 (Service Unavailable): Exponential backoff
        - 504 (Gateway Timeout): Exponential backoff
        - 529 (Overloaded - Anthropic): Treated like 503 with
          exponential backoff

        Args:
            func: Async function to execute
            operation_name: Name of operation for error messages

        Returns:
            Result of func()

        Raises:
            RuntimeError: If max retries exceeded or non-retryable
                error occurs
        """
        retryable_status_codes = {429, 500, 502, 503, 504, 529}
        max_retries_for_500_502 = 3  # Limit retries for 500/502

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                status_code = self._extract_http_status_code(e)

                # If we can't determine status code, check if it's
                # retryable by message
                if status_code is None:
                    error_str = str(e).lower()
                    # Check for common retryable error messages
                    retryable_keywords = [
                        "rate limit",
                        "too many requests",
                        "service unavailable",
                        "internal server error",
                        "bad gateway",
                        "gateway timeout",
                        "overloaded",
                        "timeout",
                    ]
                    if any(keyword in error_str for keyword in retryable_keywords):
                        # Treat as retryable, use exponential backoff
                        status_code = 503  # Default for unknown retryable
                    else:
                        # Non-retryable error, raise immediately
                        raise RuntimeError(
                            f"Error in {operation_name}: {str(e)}"
                        ) from e

                # Check if this is a retryable status code
                if status_code not in retryable_status_codes:
                    # Non-retryable error, raise immediately
                    raise RuntimeError(f"Error in {operation_name}: {str(e)}") from e

                # For 500 and 502, limit retries to max_retries_for_500_502
                if status_code in {500, 502} and attempt >= max_retries_for_500_502 - 1:
                    raise RuntimeError(
                        f"Error in {operation_name} after "
                        f"{max_retries_for_500_502} retries: {str(e)}"
                    ) from e

                # Calculate wait time
                if status_code == 429:
                    # Check for Retry-After header
                    retry_after = self._extract_retry_after(e)
                    if retry_after is not None:
                        wait_time = retry_after
                    else:
                        # Exponential backoff: 2^attempt seconds, max 60s
                        wait_time = min(2**attempt, 60)
                elif status_code in {503, 529}:
                    # Exponential backoff for capacity issues
                    wait_time = min(2**attempt, 60)
                else:  # 500, 502, 504
                    # Exponential backoff for transient errors
                    wait_time = min(2**attempt, 60)

                # Wait before retrying
                await asyncio.sleep(wait_time)

        # Max retries exceeded
        raise RuntimeError(
            f"Error in {operation_name} after {self.max_retries} retries: "
            f"{str(last_exception)}"
        ) from last_exception

    def __getattr__(self, name):
        """Delegate attribute access to the underlying llm object.

        This allows accessing attributes like temperature, max_tokens, etc.
        directly on the LLM instance, which will be forwarded to the
        underlying LangChain model (self.llm).
        """
        # Only delegate if self.llm exists and has the attribute
        if hasattr(self, "llm") and hasattr(self.llm, name):
            return getattr(self.llm, name)
        # If the attribute doesn't exist on self.llm, raise AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class JudgeLLM(LLMInterface):
    """Extended LLM interface that supports structured output generation.

    This interface is required for LLM implementations that can be used
    as judges, where structured output (using Pydantic models) is necessary
    for reliable evaluation results.

    Implementations: Claude, OpenAI, Gemini
    Not supported by: Llama (via Ollama)
    """

    @abstractmethod
    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        """Generate a structured response using Pydantic model.

        Args:
            message: The prompt message
            response_model: Pydantic model class to structure the response

        Returns:
            Instance of the response_model with structured data

        Raises:
            RuntimeError: If structured output generation fails
        """
        pass
