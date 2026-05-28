import asyncio
import copy
import random
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class RetryOnErrorCallback(Protocol):
    """Hook invoked by :meth:`LLMInterface._run_with_retry` after a failed attempt.

    Must match the positional contract of ``on_error`` in that method. Return
    values are merged into ``last_response_metadata`` when non-empty.

    Args:
        error: The exception from the failed attempt.
        attempt_number: 1-based index of this attempt (``1 .. max_attempts``).
        max_attempts: Total tries allowed (``max_llm_retries + 1`` for the first
            try plus each retry).
        retryable: ``False`` if the error matched no-retry substrings (call will
            raise ``LLMGenerationFailed`` without sleeping for another attempt).
        will_retry: ``True`` only if another attempt will run after backoff;
            ``False`` on non-retryable errors or on the last failed attempt.
    """

    def __call__(
        self,
        error: BaseException,
        attempt_number: int,
        max_attempts: int,
        retryable: bool,
        will_retry: bool,
    ) -> Optional[Dict[str, Any]]: ...


class LLMGenerationFailed(Exception):
    """Raised when an LLM call fails after all retry attempts are exhausted."""


class Role(Enum):
    """Role of the LLM in a conversation."""

    PERSONA = "persona"  # User roleplaying as a human seeking help
    PROVIDER = "provider"  # Chatbot providing support
    JUDGE = "judge"  # Judge role, used for judge operations


# Default prompt sent to the LLM when starting a conversation (no first_message set).
DEFAULT_START_PROMPT = "Start the conversation based on the system prompt"


class LLMInterface(ABC):
    """Abstract base class for LLM implementations.

    Provides basic text generation capabilities. All LLM implementations
    must support basic text generation and system prompt management.

    When conversation history is empty:
    - first_message: If set, return this string as the first turn (no LLM call).
    - start_prompt: If first_message is not set, use this as the prompt to
      the LLM to generate the first turn. Defaults to DEFAULT_START_PROMPT.
    """

    def __init__(
        self,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        first_message: Optional[str] = None,
        start_prompt: Optional[str] = None,
        *,
        max_llm_retries: int = 3,
        retry_base_delay_seconds: float = 0.75,
        retry_max_delay_seconds: float = 8.0,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt or ""
        self.first_message = first_message
        self.start_prompt = (
            start_prompt if start_prompt is not None else DEFAULT_START_PROMPT
        )
        # Retries after the first attempt (max_llm_retries=3 => 4 total attempts).
        self.max_llm_retries = max_llm_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self._last_response_metadata: Dict[str, Any] = {}
        self.conversation_id = self.create_conversation_id()

    @property
    def last_response_metadata(self) -> Dict[str, Any]:
        """Metadata from the last generate_response call. Returns a deep copy so
        callers cannot mutate internal state (including nested dicts like usage).
        """
        return copy.deepcopy(self._last_response_metadata)

    @last_response_metadata.setter
    def last_response_metadata(self, value: Optional[Dict[str, Any]]) -> None:
        """Set metadata; use _last_response_metadata for in-place updates."""
        self._last_response_metadata = value or {}

    def create_conversation_id(self) -> str:
        """Create a new unique conversation id.

        Used at init and when the API does not return one in response metadata.
        Subclasses may override to use a different id format.
        """
        return str(uuid.uuid4())

    def _update_conversation_id_from_metadata(self) -> None:
        """If the API returned a conversation_id in response metadata, use it.

        Call after generate_response once _last_response_metadata is set.
        APIs that ignore our request conversation_id but return their own
        will overwrite self.conversation_id here.
        """
        cid = (self._last_response_metadata or {}).get("conversation_id")
        if cid is not None and cid != self.conversation_id and cid != "":
            self.conversation_id = cid

    def _no_retry_substrings(self) -> tuple[str, ...]:
        """Substrings checked against str(exception); if any match, do not retry.

        Override on concrete LLMs with provider-specific quota/billing/auth phrases.
        """
        return ()

    @staticmethod
    def _error_matches_no_retry_substrings(
        err: BaseException, substrings: tuple[str, ...]
    ) -> bool:
        """Return True if ``str(err)`` contains any entry in ``substrings``.

        Used by :meth:`_run_with_retry` to classify errors as non-retryable when
        the merged no-retry list (instance override plus per-call) matches part
        of the exception message (e.g. quota or billing strings).
        """
        err_str = str(err)
        return any(sub in err_str for sub in substrings)

    def _compute_retry_delay_seconds(self, attempt_number: int) -> float:
        """Compute full-jitter exponential backoff delay for next retry."""
        raw_delay = self.retry_base_delay_seconds * (2 ** (attempt_number - 1))
        capped_delay = min(raw_delay, self.retry_max_delay_seconds)
        return random.uniform(0.0, capped_delay)

    async def _run_with_retry(
        self,
        get_coro: Callable[[], Awaitable[R]],
        *,
        provider: Optional[str] = None,
        no_retry_substrings: tuple[str, ...] = (),
        on_error: Optional[RetryOnErrorCallback] = None,
    ) -> R:
        """Run an async call with retries.

        If the error matches no_retry_substrings, do not retry; raise
        LLMGenerationFailed from that error so callers (e.g. conversation runner)
        can skip the job without crashing the batch. On exhaustion after retries,
        raise LLMGenerationFailed from the last error.
        """
        max_attempts = self.max_llm_retries + 1
        merged_no_retry = (*self._no_retry_substrings(), *no_retry_substrings)
        last_exception: Optional[BaseException] = None
        for attempt in range(max_attempts):
            attempt_number = attempt + 1
            # Clear stale metadata so failed calls cannot inherit prior success state.
            self.last_response_metadata = {}
            try:
                return await get_coro()
            except LLMGenerationFailed as e:
                # If callers already set error metadata in _invoke, preserve it;
                # otherwise write a standardized failure payload here.
                if not self._last_response_metadata:
                    self._set_response_metadata(
                        provider or "unknown",
                        error=str(e),
                        attempt=attempt_number,
                        max_attempts=max_attempts,
                        retryable=False,
                        will_retry=False,
                    )
                raise
            except Exception as e:
                last_exception = e
                retryable = not self._error_matches_no_retry_substrings(
                    e, merged_no_retry
                )
                will_retry = retryable and attempt < self.max_llm_retries

                self._set_response_metadata(
                    provider or "unknown",
                    error=str(e),
                    attempt=attempt_number,
                    max_attempts=max_attempts,
                    retryable=retryable,
                    will_retry=will_retry,
                )
                if on_error is not None:
                    extra = on_error(
                        e, attempt_number, max_attempts, retryable, will_retry
                    )
                    if extra:
                        self._last_response_metadata.update(extra)

                if not retryable:
                    raise LLMGenerationFailed(f"LLM error (non-retryable): {e}") from e
                if attempt == self.max_llm_retries:
                    break
                retry_delay_seconds = self._compute_retry_delay_seconds(attempt_number)
                self._last_response_metadata["retry_delay_seconds"] = round(
                    retry_delay_seconds, 3
                )
                await asyncio.sleep(retry_delay_seconds)
        assert last_exception is not None
        raise LLMGenerationFailed(
            f"LLM call failed after {max_attempts} attempts: {last_exception}"
        ) from last_exception

    def _set_response_metadata(self, provider: str, **extra: Any) -> None:
        """Set last_response_metadata with common fields; pass extra keys as kwargs.

        Always sets: response_id, model, provider, role, timestamp, usage.
        Override or add keys via kwargs (e.g. error=..., response_time_seconds=...).
        """
        self.last_response_metadata = {
            "response_id": extra.pop("response_id", None),
            "model": extra.pop("model", getattr(self, "model_name", None)),
            "provider": provider,
            "role": self.role.value,
            "timestamp": datetime.now().isoformat(),
            "usage": extra.pop("usage", {}),
            **extra,
        }

    def get_initial_prompt_turns(self) -> List[Dict[str, Any]]:
        """Build the initial turn(s) used to prompt the LLM when history is empty.

        Returns a list of dicts (e.g. [{"turn": 0, "response": "<start_prompt>"}])
        that can be passed to the message builder
        and then to the LLM. Used by raw LLM implementations that delegate from
        start_conversation() to generate_response(...). Subclasses may override.

        Returns:
            List of dicts representing the initial conversation turn(s)
            (e.g. [{"turn": 0, "response": "<start prompt text>"}]).
        """
        return [{"turn": 0, "response": self.start_prompt}]

    def get_first_turn_input_message(self) -> Optional[str]:
        """Return the input message used for the first turn, for metadata only.

        Called by the simulator after start_conversation() to record what prompt
        was sent to the LLM. Returns None if the first turn used first_message
        (no LLM call); otherwise returns the start_prompt text actually used.

        Subclasses that use custom logic in start_conversation() may override
        this (or set _first_turn_input in start_conversation) so metadata
        matches what was really sent.
        """
        if self.first_message is not None:
            return None
        return self.start_prompt

    @abstractmethod
    async def start_conversation(self) -> str:
        """Produce the first response of the conversation.

        Called by the simulator on turn 0. When first_message is set, return
        it (and set metadata) without calling the API. Otherwise, raw LLM
        implementations may call generate_response(self.get_initial_prompt_turns());
        service-based clients may call their own start endpoint (e.g. POST
        /start_conversation) and return the returned message.

        Returns:
            str: The first response text. Metadata in self.last_response_metadata
                (getter returns a copy so callers need not copy).
        """
        pass

    @abstractmethod
    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response based on conversation history.

        When used by the simulator, conversation_history is only passed for
        turns 1+ (turn 0 uses start_conversation()). Callers may still pass
        empty history for backward compatibility; implementations can delegate
        to await self.start_conversation() in that case if desired.

        Args:
            conversation_history: List of previous conversation turns.
                When the simulator calls this, history is non-empty with turns
                1, 2, … (first response is turn 1). Each turn: 'turn', 'speaker',
                'response'. If start_conversation() delegates here, it may pass
                get_initial_prompt_turns() (turn=0, 'response' only; no
                'speaker'). See llm_clients/claude_llm.py for an example.

        Returns:
            str: The response text. Metadata in self.last_response_metadata
                (getter returns a copy so callers need not copy).

        Note:
            For API thread/session identification, use self.conversation_id
            (set at init; send as request metadata). If your API returns a
            conversation_id in response metadata, call
            self._update_conversation_id_from_metadata() after setting
            _last_response_metadata to overwrite.
        """
        pass

    @abstractmethod
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        pass

    def _post_process_response(self, response: str) -> str:
        """Post-process a raw LLM response before it enters the conversation history.

        No-op by default. Override in subclasses to strip provider-specific
        artifacts (e.g. trailing metadata tags appended by the backend).
        """
        return response

    @property
    def custom_termination_signals(self) -> List[str]:
        """Per-implementation early-termination signals.

        Return a list of signal strings. When any signal is found in this
        speaker's response, the conversation ends early.
        Returns [] by default (no custom signals).
        Override in subclasses to add provider-specific custom termination detection.
        Each signal is only checked against the response of the LLM that declares it.
        Mostly meant for provider LLM classes to allow special termination handling.
        """
        return []

    def _should_discard_response(self, extracted_signals: List[str]) -> bool:
        """Return True if the response that produced these signals should be discarded.

        When True, the simulator removes the turn from conversation history entirely
        rather than marking it as early_termination. Override in subclasses for
        provider-specific error responses that should not appear in transcripts.
        """
        return False

    def _extract_signals(self, response: str) -> List[str]:
        """Extract custom termination signals present in a raw response.

        Called by the simulator on the raw response, before _post_process_response
        strips provider artifacts. Returns the subset of custom_termination_signals
        found in the response.

        Override in subclasses for custom extraction logic (e.g. structured parsing).
        The default implementation scans for each entry in custom_termination_signals.
        """
        import re

        found = []
        for signal in self.custom_termination_signals:
            if re.search(re.escape(signal), response, re.IGNORECASE):
                found.append(signal)
        return found

    @property
    def first_speaker(self) -> Optional["Role"]:
        """The role that should speak first in this session.

        Returns None to defer to the runner's global persona_speaks_first setting.
        Override in subclasses where the session type dictates speaker order
        (e.g. INTAKE sessions are always provider-initiated).
        """
        return None

    def prepare_sessions(self, session_types: List[str]) -> List[str]:
        """Normalize the session list before running.

        Override to enforce provider-specific prerequisites (e.g. prepend INTAKE).
        Default: return unchanged.
        """
        return session_types

    async def enter_session(self, session_type: str) -> None:
        """Prepare the provider for session_type. If a session is already active,
        finish it first. Called once at the start of each session iteration
        (including the first).
        No-op by default."""
        pass

    async def setup(self) -> None:
        """Set up any resources needed before the conversation starts.

        Called by the simulator once before the turn loop begins.
        Subclasses that need async initialization (e.g. session creation,
        auth token acquisition) should override this method.
        Default implementation does nothing.
        """
        pass

    async def cleanup(self) -> None:
        """Clean up any resources used by this LLM instance.

        Subclasses that need cleanup (like AzureLLM) should override this method.
        Default implementation does nothing.
        """
        pass

    def __getattr__(self, name):
        """Delegate attribute access to the underlying llm object.

        This allows accessing attributes like temperature, max_tokens, etc.
        directly on the LLM instance, which will be forwarded to the
        underlying LangChain model (self.llm).
        """
        # Check if self.llm exists by looking in __dict__ to avoid recursion
        # Only delegate if self.llm exists and has the attribute
        if "llm" in self.__dict__ and hasattr(self.llm, name):
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

    Implementations: Claude, OpenAI, Gemini, Azure
    Not supported by: Ollama
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
            LLMGenerationFailed: If structured output generation fails after retries
        """
        pass
