import copy
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Default system prompt when none is provided (e.g. for provider/agent LLMs)
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


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
    ):
        self.name = name
        self.role = role
        self.system_prompt = (
            system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        )
        self.first_message = first_message
        self.start_prompt = (
            start_prompt if start_prompt is not None else DEFAULT_START_PROMPT
        )
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
            RuntimeError: If structured output generation fails
        """
        pass
