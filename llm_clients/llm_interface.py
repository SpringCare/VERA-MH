from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMInterface(ABC):
    """Abstract base class for LLM implementations.

    Provides basic text generation capabilities. All LLM implementations
    must support basic text generation and system prompt management.
    """

    def __init__(self, name: str, system_prompt: Optional[str] = None):
        self.name = name
        self.system_prompt = system_prompt or ""

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
