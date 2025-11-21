from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class LLMInterface(ABC):
    """Abstract base class for LLM implementations."""

    def __init__(self, name: str, system_prompt: Optional[str] = None):
        self.name = name
        self.system_prompt = system_prompt or ""

    @abstractmethod
    async def generate_response(
        self, message: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a response to the given message asynchronously.

        Returns:
            Tuple of (response_text, metadata_dict)
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
