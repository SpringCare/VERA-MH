from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

class LLMInterface(ABC):
    """Abstract base class for LLM implementations."""
    
    def __init__(self, name: str, system_prompt: Optional[str] = None):
        self.name = name
        self.system_prompt = system_prompt or ""
    
    @abstractmethod
    async def generate_response(self, message: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
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