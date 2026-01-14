from typing import Any, Dict, List, Optional

from langchain_community.llms import Ollama

from utils.conversation_utils import format_conversation_as_string

from .config import Config
from .llm_interface import LLMInterface


class LlamaLLM(LLMInterface):
    """Llama implementation using LangChain with Ollama.

    Note: This implementation does not support structured output generation
    and therefore cannot be used as a judge. For judge operations, use
    Claude, OpenAI, or Gemini models.
    """

    def __init__(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, system_prompt)

        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_llama_config()["model"]

        # Get default config and allow kwargs to override
        config = Config.get_llama_config()
        llm_params = {
            "model": self.model_name,
            "temperature": config.get("temperature", 0.7),
            "base_url": config.get(
                "base_url", "http://localhost:11434"
            ),  # Default Ollama URL
        }

        # Override with any provided kwargs
        llm_params.update(kwargs)
        self.llm = Ollama(**llm_params)

    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response based on conversation history.

        Args:
            conversation_history: Optional list of previous conversation turns
        """
        try:
            # Build full message using utility function
            full_message = format_conversation_as_string(
                conversation_history=conversation_history,
                system_prompt=self.system_prompt,
            )

            # Ollama doesn't have native async support in langchain-community
            # So we'll use the synchronous version
            response = self.llm.invoke(full_message)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
