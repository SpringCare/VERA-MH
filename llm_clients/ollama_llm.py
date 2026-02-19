import time
from typing import Any, Dict, List, Optional

from langchain_ollama import OllamaLLM as LangChainOllamaLLM

from utils.conversation_utils import format_conversation_as_string
from utils.debug import debug_print

from .config import Config
from .llm_interface import LLMInterface, Role


class OllamaLLM(LLMInterface):
    """General Ollama implementation using LangChain.

    This implementation can work with any model hosted on Ollama
    Examples: llama, phi4, mistral, etc.

    Note: This implementation does not support structured output generation
    and therefore cannot be used as a judge. For judge operations, use
    Claude, OpenAI, or Gemini models.
    """

    def __init__(
        self,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
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

        # Use provided model name or fall back to config default
        self.model_name = (model_name or Config.get_ollama_config()["model"]).replace(
            "ollama-", ""
        )  # Remove "ollama-" prefix if present

        # Get default config and allow kwargs to override
        llm_params = {
            "model": self.model_name,
            "base_url": base_url
            or Config.get_ollama_config().get("base_url", "http://localhost:11434"),
        }

        # Store max_tokens if provided (for logging), but map to num_predict for Ollama
        # Ollama uses 'num_predict' instead of 'max_tokens'
        # If both are provided, num_predict takes precedence
        if "num_predict" in kwargs:
            llm_params["num_predict"] = kwargs.pop("num_predict")
            self.max_tokens = llm_params["num_predict"]
        elif "max_tokens" in kwargs:
            llm_params["num_predict"] = kwargs.pop("max_tokens")
            self.max_tokens = llm_params["num_predict"]
        else:
            self.max_tokens = None

        # Store temperature for logging (if provided)
        if "temperature" in kwargs:
            llm_params["temperature"] = kwargs.pop("temperature")
            self.temperature = llm_params["temperature"]
        else:
            self.temperature = None

        # Override with any remaining provided kwargs
        llm_params.update(kwargs)
        self.llm = LangChainOllamaLLM(**llm_params)

        # If max_tokens wasn't set, try to get it from the llm object
        if self.max_tokens is None:
            self.max_tokens = getattr(self.llm, "num_predict", None)
        # If temperature wasn't set, try to get it from the llm object
        if self.temperature is None:
            self.temperature = getattr(self.llm, "temperature", None)

    async def start_conversation(self) -> str:
        """Produce the first response:
        - static first_message if set, or
        - LLM with start_prompt if first_message is not set.
        """
        if self.first_message is not None:
            self._set_response_metadata("ollama", static_first_message=True)
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
        try:
            if not conversation_history or len(conversation_history) == 0:
                return await self.start_conversation()

            # Build full message using utility function
            full_message = format_conversation_as_string(
                self.role,
                conversation_history=conversation_history,
                system_prompt=self.system_prompt,
            )

            # Debug: Print message being sent to LLM
            debug_print(
                f"\n[DEBUG {self.name} - {self.role.value}] Message sent to LLM:"
            )
            preview = full_message[:100]
            content_preview = (
                preview + "..." if len(full_message) > 100 else full_message
            )
            debug_print(f"  {content_preview}")

            start_time = time.time()
            # Use ainvoke for async call - BaseLLM.ainvoke returns a string directly
            response_text = await self.llm.ainvoke(full_message)
            end_time = time.time()

            self._set_response_metadata(
                "ollama",
                response_time_seconds=round(end_time - start_time, 3),
            )

            return response_text
        except Exception as e:
            self._set_response_metadata("ollama", error=str(e))
            return f"Error generating response: {str(e)}"

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
