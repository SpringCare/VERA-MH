from langchain_community.llms import Ollama
from langchain.schema import HumanMessage, SystemMessage
from typing import Optional, Dict, Any
from .llm_interface import LLMInterface
from .config import Config

class LlamaLLM(LLMInterface):
    """Llama implementation using LangChain with Ollama."""
    
    def __init__(
        self, 
        name: str, 
        system_prompt: Optional[str] = None, 
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, system_prompt)
        
        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_llama_config()["model"]
        
        # Get default config and allow kwargs to override
        config = Config.get_llama_config()
        llm_params = {
            "model": self.model_name,
            "temperature": config.get("temperature", 0.7),
            "base_url": config.get("base_url", "http://localhost:11434")  # Default Ollama URL
        }
        
        # Override with any provided kwargs
        llm_params.update(kwargs)
        self.llm = Ollama(**llm_params)
    
    async def generate_response(self, message: str) -> str:
        """Generate a response to the given message asynchronously."""
        try:
            # Format the message with system prompt if available
            full_message = message
            if self.system_prompt:
                full_message = f"System: {self.system_prompt}\n\nHuman: {message}\n\nAssistant:"
            
            # Ollama doesn't have native async support in langchain-community
            # So we'll use the synchronous version
            response = self.llm.invoke(full_message)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt