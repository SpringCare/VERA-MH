from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Optional, Dict, Any
from .llm_interface import LLMInterface
from .config import Config

class GeminiLLM(LLMInterface):
    """Gemini implementation using LangChain."""
    
    def __init__(
        self, 
        name: str, 
        system_prompt: Optional[str] = None, 
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, system_prompt)
        
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_gemini_config()["model"]
        
        # Get default config and allow kwargs to override
        config = Config.get_gemini_config()
        llm_params = {
            "google_api_key": Config.GOOGLE_API_KEY,
            "model": self.model_name,
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 1000)
        }
        
        # Override with any provided kwargs
        llm_params.update(kwargs)
        self.llm = ChatGoogleGenerativeAI(**llm_params)
    
    async def generate_response(self, message: str) -> str:
        """Generate a response to the given message asynchronously."""
        messages = []
        
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        messages.append(HumanMessage(content=message))
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt