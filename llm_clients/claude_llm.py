from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from typing import Optional, Dict, Any
from .llm_interface import LLMInterface
from .config import Config
import time
from datetime import datetime

class ClaudeLLM(LLMInterface):
    """Claude implementation using LangChain."""
    
    def __init__(
        self, 
        name: str, 
        system_prompt: Optional[str] = None, 
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, system_prompt)
        
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_claude_config()["model"]
        
        # Get default config and allow kwargs to override
        config = Config.get_claude_config()
        llm_params = {
            "anthropic_api_key": Config.ANTHROPIC_API_KEY,
            "model": self.model_name,
            # "temperature": config.get("temperature", 0.7),
            # "max_tokens": config.get("max_tokens", 1000)
        }
        
        # Override with any provided kwargs
        llm_params.update(kwargs)
        self.llm = ChatAnthropic(**llm_params)
        
        # Store metadata from last response
        self.last_response_metadata: Dict[str, Any] = {}
    
    async def generate_response(self, message: Optional[str] = None) -> str:
        """Generate a response to the given message asynchronously."""
        messages = []
        
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        messages.append(HumanMessage(content=message))
        
        try:
            start_time = time.time()
            response = await self.llm.ainvoke(messages)
            end_time = time.time()
            
            # Extract metadata from response
            self.last_response_metadata = {
                "response_id": getattr(response, 'id', None),
                "model": getattr(response.response_metadata, 'model', self.model_name) if hasattr(response, 'response_metadata') else self.model_name,
                "provider": "claude",
                "timestamp": datetime.now().isoformat(),
                "response_time_seconds": round(end_time - start_time, 3),
                "usage": {},
                "stop_reason": None,
                "response": response
            }
            
            # Extract usage information if available
            if hasattr(response, 'response_metadata') and response.response_metadata:
                metadata = response.response_metadata
                
                # Extract token usage
                if 'usage' in metadata:
                    usage = metadata['usage']
                    self.last_response_metadata["usage"] = {
                        "input_tokens": usage.get('input_tokens', 0),
                        "output_tokens": usage.get('output_tokens', 0),
                        "total_tokens": usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
                    }
                
                # Extract stop reason
                self.last_response_metadata["stop_reason"] = metadata.get('stop_reason')
                
                # Store raw metadata
                self.last_response_metadata["raw_metadata"] = dict(metadata)
            
            return response.content
        except Exception as e:
            # Store error metadata
            self.last_response_metadata = {
                "response_id": None,
                "model": self.model_name,
                "provider": "claude",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "usage": {},
            }
            return f"Error generating response: {str(e)}"
    
    def get_last_response_metadata(self) -> Dict[str, Any]:
        """Get metadata from the last response."""
        return self.last_response_metadata.copy()
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt