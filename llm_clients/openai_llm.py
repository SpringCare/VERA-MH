from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import Optional, Dict, Any
from .llm_interface import LLMInterface
from .config import Config
import time
from datetime import datetime

class OpenAILLM(LLMInterface):
    """OpenAI implementation using LangChain."""
    
    def __init__(
        self, 
        name: str, 
        system_prompt: Optional[str] = None, 
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, system_prompt)
        
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_openai_config()["model"]
        
        # Get default config and allow kwargs to override
        config = Config.get_openai_config()
        llm_params = {
            "openai_api_key": Config.OPENAI_API_KEY,
            "model": self.model_name,
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 1000)
        }
        
        # Override with any provided kwargs
        llm_params.update(kwargs)
        self.llm = ChatOpenAI(**llm_params)
        
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
            
            
            # Extract metadata from response - capturing all available fields
            self.last_response_metadata = {
                "response_id": getattr(response, 'id', None),
                "model": self.model_name,  # Will be updated from response_metadata if available
                "provider": "openai",
                "timestamp": datetime.now().isoformat(),
                "response_time_seconds": round(end_time - start_time, 3),
                "usage": {},
                "finish_reason": None,
                "additional_kwargs": {},
                "system_fingerprint": None,
                "logprobs": None,
                "response": response
            }
            
            # Extract additional_kwargs if available
            if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
                self.last_response_metadata["additional_kwargs"] = dict(response.additional_kwargs)
            
            # Extract response_metadata if available
            if hasattr(response, 'response_metadata') and response.response_metadata:
                metadata = response.response_metadata
                
                # Update model name from response metadata
                if 'model_name' in metadata:
                    self.last_response_metadata["model"] = metadata['model_name']
                
                # Extract token usage from response_metadata
                if 'token_usage' in metadata:
                    token_usage = metadata['token_usage']
                    self.last_response_metadata["usage"] = {
                        "prompt_tokens": token_usage.get('prompt_tokens', 0),
                        "completion_tokens": token_usage.get('completion_tokens', 0),
                        "total_tokens": token_usage.get('total_tokens', 0)
                    }
                
                # Extract other metadata fields
                self.last_response_metadata["finish_reason"] = metadata.get('finish_reason')
                self.last_response_metadata["system_fingerprint"] = metadata.get('system_fingerprint')
                self.last_response_metadata["logprobs"] = metadata.get('logprobs')
                
                # Store raw response_metadata
                self.last_response_metadata["raw_response_metadata"] = dict(metadata)
            
            # Extract usage_metadata if available (separate from response_metadata)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_meta = response.usage_metadata
                # Merge with existing usage info, preferring usage_metadata values
                self.last_response_metadata["usage"].update({
                    "input_tokens": usage_meta.get('input_tokens', 0),
                    "output_tokens": usage_meta.get('output_tokens', 0),
                    "total_tokens": usage_meta.get('total_tokens', 0)
                })
                # Store raw usage_metadata
                self.last_response_metadata["raw_usage_metadata"] = dict(usage_meta)
             
            return response.content
        except Exception as e:
            # Store error metadata
            self.last_response_metadata = {
                "response_id": None,
                "model": self.model_name,
                "provider": "metadata",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "usage": {},
                "additional_kwargs": {},
                "system_fingerprint": None,
                "logprobs": None
            }
            return f"Error generating response: {str(e)}"
    
    def get_last_response_metadata(self) -> Dict[str, Any]:
        """Get metadata from the last response."""
        return self.last_response_metadata.copy()
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt