from typing import Optional, Dict, Any
from .llm_interface import LLMInterface

class LLMFactory:
    """Factory class for creating LLM instances based on model name/version."""
    
    @staticmethod
    def create_llm(
        model_name: str, 
        name: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMInterface:
        """
        Create an LLM instance based on the model name.
        
        Args:
            model_name: The model identifier (e.g., "claude-3-5-sonnet-20241022", "gpt-4")
            name: Display name for this LLM instance
            system_prompt: Optional system prompt
            **kwargs: Additional model-specific parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLMInterface instance
        """
        # Normalize model name to determine provider
        model_lower = model_name.lower()
        print(f"creating llm with {model_name}")
        
        # Filter out non-model-specific parameters
        model_params = {k: v for k, v in kwargs.items() 
                       if k not in ['model', 'name', 'prompt_name', 'system_prompt']}
        
        if "claude" in model_lower:
            from .claude_llm import ClaudeLLM
            return ClaudeLLM(name, system_prompt, model_name, **model_params)
        elif "gpt" in model_lower or "openai" in model_lower:
            from .openai_llm import OpenAILLM
            return OpenAILLM(name, system_prompt, model_name, **model_params)
        elif "gemini" in model_lower or "google" in model_lower:
            from .gemini_llm import GeminiLLM
            return GeminiLLM(name, system_prompt, model_name, **model_params)
        elif "llama" in model_lower or "ollama" in model_lower:
            from .llama_llm import LlamaLLM
            return LlamaLLM(name, system_prompt, model_name, **model_params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    @staticmethod
    def get_supported_models() -> Dict[str, list]:
        """Get a dictionary of supported model providers and their models."""
        return {
            "claude": [
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229", 
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ],
            "openai": [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ],
            "gemini": [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-pro"
            ],
            "llama": [
                "llama2:7b",
                "llama2:13b",
                "llama3:8b",
                "llama3:70b"
            ]
        }