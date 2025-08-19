import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # For Gemini
    # Note: Llama via Ollama doesn't require an API key
    
    # Default model configurations
    MODELS_CONFIG = {
        "claude-3-5-sonnet-20241022": {
            "provider": "anthropic",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "claude-3-opus-20240229": {
            "provider": "anthropic", 
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "claude-3-sonnet-20240229": {
            "provider": "anthropic",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "claude-3-haiku-20240307": {
            "provider": "anthropic",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "gpt-4": {
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "gpt-4-turbo": {
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "gpt-3.5-turbo": {
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "gemini-1.5-pro": {
            "provider": "google",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "gemini-1.5-flash": {
            "provider": "google",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "gemini-pro": {
            "provider": "google",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "llama2:7b": {
            "provider": "ollama",
            "temperature": 0.7,
            "base_url": "http://localhost:11434"
        },
        "llama2:13b": {
            "provider": "ollama",
            "temperature": 0.7,
            "base_url": "http://localhost:11434"
        },
        "llama3:8b": {
            "provider": "ollama",
            "temperature": 0.7,
            "base_url": "http://localhost:11434"
        },
        "llama3:70b": {
            "provider": "ollama",
            "temperature": 0.7,
            "base_url": "http://localhost:11434"
        }
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return cls.MODELS_CONFIG.get(model_name, {
            "provider": "unknown",
            "temperature": 0.7,
            "max_tokens": 1000
        })
    
    @classmethod
    def get_claude_config(cls) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return {
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    
    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """Get default OpenAI configuration."""
        return {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    
    @classmethod
    def get_gemini_config(cls) -> Dict[str, Any]:
        """Get default Gemini configuration."""
        return {
            "model": "gemini-1.5-pro",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    
    @classmethod
    def get_llama_config(cls) -> Dict[str, Any]:
        """Get default Llama configuration."""
        return {
            "model": "llama3:8b",
            "temperature": 0.7,
            "base_url": "http://localhost:11434"
        }