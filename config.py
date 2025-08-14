import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For future use
    
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