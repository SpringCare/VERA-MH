"""Configuration module for LLM clients.

Design Philosophy:
- Model-specific parameters (temperature, max_tokens) should be passed at
  runtime via CLI
- Config class only provides fallback model names and API key access
- All LLM implementations rely on runtime parameters or LangChain/provider
  defaults
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration for LLM clients.

    Provides API key access and fallback model names.
    Runtime parameters should be passed via CLI arguments.
    """

    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # For Gemini
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")  # Optional

    @classmethod
    def get_claude_config(cls) -> Dict[str, Any]:
        """Get default Claude model name.

        Returns only the model name. Runtime parameters (temperature, max_tokens)
        should be passed explicitly via CLI arguments.
        """
        return {"model": "claude-sonnet-4-5-20250929"}

    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """Get default OpenAI model name.

        Returns only the model name. Runtime parameters (temperature, max_tokens)
        should be passed explicitly via CLI arguments.
        """
        return {"model": "gpt-4"}

    @classmethod
    def get_gemini_config(cls) -> Dict[str, Any]:
        """Get default Gemini model name.

        Returns only the model name. Runtime parameters (temperature, max_tokens)
        should be passed explicitly via CLI arguments.
        """
        return {"model": "gemini-1.5-pro"}

    @classmethod
    def get_azure_config(cls) -> Dict[str, Any]:
        """Get default Azure model name.

        Returns only the model name. Runtime parameters (temperature, max_tokens)
        should be passed explicitly via CLI arguments. The endpoint and API key
        are loaded from environment variables.
        """
        return {"model": "azure-gpt-4"}

    @classmethod
    def get_ollama_config(cls) -> Dict[str, Any]:
        """Get default Ollama configuration.

        Returns model name and base_url. Runtime parameters (temperature, max_tokens)
        should be passed explicitly via CLI arguments. This is a general config
        for any Ollama-hosted model (llama, phi4, mistral, etc.).
        """
        return {
            "model": "llama3:8b",
            "base_url": "http://localhost:11434",  # Default Ollama URL
        }
