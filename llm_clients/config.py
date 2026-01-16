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
    # Note: Llama via Ollama doesn't require an API key

    @classmethod
    def get_claude_config(cls) -> Dict[str, Any]:
        """Get default Claude model name.

        Returns only the model name. Runtime parameters (temperature, max_tokens)
        should be passed explicitly via CLI arguments.
        """
        return {"model": "claude-3-5-sonnet-20241022"}

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
    def get_llama_config(cls) -> Dict[str, Any]:
        """Get default Llama model name.

        Returns only the model name. Runtime parameters (temperature, max_tokens)
        should be passed explicitly via CLI arguments. The base_url for Ollama
        is hardcoded in llama_llm.py for connectivity purposes.
        """
        return {"model": "llama3:8b"}
