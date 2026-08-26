"""Configuration module for LLM clients.

Design Philosophy:
- Model-specific parameters (temperature, max_tokens) should be passed at
  runtime via CLI
- Config class only provides fallback model names and API key access
- All LLM implementations rely on runtime parameters or LangChain/provider
  defaults
"""

import os
from typing import Any, Dict, Optional

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

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

    ENDPOINT_API_KEY = os.getenv("ENDPOINT_API_KEY", None)
    ENDPOINT_URL = os.getenv("ENDPOINT_URL", None)
    ENDPOINT_START_URL = os.getenv("ENDPOINT_START_URL", None)

    # Optional base URL overrides. Set these to send Claude/OpenAI/Gemini traffic
    # through a proxy or gateway (e.g. LiteLLM) instead of the provider's public
    # API. API_BASE_URL applies to every provider; the per-provider vars win when
    # both are set. Leave unset to call the providers directly.
    API_BASE_URL = os.getenv("API_BASE_URL", None)
    ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", None)
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
    GOOGLE_BASE_URL = os.getenv("GOOGLE_BASE_URL", None)

    @classmethod
    def get_base_url(cls, provider: str) -> Optional[str]:
        """Get the optional base URL for ``provider``, or None to use its default.

        ``provider`` is one of ``anthropic``, ``openai``, ``google``. Falls back to
        ``API_BASE_URL`` when no provider-specific override is set. Trailing
        slashes are stripped so callers can append paths safely.
        """
        provider_urls = {
            "anthropic": cls.ANTHROPIC_BASE_URL,
            "openai": cls.OPENAI_BASE_URL,
            "google": cls.GOOGLE_BASE_URL,
        }
        if provider not in provider_urls:
            raise ValueError(
                f"Unknown provider {provider!r}; expected one of "
                f"{', '.join(sorted(provider_urls))}"
            )
        url = provider_urls[provider] or cls.API_BASE_URL
        return url.rstrip("/") if url else None

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
        return {"model": "gpt-5.2"}

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
        return {"model": "azure-gpt-5.2"}

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

    @classmethod
    def get_endpoint_config(cls) -> Dict[str, Any]:
        """Get custom endpoint configuration.

        Returns base_url (no /api/chat path), api_key, and default model.
        Runtime parameters can override via kwargs.
        Raises ValueError if ENDPOINT_API_KEY or ENDPOINT_URL
        are not set in the environment.
        ENDPOINT_START_URL is optional and can be set to None.
        """
        missing = []
        if cls.ENDPOINT_API_KEY is None:
            missing.append("ENDPOINT_API_KEY")
        if cls.ENDPOINT_URL is None:
            missing.append("ENDPOINT_URL")
        if cls.ENDPOINT_START_URL is None:
            print("ENDPOINT_START_URL is not set in the environment.")
        if missing:
            raise ValueError(
                "Custom endpoint requires these environment variables: "
                f"{', '.join(missing)}"
            )
        return {
            "base_url": cls.ENDPOINT_URL,
            "api_key": cls.ENDPOINT_API_KEY,
            "start_url": cls.ENDPOINT_START_URL,
            "model": "phi4",
        }
