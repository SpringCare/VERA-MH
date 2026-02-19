"""
LLM Clients Package - Shared LLM abstraction layer
Provides unified interface for different LLM providers
  - OpenAI (gpt-*)
  - Claude (claude-*)
  - Gemini (gemini-*)
  - Azure (azure-*)
  - Ollama (ollama-*)
  - Custom endpoint (endpoint, endpoint-*)
"""

from .config import Config
from .llm_factory import LLMFactory
from .llm_interface import JudgeLLM, LLMInterface, Role

__all__ = ["LLMInterface", "JudgeLLM", "LLMFactory", "Config", "Role"]
