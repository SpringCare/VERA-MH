import time
from datetime import datetime
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Config
from .llm_interface import LLMInterface


class GeminiLLM(LLMInterface):
    """Gemini implementation using LangChain."""

    def __init__(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
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
            # "temperature": config.get("temperature", 0.7),
            # "max_tokens": config.get("max_tokens", 1000)
        }

        # Override with any provided kwargs
        llm_params.update(kwargs)
        self.llm = ChatGoogleGenerativeAI(**llm_params)

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
                "response_id": getattr(response, "id", None),
                "model": (
                    getattr(response.response_metadata, "model_name", self.model_name)
                    if hasattr(response, "response_metadata")
                    else self.model_name
                ),
                "provider": "gemini",
                "timestamp": datetime.now().isoformat(),
                "response_time_seconds": round(end_time - start_time, 3),
                "usage": {},
                "finish_reason": None,
                "response": response,
            }

            # Extract usage information if available
            if hasattr(response, "response_metadata") and response.response_metadata:
                metadata = response.response_metadata

                # Extract token usage - Gemini may have different structure
                if "usage_metadata" in metadata:
                    usage = metadata["usage_metadata"]
                    self.last_response_metadata["usage"] = {
                        "prompt_token_count": usage.get("prompt_token_count", 0),
                        "candidates_token_count": usage.get(
                            "candidates_token_count", 0
                        ),
                        "total_token_count": usage.get("total_token_count", 0),
                    }
                elif "token_usage" in metadata:
                    # Fallback structure
                    usage = metadata["token_usage"]
                    self.last_response_metadata["usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                # Extract finish reason
                self.last_response_metadata["finish_reason"] = metadata.get(
                    "finish_reason"
                )

                # Store raw metadata
                self.last_response_metadata["raw_metadata"] = dict(metadata)

            return response.content
        except Exception as e:
            # Store error metadata
            self.last_response_metadata = {
                "response_id": None,
                "model": self.model_name,
                "provider": "gemini",
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
