import time
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from .config import Config
from .llm_interface import LLMInterface


class EndpointLLM(LLMInterface):
    """HTTP endpoint implementation for custom chat API.

    This implementation connects to a custom HTTP chat API endpoint that follows
    a simple request/response pattern. It maintains conversation state via
    conversation_id tracking across multiple requests.

    Example API Request:
        POST /api/chat
        Headers:
            Content-Type: application/json
            X-API-Key: {api_key}
        Body:
            {
                "messages": [{"role": "user", "content": "Hello there!"}]
            }

    Example API Response:
        {
            "model": "phi4",
            "created_at": "2026-01-08T20:09:57.11564Z",
            "message": {
                "role": "assistant",
                "content": "Hi! How can I assist you today?"
            },
            "done": true,
            "conversation_id": "1eda1c88-4421-4e65-a7f3-a8917db6dc97",
            "total_duration": 1533803209,
            ...
        }

    Note: This implementation does not support structured output generation
    and therefore cannot be used as a judge. For judge operations, use
    Claude, OpenAI, or Gemini models.

    Args:
        name: Display name for this LLM instance
        system_prompt: Optional system prompt (stored but ignored - API doesn't
                      support system messages)
        model_name: Optional model identifier (used for display/metadata purposes)
        endpoint_url: URL for the chat API endpoint (e.g., "http://0.0.0.0:8000/api/chat")
                     Can be provided via ENDPOINT_URL environment variable or kwargs
        api_key: API key for authentication (X-API-Key header)
                Can be provided via ENDPOINT_API_KEY environment variable or kwargs
        **kwargs: Additional parameters (temperature, max_tokens, etc. - may be
                 ignored by API)
    """

    def __init__(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, system_prompt)

        # Get endpoint URL from kwargs, environment variable, or default
        self.endpoint_url = endpoint_url or Config.ENDPOINT_URL

        # Validate and normalize endpoint URL
        parsed = urlparse(self.endpoint_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                f"Invalid endpoint URL: {self.endpoint_url}. "
                "Must be a valid HTTP/HTTPS URL (e.g., http://0.0.0.0:8000/api/chat)"
            )

        # Get API key from kwargs, environment variable, or default
        self.api_key = api_key or Config.ENDPOINT_API_KEY or "howdy"

        # Store model name for metadata (use endpoint URL as identifier if not provided)
        self.model_name = model_name or self.endpoint_url

        # Store any additional kwargs (API may ignore these)
        self.extra_params = kwargs

        # Store temperature and max_tokens for logging compatibility
        # Extract from kwargs if provided, otherwise set to None
        self.temperature = kwargs.get("temperature", None)
        self.max_tokens = kwargs.get("max_tokens", None)

        # Track conversation state
        self.conversation_id: Optional[str] = None

        # Store metadata from last response
        self.last_response_metadata: Dict[str, Any] = {}

        # Create HTTP client (will be reused for all requests)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

        print("Creating Endpoint LLM with parameters:")
        print(f"  Endpoint URL: {self.endpoint_url}")
        print(f"  Model: {self.model_name}")
        if self.extra_params:
            print(f"  Extra parameters: {self.extra_params}")

    async def generate_response(self, message: Optional[str] = None) -> str:
        """Generate a response to the given message asynchronously.

        The custom endpoint API maintains conversation state via conversation_id.
        On the first request, it creates a new conversation & stores the id.
        Include the conversation_id in remaining requests to continue the conversation.

        Args:
            message: The user's input message

        Returns:
            The API's response content as a string

        Raises:
            Exception: Returns error message as string if request fails
        """
        if message is None:
            message = ""

        try:
            start_time = time.time()

            # Prepare request payload
            payload: Dict[str, Any] = {
                "messages": [{"role": "user", "content": message}]
            }

            # Include conversation_id if we have one (for continuing conversation)
            if self.conversation_id is not None:
                payload["conversation_id"] = self.conversation_id

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
            }

            # assert self.endpoint_url is not None
            if self.endpoint_url is None:
                raise ValueError("Endpoint URL is not set")

            # Make HTTP request
            response = await self.client.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

            # Parse response
            response_data = response.json()
            end_time = time.time()

            # Extract response content
            if "message" in response_data and "content" in response_data["message"]:
                content = response_data["message"]["content"]
            else:
                # Fallback: try to get content directly or use full response
                content = response_data.get("content", str(response_data))

            # Extract and store conversation_id (for subsequent requests)
            if "conversation_id" in response_data:
                self.conversation_id = response_data["conversation_id"]

            # Extract metadata from response
            self.last_response_metadata = {
                "model": response_data.get("model", self.model_name),
                "conversation_id": self.conversation_id,
                "created_at": response_data.get("created_at"),
                "provider": "endpoint",
                "timestamp": datetime.now().isoformat(),
                "response_time_seconds": round(end_time - start_time, 3),
                "done": response_data.get("done", False),
                "total_duration": response_data.get("total_duration"),
                "load_duration": response_data.get("load_duration"),
                "prompt_eval_count": response_data.get("prompt_eval_count"),
                "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                "eval_count": response_data.get("eval_count"),
                "eval_duration": response_data.get("eval_duration"),
                "usage": {
                    "prompt_tokens": response_data.get("prompt_eval_count", 0),
                    "completion_tokens": response_data.get("eval_count", 0),
                    "total_tokens": response_data.get("prompt_eval_count", 0)
                    + response_data.get("eval_count", 0),
                },
                "raw_response": response_data,
            }

            return content

        except httpx.HTTPStatusError as e:
            # Handle HTTP errors (4xx, 5xx)
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            self.last_response_metadata = {
                "model": self.model_name,
                "provider": "endpoint",
                "timestamp": datetime.now().isoformat(),
                "error": error_msg,
                "status_code": e.response.status_code,
                "usage": {},
            }
            return f"Error generating response: {error_msg}"
        except httpx.RequestError as e:
            # Handle network errors (connection failed, timeout, etc.)
            error_msg = f"Network error: {str(e)}"
            self.last_response_metadata = {
                "model": self.model_name,
                "provider": "endpoint",
                "timestamp": datetime.now().isoformat(),
                "error": error_msg,
                "usage": {},
            }
            return f"Error generating response: {error_msg}"
        except Exception as e:
            # Handle any other errors
            error_msg = str(e)
            self.last_response_metadata = {
                "model": self.model_name,
                "provider": "endpoint",
                "timestamp": datetime.now().isoformat(),
                "error": error_msg,
                "usage": {},
            }
            return f"Error generating response: {error_msg}"

    def get_last_response_metadata(self) -> Dict[str, Any]:
        """Get metadata from the last response.

        Returns:
            Dictionary containing metadata from the last API call
        """
        return self.last_response_metadata.copy()

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt.

        Note: System prompts are stored but ignored when making API requests,
        as the endpoint API does not support system messages.

        Args:
            system_prompt: The system prompt to store
        """
        self.system_prompt = system_prompt

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close HTTP client."""
        await self.client.aclose()

    def __del__(self):
        """Cleanup: ensure HTTP client is closed when object is destroyed."""
        if hasattr(self, "client"):
            try:
                # Try to close client (may fail if event loop is closed)
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't close in running loop - client will be cleaned up
                    pass
                else:
                    loop.run_until_complete(self.client.aclose())
            except Exception:
                # Ignore cleanup errors
                pass
