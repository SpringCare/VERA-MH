import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from utils.conversation_utils import build_langchain_messages
from utils.debug import debug_print

from .config import Config
from .llm_interface import JudgeLLM, Role

T = TypeVar("T", bound=BaseModel)


class GeminiLLM(JudgeLLM):
    """Gemini implementation using LangChain."""

    def __init__(
        self,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        first_message = kwargs.pop("first_message", None)
        start_prompt = kwargs.pop("start_prompt", None)
        super().__init__(
            name,
            role,
            system_prompt,
            first_message=first_message,
            start_prompt=start_prompt,
        )

        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_gemini_config()["model"]

        # Get default config and allow kwargs to override
        llm_params = {
            "google_api_key": Config.GOOGLE_API_KEY,
            "model": self.model_name,
        }

        # Override with any provided kwargs
        llm_params.update(kwargs)

        # Print configuration before creating LLM
        print("Creating Gemini LLM with parameters:")
        print(f"  Model: {llm_params['model']}")
        print(f"  Temperature: {llm_params.get('temperature', 'default')}")
        print(f"  Max tokens: {llm_params.get('max_tokens', 'default')}")
        extra_params = {
            k: v for k, v in llm_params.items() if k not in ["model", "google_api_key"]
        }
        if extra_params:
            print(f"  Extra parameters: {extra_params}")

        self.llm = ChatGoogleGenerativeAI(**llm_params)

        print(f"Using Gemini model: {self.llm.model}")

        # Store configuration parameters for logging
        self.temperature = getattr(self.llm, "temperature", None)
        self.max_tokens = getattr(self.llm, "max_tokens", None)

    async def start_conversation(self) -> str:
        """Produce the first response:
        - static first_message if set, or
        - LLM with start_prompt if first_message is not set.
        """
        if self.first_message is not None:
            self._set_response_metadata("gemini", static_first_message=True)
            return self.first_message
        return await self.generate_response(self.get_initial_prompt_turns())

    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response based on conversation history.

        Args:
            conversation_history: Optional list of previous conversation turns
        """
        if not conversation_history or len(conversation_history) == 0:
            return await self.start_conversation()

        messages = []

        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        # Build messages from history
        messages.extend(build_langchain_messages(self.role, conversation_history))

        # Debug: Print messages being sent to LLM
        debug_print(f"\n[DEBUG {self.name} - {self.role.value}] Messages sent to LLM:")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            preview = msg.text[:100]
            content_preview = preview + "..." if len(msg.text) > 100 else msg.text
            debug_print(f"  {i + 1}. {msg_type}: {content_preview}")

        try:
            start_time = time.time()
            response = await self.llm.ainvoke(messages)
            end_time = time.time()

            model = (
                getattr(response.response_metadata, "model_name", self.model_name)
                if hasattr(response, "response_metadata")
                else self.model_name
            )
            self._set_response_metadata(
                "gemini",
                response_id=getattr(response, "id", None),
                model=model,
                response_time_seconds=round(end_time - start_time, 3),
                finish_reason=None,
                response=response,
            )

            if hasattr(response, "response_metadata") and response.response_metadata:
                metadata = response.response_metadata

                # Extract token usage - Gemini may have different structure
                if "usage_metadata" in metadata:
                    usage = metadata["usage_metadata"]
                    self._last_response_metadata["usage"] = {
                        "prompt_token_count": usage.get("prompt_token_count", 0),
                        "candidates_token_count": usage.get(
                            "candidates_token_count", 0
                        ),
                        "total_token_count": usage.get("total_token_count", 0),
                    }
                elif "token_usage" in metadata:
                    # Fallback structure
                    usage = metadata["token_usage"]
                    self._last_response_metadata["usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                # Extract finish reason
                self._last_response_metadata["finish_reason"] = metadata.get(
                    "finish_reason"
                )

                # Store raw metadata
                self._last_response_metadata["raw_metadata"] = dict(metadata)

            return response.text
        except Exception as e:
            self._set_response_metadata("gemini", error=str(e))
            return f"Error generating response: {str(e)}"

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        """Generate a structured response using Pydantic model.

        Args:
            message: The prompt message
            response_model: Pydantic model class to structure the response

        Returns:
            Instance of the response_model with structured data
        """
        messages = []

        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        messages.append(HumanMessage(content=message))

        try:
            # Create a structured LLM using with_structured_output
            structured_llm = self.llm.with_structured_output(response_model)

            start_time = time.time()
            response = await structured_llm.ainvoke(messages)
            end_time = time.time()

            self._set_response_metadata(
                "gemini",
                response_time_seconds=round(end_time - start_time, 3),
                structured_output=True,
            )

            # Ensure response is the correct type
            if not isinstance(response, response_model):
                raise ValueError(
                    f"Response is not an instance of {response_model.__name__}"
                )

            return response  # type: ignore[return-value]
        except Exception as e:
            self._set_response_metadata("gemini", error=str(e))
            raise RuntimeError(f"Error generating structured response: {str(e)}") from e

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
