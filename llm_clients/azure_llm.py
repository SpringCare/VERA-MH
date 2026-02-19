import asyncio
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from azure.core.credentials import AzureKeyCredential
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from llm_clients import Role
from utils.conversation_utils import build_langchain_messages
from utils.debug import debug_print

from .config import Config
from .llm_interface import JudgeLLM

# Define type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


class AzureLLM(JudgeLLM):
    """Azure OpenAI implementation using LangChain."""

    DEFAULT_API_VERSION = "2024-05-01-preview"

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

        if not Config.AZURE_API_KEY:
            raise ValueError("AZURE_API_KEY not found in environment variables")
        if not Config.AZURE_ENDPOINT:
            raise ValueError("AZURE_ENDPOINT not found in environment variables")

        # Use provided model name or fall back to config default
        # NOTE: to use the Azure LLM from generate.py
        # you need to prepend "azure-" to the model name (e.g. "azure-grok-4")
        self.model_name = (model_name or Config.get_azure_config()["model"]).replace(
            "azure-", ""
        )

        # Normalize endpoint
        endpoint = Config.AZURE_ENDPOINT.rstrip("/")

        # For Azure AI Foundry, endpoint needs /models
        if ".services.ai.azure.com" in endpoint and not endpoint.endswith("/models"):
            # Check if /models is already in the path
            if "/models" not in endpoint:
                endpoint = f"{endpoint}/models"

        # Store the final normalized endpoint
        self.endpoint = endpoint

        # Use AzureKeyCredential for authentication
        credential = AzureKeyCredential(Config.AZURE_API_KEY)

        # Get default config and allow kwargs to override
        llm_params = {
            "endpoint": self.endpoint,
            "credential": credential,
            "model": self.model_name,
        }

        # Add API version if configured (required for some Azure services)
        # Store as instance variable for reuse
        if Config.AZURE_API_VERSION:
            self.api_version = Config.AZURE_API_VERSION
        else:
            # Default API version if not specified
            # This is often required for Azure AI Foundry services
            self.api_version = self.DEFAULT_API_VERSION
        llm_params["api_version"] = self.api_version

        # Enable logging to see actual requests (helps debug 404s)
        # This will show the actual URL being called
        llm_params["client_kwargs"] = {"logging_enable": True}

        # Override with any provided kwargs
        llm_params.update(kwargs)

        # Print configuration before creating LLM
        print("Creating Azure LLM with parameters:")
        print(f"  Model (deployment name): {llm_params['model']}")
        print(f"  Endpoint: {llm_params['endpoint']}")
        print(f"  API Version: {llm_params.get('api_version', 'default')}")
        print(f"  Temperature: {llm_params.get('temperature', 'default')}")
        print(f"  Max tokens: {llm_params.get('max_tokens', 'default')}")
        print(f"  Original endpoint from config: {Config.AZURE_ENDPOINT}")
        extra_params = {
            k: v
            for k, v in llm_params.items()
            if k not in ["model", "endpoint", "credential", "api_version"]
        }
        if extra_params:
            print(f"  Extra parameters: {extra_params}")

        # Validate endpoint format
        if not self.endpoint.startswith("https://"):
            raise ValueError(
                f"Azure endpoint must start with 'https://'. Got: {self.endpoint}"
            )

        # Validate endpoint matches expected Azure patterns
        expected_patterns = (".openai.azure.com", ".services.ai.azure.com")
        if not any(pattern in self.endpoint for pattern in expected_patterns):
            raise ValueError(
                f"Azure endpoint must match expected patterns. "
                f"Expected '.openai.azure.com' or '.services.ai.azure.com'. "
                f"Got: {self.endpoint}"
            )

        self.llm = AzureAIChatCompletionsModel(**llm_params)

        print(f"Using Azure model: {self.llm.model_name}")

        # Store configuration parameters for logging
        self.temperature = getattr(self.llm, "temperature", None)
        self.max_tokens = getattr(self.llm, "max_tokens", None)
        self.top_p = getattr(self.llm, "top_p", None)

    async def start_conversation(self) -> str:
        """Produce the first response:
        - static first_message if set, or
        - LLM with start_prompt if first_message is not set.
        """
        if self.first_message is not None:
            self._set_response_metadata("azure", static_first_message=True)
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
        # Role reminder is automatically added for personas by build_langchain_messages
        messages.extend(build_langchain_messages(self.role, conversation_history))

        # Debug: Print messages being sent to LLM
        debug_print(f"\n[DEBUG {self.name} - {self.role.value}] Messages sent to LLM:")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            preview = msg.text[:100]
            content_preview = preview + "..." if len(msg.text) > 100 else msg.text
            debug_print(f"  {i+1}. {msg_type}: {content_preview}")

        try:
            # Debug: Print what we're about to send
            debug_print(f"\n[DEBUG {self.name} - {self.role.value}] Calling Azure:")
            debug_print(f"  Model: {self.model_name}")
            debug_print(f"  Endpoint: {self.endpoint}")
            debug_print(f"  API Version: {self.api_version}")
            debug_print(f"  Number of messages: {len(messages)}")

            start_time = time.time()
            response = await self.llm.ainvoke(messages)
            end_time = time.time()

            model = (
                getattr(response.response_metadata, "model", self.model_name)
                if hasattr(response, "response_metadata")
                else self.model_name
            )
            self._set_response_metadata(
                "azure",
                response_id=getattr(response, "id", None),
                model=model,
                response_time_seconds=round(end_time - start_time, 3),
                finish_reason=None,
                response=response,
            )

            if hasattr(response, "response_metadata") and response.response_metadata:
                metadata = response.response_metadata

                # Extract token usage
                if "token_usage" in metadata:
                    usage = metadata["token_usage"]
                    self._last_response_metadata["usage"] = {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
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
            error_msg = str(e)
            self._set_response_metadata("azure", error=error_msg)

            # Provide helpful error message for 404 errors
            if "404" in error_msg or "Resource not found" in error_msg:
                # Try to get more details from the error
                error_details = str(e)
                # Check if it's an Azure SDK error with more details
                if hasattr(e, "response"):
                    response = getattr(e, "response", None)
                    if response and hasattr(response, "url"):
                        error_details += f"\n  Request URL: {response.url}"
                if hasattr(e, "status_code"):
                    status = getattr(e, "status_code", "N/A")
                    error_details += f"\n  Status Code: {status}"

                helpful_msg = (
                    "Azure 404 Error - Resource not found. Common causes:\n"
                    f"  1. Model name '{self.model_name}' doesn't match "
                    "deployment name in Azure portal (check case sensitivity)\n"
                    f"     → Check Azure portal → Deployments for exact name\n"
                    f"  2. Endpoint '{self.endpoint}' is incorrect "
                    "or resource doesn't exist\n"
                    f"     → Original config: {Config.AZURE_ENDPOINT}\n"
                    f"  3. API version '{self.api_version}' not supported\n"
                    "  4. Deployment not active or not accessible "
                    "with current credentials\n"
                    f"\n  Error details: {error_details}"
                )
                debug_print(
                    f"\n[DEBUG {self.name} - {self.role.value}] " f"{helpful_msg}"
                )
                return f"Error generating response: {helpful_msg}"

            return f"Error generating response: {error_msg}"

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
                "azure",
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
            self._set_response_metadata("azure", error=str(e))
            raise RuntimeError(f"Error generating structured response: {str(e)}") from e

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt

    async def cleanup(self) -> None:
        """Clean up Azure LLM resources by closing HTTP sessions.

        This method closes any HTTP client sessions to prevent resource leaks.
        Should be called when the LLM instance is no longer needed.
        """
        await self.close()

    async def close(self) -> None:
        """Close the underlying LLM client and clean up resources.

        This method closes any HTTP client sessions to prevent resource leaks.
        Should be called when the LLM instance is no longer needed.
        """
        # Check __dict__ directly to avoid recursion through __getattr__
        if "llm" not in self.__dict__ or not self.llm:
            return

        # AzureAIChatCompletionsModel has an aclose() method that closes _async_client
        # This is the proper way to clean up the async HTTP client
        if hasattr(self.llm, "aclose"):
            try:
                await self.llm.aclose()
                # Give a small delay to allow cleanup to complete
                await asyncio.sleep(0.1)
                return
            except Exception as e:
                # Log but don't fail if cleanup fails
                debug_print(f"[DEBUG {self.name}] Failed to close via aclose(): {e}")

        # Fallback: Try to close _async_client directly if aclose() doesn't exist
        # AzureAIChatCompletionsModel stores the async client in _async_client
        # Note: _async_client is a PrivateAttr, so we need to check if it exists
        if hasattr(self.llm, "_async_client"):
            try:
                async_client = getattr(self.llm, "_async_client", None)
                if async_client and hasattr(async_client, "close"):
                    close_method = getattr(async_client, "close")
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                        # Give a small delay to allow cleanup to complete
                        await asyncio.sleep(0.1)
                    else:
                        close_method()
            except Exception as e:
                debug_print(f"[DEBUG {self.name}] Failed to close _async_client: {e}")
