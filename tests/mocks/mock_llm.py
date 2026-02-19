from typing import Any, Dict, List, Optional, Type, TypeVar

from llm_clients.llm_interface import JudgeLLM, Role

T = TypeVar("T")


class MockLLM(JudgeLLM):
    """Mock LLM for testing without API calls.

    Implements JudgeLLM to support structured output generation for judge operations.
    """

    def __init__(
        self,
        name: str = "mock-llm",
        role: Role = Role.PROVIDER,
        responses: Optional[List[str]] = None,
        model_name: str = "mock-model",
        system_prompt: Optional[str] = None,
        first_message: Optional[str] = None,
        start_prompt: Optional[str] = None,
        simulate_error: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        super().__init__(
            name,
            role,
            system_prompt,
            first_message=first_message,
            start_prompt=start_prompt,
        )
        self.responses = responses or ["Mock response"]
        self.response_index = 0
        self.calls: List[str] = []
        self.simulate_error = simulate_error
        self.last_response_metadata: Dict[str, Any] = {}
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def start_conversation(self) -> str:
        """Produce the first response (static first_message or next in sequence)."""
        if self.first_message is not None:
            self._set_response_metadata("mock", static_first_message=True)
            return self.first_message
        return await self.generate_response(self.get_initial_prompt_turns())

    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Return predetermined responses in sequence.

        Args:
            conversation_history: Optional list of previous conversation turns

        Returns:
            Response text string
        """
        if not conversation_history or len(conversation_history) == 0:
            return await self.start_conversation()

        # Extract the last message from conversation history for tracking
        message = ""
        if conversation_history and len(conversation_history) > 0:
            message = conversation_history[-1].get("response", "")
        self.calls.append(message or "")

        if self.simulate_error:
            self._set_response_metadata("mock", error="Simulated API error")
            raise Exception("Simulated API error")

        if self.response_index >= len(self.responses):
            response = f"Mock response {self.response_index + 1}"
        else:
            response = self.responses[self.response_index]
            self.response_index += 1

        self._set_response_metadata(
            "mock",
            model=self.model_name,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

        return response

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        """Generate a structured response using Pydantic model.

        For testing, this creates a mock instance of the response model
        with default/example values.

        Args:
            message: The prompt message
            response_model: Pydantic model class to structure the response

        Returns:
            Instance of response_model with mock data
        """
        # Get the next response or use default
        if self.response_index < len(self.responses):
            response_text = self.responses[self.response_index]
            self.response_index += 1
        else:
            response_text = f"Mock response {self.response_index + 1}"
            self.response_index += 1

        self.calls.append(message or "")

        # Try to parse the response as JSON and create model instance
        try:
            import json

            response_data = json.loads(response_text)
            return response_model(**response_data)
        except (json.JSONDecodeError, ValueError, TypeError):
            # Create instance with mock default values
            # Check if the model has example data in json_schema_extra
            if hasattr(response_model, "model_config") and isinstance(
                response_model.model_config.get("json_schema_extra"), dict
            ):
                example_data = response_model.model_config.get(
                    "json_schema_extra", {}
                ).get("example", {})
                if example_data:
                    return response_model(**example_data)
            elif hasattr(response_model, "Config") and hasattr(
                response_model.Config, "json_schema_extra"
            ):
                # Pydantic v1 style config
                json_schema_extra = response_model.Config.json_schema_extra
                if (
                    isinstance(json_schema_extra, dict)
                    and "example" in json_schema_extra
                ):
                    return response_model(**json_schema_extra["example"])

            # If no example, construct default values based on field types
            default_values = {}
            for field_name, field_info in response_model.model_fields.items():
                annotation = field_info.annotation
                # Check if field is a string type
                if annotation is str or (
                    isinstance(annotation, type) and issubclass(annotation, str)
                ):
                    default_values[field_name] = f"Mock {field_name}"
                elif annotation is int:
                    default_values[field_name] = 0
                elif annotation is bool:
                    default_values[field_name] = False
                elif annotation is float:
                    default_values[field_name] = 0.0
                # Fallback for string annotation (handle Union, Optional, etc.)
                elif "str" in str(annotation):
                    default_values[field_name] = f"Mock {field_name}"
            return response_model(**default_values)
