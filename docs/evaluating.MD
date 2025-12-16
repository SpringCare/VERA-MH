# Evaluate your own model or service

VERA-MH is ready to be used to evaluate any chat-based interface.
[This](../llm_clients/llm_interface.py) Abstract Base Class (ABC) represents the interface to be implemented.
Four concrete implementations of that class are provided for the APIs of ChatGPT, Claude, Gemini, and Llama (via Ollama).

To test your service, you need to instantiate a concrete class and implement two key methods:
- `generate_response()`: Returns a string (the chatbot response) given a string (the user input)
- `generate_structured_response()`: Returns a Pydantic model instance for structured outputs (used by the judge system)

## Adding Support for a New LLM Provider

Follow these steps to add a new LLM provider:

### 1. Create a new class that inherits from `LLMInterface`

```python
from llm_clients.llm_interface import LLMInterface
from typing import Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class YourLLM(LLMInterface):
    """Your LLM implementation."""

    def __init__(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, system_prompt)
        # Initialize your LLM client here
```

### 2. Implement the required methods

#### `generate_response()` - For conversation generation
```python
async def generate_response(self, message: Optional[str] = None) -> str:
    """Generate a text response.

    Args:
        message: The user's input message

    Returns:
        The LLM's response as a string
    """
    # Your implementation here
    # Should be async and return a string
```

#### `generate_structured_response()` - For judge evaluation
```python
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
    # Option 1: Use LangChain's with_structured_output (recommended)
    structured_llm = self.llm.with_structured_output(response_model)
    response = await structured_llm.ainvoke(message)
    return response

    # Option 2: If your provider doesn't support structured output,
    # parse the text response and construct the Pydantic model
```

#### `set_system_prompt()` - For updating prompts
```python
def set_system_prompt(self, system_prompt: str) -> None:
    """Set or update the system prompt."""
    self.system_prompt = system_prompt
```

### 3. Add the new LLM client to the factory

Update [llm_factory.py](../llm_clients/llm_factory.py):

```python
from .your_llm import YourLLM

class LLMFactory:
    @staticmethod
    def create_llm(model_name: str, name: str, system_prompt: Optional[str] = None, **kwargs):
        if "your-model-prefix" in model_name.lower():
            return YourLLM(name, system_prompt, model_name, **kwargs)
        # ... existing conditions
```

### 4. Update configuration (if needed)

Add configuration to [config.py](../llm_clients/config.py) if your LLM requires API keys or special settings.

### 5. Use the new LLM in your simulations

```bash
python3 generate.py -u your-model-name -p your-model-name -t 5 -r 1
python3 judge.py -f conversations/{YOUR_FOLDER} -j your-model-name
```

## Important Notes

- **Async Support**: The current implementation uses `async` to avoid blocking when multiple conversations are being generated
- **Structured Output**: For the judge system to work properly, your LLM should support structured output via `generate_structured_response()`
- **LangChain Integration**: The provided implementations use LangChain for robust LLM interactions
- **Error Handling**: Make sure to handle errors gracefully and return appropriate error messages

## Structured Output Support

### Native Support (Recommended)
Claude, OpenAI, and Gemini support structured output natively through their APIs via LangChain's `with_structured_output()`:

```python
structured_llm = self.llm.with_structured_output(response_model)
response = await structured_llm.ainvoke(messages)
```

### Limited Support
If your LLM doesn't support native structured output (like Llama/Ollama), you can:
1. Raise a `NotImplementedError` and recommend using a different model for judging
2. Implement prompt-based parsing (less reliable)

See [llama_llm.py](../llm_clients/llama_llm.py) for an example of limited structured output support.
 