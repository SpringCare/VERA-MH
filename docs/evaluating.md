# Evaluate your own model or service

VERA-MH is ready to be used to evaluate any chat-based interface.
[This](../llm_clients/llm_interface.py) Abstract Base Class (ABC) represents the interface to be implemented.
Four concrete implementations of that class are provided for the APIs of ChatGPT, Claude, Gemini, Azure, and Llama (via Ollama).

To test your service, you need to instantiate a concrete class and implement two key methods:
- `generate_response()`: Returns a string (the chatbot response) given conversation history (list of conversation turns)
- `generate_structured_response()`: Returns a Pydantic model instance for structured outputs (used by the judge system)

## Adding Support for a New LLM Provider

Follow these steps to add a new LLM provider:

### 1. Create a new class that inherits from `LLMInterface` (conversation generation only) or `JudgeLLM` (conversation generation && LLM-as-a-Judge support)

**For conversation generation only:**
```python
from datetime import datetime
from llm_clients.llm_interface import LLMInterface
from typing import Any, Dict, List, Optional

class YourLLM(LLMInterface):
    """Your LLM implementation for conversation generation."""

    def __init__(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, system_prompt)
        # Initialize your LangChain LLM client here
        # Example: self.llm = ChatYourProvider(model=model_name, **kwargs)
        # Store metadata
        self.last_response_metadata: Dict[str, Any] = {}
```

**For judge evaluation (structured output support):**
```python
from datetime import datetime
from llm_clients.llm_interface import JudgeLLM
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class YourLLM(JudgeLLM):
    """Your LLM implementation with LLM-as-a-Judge support."""

    def __init__(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, system_prompt)
        # Initialize your LangChain LLM client here
        # Example: self.llm = ChatYourProvider(model=model_name, **kwargs)
        # Store metadata
        self.last_response_metadata: Dict[str, Any] = {}
```

### 2. Implement the required methods

#### `generate_response()` - For conversation generation
```python
from langchain_core.messages import SystemMessage
from utils.conversation_utils import build_langchain_messages

async def generate_response(
    self,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a response based on conversation history.

    Args:
        conversation_history: List of previous conversation turns.
            Each turn is a dict with keys: 'turn', 'speaker', 'response'.
            On the first turn (turn 0), conversation_history will contain
            a single entry with turn=0, speaker="system", and the initial
            message in the 'response' field.

    Returns:
        The LLM's response as a string
    """
    messages = []
    
    # Add system prompt if present
    if self.system_prompt:
        messages.append(SystemMessage(content=self.system_prompt))
    
    # Convert conversation history to LangChain messages
    # This utility handles role conversion and formatting
    messages.extend(
        build_langchain_messages(conversation_history, self.system_prompt)
    )
    
    try:
        # Invoke the LLM
        response = await self.llm.ainvoke(messages)
        
        # Store metadata (optional but recommended)
        self.last_response_metadata = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            # Add other metadata as needed
        }
        
        return response.text
    except Exception as e:
        # Store error metadata
        self.last_response_metadata = {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        return f"Error generating response: {str(e)}"
```

#### `generate_structured_response()` - For judge support (JudgeLLM only)
```python
from langchain_core.messages import SystemMessage, HumanMessage

async def generate_structured_response(
    self, message: Optional[str], response_model: Type[T]
) -> T:
    """Generate a structured response using Pydantic model.

    Args:
        message: The prompt message
        response_model: Pydantic model class to structure the response

    Returns:
        Instance of the response_model with structured data
        
    Raises:
        RuntimeError: If structured output generation fails
    """
    messages = []
    
    # Add system prompt if present
    if self.system_prompt:
        messages.append(SystemMessage(content=self.system_prompt))
    
    # Add the user message
    messages.append(HumanMessage(content=message))
    
    try:
        # Create structured LLM using LangChain's with_structured_output
        structured_llm = self.llm.with_structured_output(response_model)
        
        # Invoke and get structured response
        response = await structured_llm.ainvoke(messages)
        
        # Validate response type
        if not isinstance(response, response_model):
            raise ValueError(
                f"Response is not an instance of {response_model.__name__}"
            )
        
        # Store metadata (optional)
        self.last_response_metadata = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "structured_output": True,
        }
        
        return response
    except Exception as e:
        # Store error metadata
        self.last_response_metadata = {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        raise RuntimeError(
            f"Error generating structured response: {str(e)}"
        ) from e
```

#### `set_system_prompt()` - For updating prompts
```python
def set_system_prompt(self, system_prompt: str) -> None:
    """Set or update the system prompt."""
    self.system_prompt = system_prompt
```

#### `get_last_response_metadata()` - Get response metadata (optional but recommended)
```python
def get_last_response_metadata(self) -> Dict[str, Any]:
    """Get metadata from the last response."""
    return self.last_response_metadata.copy()
```

### 3. Add the new LLM client to the factory

Update [llm_factory.py](../llm_clients/llm_factory.py):

```python
from .your_llm import YourLLM

class LLMFactory:
    @staticmethod
    def create_llm(model_name: str, name: str, system_prompt: Optional[str] = None, **kwargs):
        model_lower = model_name.lower()
        if "your-model-prefix" in model_lower:
            return YourLLM(name=name, system_prompt=system_prompt, model_name=model_name, **kwargs)
        # ... existing conditions for other models
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
Claude, OpenAI, Azure, and Gemini support structured output natively through their APIs via LangChain's `with_structured_output()`:

```python
# Build messages list (include system prompt if needed)
messages = []
if self.system_prompt:
    from langchain_core.messages import SystemMessage
    messages.append(SystemMessage(content=self.system_prompt))
from langchain_core.messages import HumanMessage
messages.append(HumanMessage(content=message))

structured_llm = self.llm.with_structured_output(response_model)
response = await structured_llm.ainvoke(messages)
```

### Limited Support
If your LLM doesn't support native structured output (like Llama/Ollama), you can:
1. Raise a `NotImplementedError` and recommend using a different model for judging
2. Implement prompt-based parsing (less reliable)

See [llama_llm.py](../llm_clients/llama_llm.py) for an example of limited structured output support.