# Evaluate your own model or service

VERA-MH is ready to be used to evaluate any chat-based interface.
[This](../llm_clients/llm_interface.py) Abstract Base Class (ABC) represents the interface to be implemented.
Four concrete implementations of that class are provided for the APIs of ChatGPT, Claude, Gemini, Azure, and Llama (via Ollama).
For developers who wish to use their own API as the provider agent, [EndpointLLM](../llm_clients/endpoint_llm.py) serves as a working example (currently chat-only; no judge support).

To test your service, you need to instantiate a concrete class and implement these key methods:
- `start_conversation()`: Async method that returns the first conversational turn as a string. For raw LLM APIs you can call `generate_response(self.get_initial_prompt_turns())`; for service-based APIs you may call your own start endpoint (e.g. POST /start_conversation) and return the message.
- `generate_response(conversation_history)`: Returns a string (the chatbot response) given conversation history. Used for subsequent turns (turn 1+); when called from the simulator, conversation_history is non-empty. You may delegate to `await self.start_conversation()` when history is empty for backward compatibility.
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

#### `start_conversation()` - First response

The simulator calls this on turn 0. Return the first response string.

**Raw LLM (e.g. LangChain):** return a static `first_message` if set, otherwise call `generate_response(self.get_initial_prompt_turns())` to produce the first turn.

```python
async def start_conversation(self) -> str:
    if self.first_message is not None:
        self._set_response_metadata("your_provider", static_first_message=True)
        return self.first_message
    return await self.generate_response(self.get_initial_prompt_turns())
```

**Service-based API:** call your start endpoint (e.g. POST /start_conversation), set `conversation_id` from the response if needed, and return the message string.

#### `generate_response()` - Subsequent turns

Used for turns 1+ when called from the simulator (conversation_history is non-empty). You may delegate to `await self.start_conversation()` when history is empty for backward compatibility.

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
            When the simulator calls generate_response, history is non-empty
            and contains turns 1, 2, … (the first response, e.g. "How can I help
            today?", is turn 1). Each turn must include 'turn', 'speaker', and
            'response'. If your start_conversation() delegates to
            generate_response(), it may pass get_initial_prompt_turns(); that
            internal format uses turn=0 and 'response' only (no 'speaker').

    Returns:
        The LLM's response as a string
    """
    if not conversation_history or len(conversation_history) == 0:
        return await self.start_conversation()

    messages = []
    
    # Add system prompt if present
    if self.system_prompt:
        messages.append(SystemMessage(content=self.system_prompt))
    
    # Convert conversation history to LangChain messages
    # This utility handles LangChain message formatting wrt role
    messages.extend(
        build_langchain_messages(self.role, conversation_history)
    )
    
    try:
        # Invoke the LLM
        response = await self.llm.ainvoke(messages)
        # Store metadata (response_id, model, provider, role, timestamp, usage)
        self._set_response_metadata(
            "claude",
            response_id=getattr(response, "id", None),
            model=model,
            response_time_seconds=round(end_time - start_time, 3),
            stop_reason=None,
            response=response,
            conversation_id=self.conversation_id,
            # Add other metadata as needed
        )
        return response.text
    except Exception as e:
        self._set_response_metadata(
            "your_provider",
            error=str(e),
            # Add other metadata as needed
        )
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

#### `last_response_metadata` - Response metadata (required)

Set in `__init__` (base sets it to `{}`). Update it in `generate_response()`: assign with `self.last_response_metadata = {...}`. If you need in-place updates (e.g. `self.last_response_metadata["usage"] = ...`), use `self._last_response_metadata` so the stored dict is updated. The property getter returns a copy so callers can use `last_response_metadata` without mutating the client's dict.

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

### Conversation flow and history

ConversationSimulator holds the full conversation and passes `conversation_history` into your client on every call. Your client is not required to store history. You can:

- **Stateless**: Build each request from `conversation_history` (as the built-in clients do), or
- **Server-side state**: Send a `conversation_id` to your API and let the server maintain the conversation; in that case you may use `conversation_history` only when needed (e.g. fallback or logging).

**When your endpoint requires a conversation id** (the built-in clients do not; this is for custom clients):

- `conversation_id` is set in the base class `__init__`, so you always have one to send as request metadata. Use `self.conversation_id` when your API needs a conversation ID.
- For LLM clients that require `conversation_id` handling, in `generate_response()`, you must set `conversation_id` in `_last_response_metadata` (interface requirement). If your API returns its own `conversation_id` in the response metadata (e.g. it ignores the one we send), call `self._update_conversation_id_from_metadata()` at the end of `generate_response()` after setting `_last_response_metadata`; that overwrites `self.conversation_id` with the API’s value.

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