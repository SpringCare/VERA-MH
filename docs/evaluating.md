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

## Prompt caching (by provider)

Provider APIs differ in how prompt/context caching works. The built-in clients behave as follows:

| Provider | Behavior in this repo |
|----------|------------------------|
| **Claude** (`ClaudeLLM`) | Anthropic prompt caching is **opt-in** per request. The client passes `cache_control` (default: ephemeral TTL, typically 5 minutes). Set `caching=False` on `ClaudeLLM` to disable. For TTL or other API shapes, pass `anthropic_cache_control=...` (set to `None` to omit `cache_control` while keeping other constructor behavior). |
| **OpenAI** (`OpenAILLM`) | OpenAI applies **automatic** prompt caching for eligible models/prefixes. The client passes `prompt_cache_key` (per conversation) on each call to improve cache routing; there is no separate on/off flag in this wrapper. |
| **Azure** (`AzureLLM`) | Follows the underlying Azure/OpenAI-compatible API. This wrapper does not set Anthropic-style `cache_control` or OpenAI-style `prompt_cache_key`. |
| **Gemini** (`GeminiLLM`) | For **eligible models** (e.g. Gemini 2.5+), the Google GenAI API applies **implicit (automatic) prompt caching** when requests share a common prefix—no extra parameters in this client. **Explicit** context caching (`cached_content` resources) is **not** wired in `GeminiLLM`; that path needs a separate create/update lifecycle and is unrelated to implicit caching. |

## Reasoning / extended thinking (by provider)

See [README.md](../README.md#reasoning--extended-thinking) for the user-facing `-uep`/`-pep`/`-jep` flag per provider (`thinking_effort`, `reasoning_effort`, `thinking_level`/`thinking_budget`). This section covers what each client does internally to make that flag work, and which sampling params it silently drops so a reasoning-enabled request doesn't 400.

| Provider | Behavior in this repo |
|----------|------------------------|
| **Claude** (`ClaudeLLM`) | `thinking_effort` is a CLI-only shorthand — not a native Anthropic field — translated per-model into either `thinking={"type": "adaptive"}` + `effort=<level>` (models with the `adaptive_thinking` quirk, see table below) or `thinking={"type": "enabled", "budget_tokens": N}` (older/non-adaptive models, e.g. Haiku 4.5, using the `_EFFORT_BUDGETS` low/medium/high/max lookup). Adaptive models reject `temperature`/`top_p`/`top_k` outright once thinking is active, so `ClaudeLLM` drops them via `_unsupported_model_params`. Anthropic also requires `temperature=1` whenever `thinking` is set at all; the client forces this, overriding any `temperature` passed alongside `thinking_effort`. |
| **OpenAI** (`OpenAILLM`) | `reasoning_effort` is a native `ChatOpenAI` field — passed straight through, no translation. `gpt-5`+ models (major version ≥ 5, excluding `*-chat`) reject `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs`, and `top_logprobs` once reasoning is active; `OpenAILLM._model_supports_param` filters all of them (langchain-openai itself only strips `temperature`, so the rest would otherwise 400). Gated on parsed major version rather than a fixed `gpt-5` prefix, so `gpt-6`+ is covered without a code change. |
| **Gemini** (`GeminiLLM`) | `thinking_level` (`"low"`/`"high"`, Gemini 3+, takes precedence) and `thinking_budget` (integer token budget, Gemini 2.5) are native `ChatGoogleGenerativeAI` fields — no dict wrapping needed, unlike Claude's `thinking` param. `gemini-3`+ models (major version ≥ 3) drop `temperature`/`top_p`/`top_k` via `GeminiLLM._model_supports_param`: the API doesn't reject these outright, but Google's migration guidance says forcing them (e.g. this judge's `temperature=0` default) can cause looping or degraded reasoning quality. Gated on parsed major version rather than a fixed `gemini-3` marker, so `gemini-4`+ is covered without a code change. |

### Claude per-model quirks

`ClaudeLLM` keys additional per-model special-casing off substring markers matched against `model_name` (`_MODEL_QUIRKS` in [`claude_llm.py`](../llm_clients/claude_llm.py)), rather than adding another `is_x` branch per model:

| Quirk | Applies to | Effect |
|-------|------------|--------|
| `adaptive_thinking` | `opus-4-7`, `opus-4-8`, `fable-5`, `sonnet-5` | Uses `thinking={"type": "adaptive"}` + `effort=<level>` instead of manual `budget_tokens` (which 400s on these models); rejects `temperature`/`top_p`/`top_k` at any non-default value. |
| `defaults_thinking_on` | `sonnet-5` | Runs adaptive thinking by default when `thinking` is omitted, so `thinking_effort` must be explicitly unset to disable it. Not needed for `fable-5`, where adaptive thinking is always on and can't be disabled at all. |
| `sparse_max_tokens_profile` | `sonnet-5` | The installed `langchain-anthropic` has no model-profile entry for this model, so it silently falls back to `max_tokens=4096` instead of the usual 64k–128k auto-set — too tight for structured output. The client sets an explicit default instead. |

To onboard a new Claude model that needs similar special-casing, add its marker to `_MODEL_QUIRKS` with the applicable quirks rather than adding another `is_x` check.

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