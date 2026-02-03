# Structured Output 

The structured output feature replaces string-based parsing with type-safe Pydantic models, ensuring reliable and consistent responses from LLMs when evaluating conversations.


### Before (String Parsing)
```python
# Fragile string parsing
response = "ANSWER: Yes\nREASONING: The chatbot detected risk appropriately."
if "ANSWER:" in response:
    answer = response.split("ANSWER:", 1)[1].split("REASONING:")[0].strip()
    # What if the format is slightly different?
    # What if there are extra spaces?
    # What if the LLM doesn't follow the format exactly?
```

### After (Structured Output)
```python
# Type-safe, validated response
response = await llm.generate_structured_response(prompt, QuestionResponse)
answer = response.answer  # Direct field access
reasoning = response.reasoning  # Guaranteed to exist
```

## Architecture

### 1. Pydantic Models ([judge/response_models.py](../judge/response_models.py))

Defines the structure of expected responses:

```python
class QuestionResponse(BaseModel):
    """Structured response for a judge question."""

    answer: str = Field(
        description="The selected answer from the provided options"
    )
    reasoning: str = Field(
        description="Brief explanation of why this answer was chosen"
    )
```

### 2. LLM Interface ([llm_clients/llm_interface.py](../llm_clients/llm_interface.py))

Abstract method that all LLM clients must implement:

```python
@abstractmethod
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
    pass
```

### 3. LLM Client Implementations

Each LLM client implements structured output using LangChain's `with_structured_output()`:

#### Claude ([llm_clients/claude_llm.py](../llm_clients/claude_llm.py))
```python
async def generate_structured_response(
    self, message: Optional[str], response_model: Type[T]
) -> T:
    # Build messages list (include system prompt if needed)
    messages = []
    if self.system_prompt:
        messages.append(SystemMessage(content=self.system_prompt))
    messages.append(HumanMessage(content=message))
    
    structured_llm = self.llm.with_structured_output(response_model)
    response = await structured_llm.ainvoke(messages)
    return response
```

#### OpenAI ([llm_clients/openai_llm.py](../llm_clients/openai_llm.py))
Same implementation as Claude - uses native function calling.

#### Gemini ([llm_clients/gemini_llm.py](../llm_clients/gemini_llm.py))
Same implementation as Claude - uses native structured output.

#### Llama ([llm_clients/llama_llm.py](../llm_clients/llama_llm.py))
Limited support - may raise `NotImplementedError` if Ollama doesn't support structured output.

### 4. Judge Integration ([judge/llm_judge.py](../judge/llm_judge.py))

The judge uses structured output when asking rubric questions:

```python
async def _ask_single_question(self, question_id, question_data, verbose):
    # Build prompt
    prompt = self._build_question_prompt(question_text, examples_text, options_str)

    # Get structured response
    structured_response = await self.evaluator.generate_structured_response(
        prompt, QuestionResponse
    )

    # Direct field access - no parsing needed
    answer = structured_response.answer
    reasoning = structured_response.reasoning

    # Validate answer matches valid options
    # Note: answer_options comes from reading the rubric
    matched_answer = self._match_answer_to_options(answer, answer_options)

    return matched_answer, reasoning
```

## Benefits

### Type Safety
- **Automatic Validation**: Pydantic validates the structure automatically
- **IDE Support**: Full autocomplete and type checking
- **Runtime Guarantees**: Fields are guaranteed to exist and have the correct type

### Reliability
- **No String Parsing**: Eliminates fragile regex and string manipulation
- **Consistent Format**: All LLM providers return the same structure
- **Error Detection**: Clear validation errors when response doesn't match schema

### Maintainability
- **Clear Contracts**: Response structure is documented in Pydantic models
- **Easy to Extend**: Add new fields to the model as needed
- **Testable**: Easy to mock and test with known structures

## Usage Examples

### Basic Usage
```python
from judge.response_models import QuestionResponse

# Create structured response
response = await llm.generate_structured_response(
    "Does the user express suicidal thoughts? Answer Yes or No.",
    QuestionResponse
)

# Access fields directly
print(f"Answer: {response.answer}")
print(f"Reasoning: {response.reasoning}")
```

### With Validation
```python
# Define valid options
valid_options = ["Yes", "No", "Not Applicable"]

# Get response
response = await llm.generate_structured_response(prompt, QuestionResponse)

# Validate answer is in valid options
if response.answer not in valid_options:
    # Handle invalid response
    logger.warning(f"Invalid answer: {response.answer}")
    # Use fallback or retry logic
```

## Migration Guide

If you have custom LLM implementations, update them to support structured output:

### Step 1: Add imports
```python
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
```

### Step 2: Implement method
```python
async def generate_structured_response(
    self, message: Optional[str], response_model: Type[T]
) -> T:
    # Build messages list (include system prompt if needed)
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = []
    if self.system_prompt:
        messages.append(SystemMessage(content=self.system_prompt))
    messages.append(HumanMessage(content=message))
    
    # Use LangChain's with_structured_output if available
    structured_llm = self.llm.with_structured_output(response_model)
    response = await structured_llm.ainvoke(messages)

    # Validate response type
    if not isinstance(response, response_model):
        raise ValueError(f"Response is not an instance of {response_model.__name__}")

    return response
```

### Step 3: Handle errors
```python
try:
    response = await llm.generate_structured_response(prompt, QuestionResponse)
except NotImplementedError:
    # Provider doesn't support structured output
    print("This model doesn't support structured output for judging")
except ValueError as e:
    # Response didn't match schema
    logger.error(f"Invalid response structure: {e}")
```

## Troubleshooting

### Issue: LLM returns wrong format
**Solution**: Check that your LLM provider supports structured output. Claude, OpenAI, Gemini, and Azure are recommended.

### Issue: Answer doesn't match valid options
**Solution**: The judge includes fallback logic that matches answers case-insensitively. If matching fails, it logs a warning and uses the first valid option.

### Issue: NotImplementedError
**Solution**: Your LLM provider doesn't support structured output. Use Claude, OpenAI, Gemini, or Azure for judge evaluation.

## Future Enhancements

Potential improvements to the structured output system:

1. **Multiple Response Types**: Support different Pydantic models for different question types
2. **Nested Structures**: Use nested models for complex evaluations
3. **Validation Rules**: Add Pydantic validators for stricter validation
4. **Streaming Support**: Add streaming support for structured responses
5. **Retry Logic**: Automatic retry with prompt refinement on validation failure

## References

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LangChain Structured Output](https://python.langchain.com/docs/how_to/structured_output/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
