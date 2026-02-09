# LLM Client Testing Documentation

This directory contains comprehensive unit tests for all LLM client implementations in the VERA-MH project.

## Architecture

The test suite uses a base class hierarchy that ensures all LLM implementations are tested consistently and completely:

```
TestLLMBase (abstract)
    ├── Defines common tests for all LLMInterface implementations
    └── TestJudgeLLMBase (abstract, extends TestLLMBase)
            └── Adds structured output tests for JudgeLLM implementations
```

### Base Test Classes

Located in [`test_base_llm.py`](test_base_llm.py):

- **`TestLLMBase`**: Abstract base class for testing `LLMInterface` implementations
  - Provides standard tests: initialization, response generation, system prompts, metadata, error handling
  - Requires subclasses to implement factory methods
  
- **`TestJudgeLLMBase`**: Abstract base class for testing `JudgeLLM` implementations
  - Extends `TestLLMBase` with all standard tests
  - Adds structured output generation tests
  - Tests Pydantic model validation, complex nested models, error handling

### Coverage Validation

Located in [`test_coverage.py`](test_coverage.py):

Automated tests that run in CI to ensure:
1. ✅ All LLM implementations have corresponding test files
2. ✅ All `JudgeLLM` implementations test structured output generation
3. ✅ No duplicate implementation names
4. ✅ All expected implementations exist

**These tests prevent incomplete test coverage for future LLM implementations.**

All JudgeLLM implementations include:
- Standard LLMInterface tests
- Structured output generation tests (simple and complex models)
- Error handling for structured output
- Provider-specific tests (e.g., Azure endpoint handling)

## Adding Tests for New LLM Implementations

When adding a new LLM client, follow this checklist:

### 1. Determine the Base Class

- **Implementing only `LLMInterface`?** → Extend `TestLLMBase`
- **Implementing `JudgeLLM`?** → Extend `TestJudgeLLMBase`

### 2. Create Test File

File naming convention: `test_{provider}_llm.py`

Example for a new provider "MyProvider":

```python
"""Unit tests for MyProviderLLM class."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from llm_clients import Role
from llm_clients.my_provider_llm import MyProviderLLM

from .test_base_llm import TestJudgeLLMBase
from .test_helpers import (
    assert_metadata_structure,
    assert_response_timing,
    # ... other helpers
)


@pytest.mark.unit
class TestMyProviderLLM(TestJudgeLLMBase):
    """Unit tests for MyProviderLLM class."""

    def create_llm(self, role: Role, **kwargs):
        """Create MyProviderLLM instance for testing."""
        return MyProviderLLM(name="TestMyProvider", role=role, **kwargs)

    def get_provider_name(self) -> str:
        """Get provider name for metadata validation."""
        return "myprovider"

    @contextmanager
    def get_mock_patches(self):
        """Set up mocks for MyProvider."""
        with (
            patch("llm_clients.my_provider_llm.Config.MYPROVIDER_API_KEY", "test-key"),
            patch("llm_clients.my_provider_llm.ChatMyProvider") as mock_client,
        ):
            yield mock_client

    # Add provider-specific tests here
    def test_provider_specific_feature(self):
        """Test MyProvider-specific functionality."""
        with self.get_mock_patches():
            llm = self.create_llm(role=Role.PROVIDER)
            # Test provider-specific behavior
            pass
```

### 3. Add Your Provider to conftest.py

In `conftest.py`, add an `elif provider == "yourprovider":` branch in `mock_response_factory`’s `_create_mock_response`. See [Adding a New Provider to the Mocks](#adding-a-new-provider-to-the-mocks) below.

### 4. Implement Required Factory Methods

All test classes must implement these three abstract methods:

#### `create_llm(role, **kwargs)`
Creates an instance of your LLM for testing.

```python
def create_llm(self, role: Role, **kwargs):
    return MyProviderLLM(name="TestMyProvider", role=role, **kwargs)
```

#### `get_provider_name()`
Returns the provider name string for metadata validation.

```python
def get_provider_name(self) -> str:
    return "myprovider"  # Must match metadata["provider"]
```

#### `get_mock_patches()`
Returns a context manager that patches API keys and external dependencies.

```python
@contextmanager
def get_mock_patches(self):
    with (
        patch("llm_clients.my_provider_llm.Config.API_KEY", "test-key"),
        patch("llm_clients.my_provider_llm.ChatProvider") as mock,
    ):
        yield mock
```

### 5. Inherited Tests

By extending the base classes, you automatically get these tests:

**From `TestLLMBase`:**
- ✅ Basic initialization
- ✅ System prompt management
- ✅ Response generation with conversation history
- ✅ Metadata structure and copying
- ✅ Error handling and error metadata
- ✅ Timing tracking

**From `TestJudgeLLMBase` (if applicable):**
- ✅ Structured output with simple Pydantic models
- ✅ Structured output with complex nested models
- ✅ Structured output error handling
- ✅ Structured response metadata validation

### 6. Add Provider-Specific Tests

Beyond the inherited tests, add tests for provider-specific behavior:

```python
class TestMyProviderLLM(TestJudgeLLMBase):
    # ... factory methods ...

    def test_special_endpoint_handling(self):
        """Test provider-specific endpoint logic."""
        with self.get_mock_patches():
            llm = self.create_llm(role=Role.PROVIDER)
            # Test unique behavior
            pass

    @pytest.mark.asyncio
    async def test_custom_metadata_extraction(self, mock_response_factory):
        """Test provider-specific metadata fields."""
        with self.get_mock_patches() as mock_client:
            mock_response = mock_response_factory(
                text="Response",
                provider=self.get_provider_name(),
                metadata={"custom_field": "value"}
            )
            # Test custom behavior
            pass
```

### 7. Run Coverage Validation

After creating your tests, run the coverage validation:

```bash
pytest tests/unit/llm_clients/test_coverage.py -v
```

This will verify:
- ✅ Your test file exists and is named correctly
- ✅ Structured output tests are present (for JudgeLLM)
- ✅ No naming conflicts with existing implementations

## Helper Functions

Located in [`test_helpers.py`](test_helpers.py):

### Metadata Assertions
- `assert_metadata_structure()` - Validates LLM metadata fields and structure
- `assert_iso_timestamp()` - Validates ISO timestamp format
- `assert_metadata_copy_behavior()` - Verifies metadata copy behavior
- `assert_response_timing()` - Validates timing fields
- `assert_error_metadata()` - Validates error metadata structure

### Response Assertions
- `assert_error_response()` - Validates error message format

### Mock Verification
- `verify_no_system_message_in_call()` - Checks system message absence
- `verify_message_types_for_persona()` - Validates persona role message flipping

## Shared Fixtures

Located in [`conftest.py`](conftest.py):

### Adding a New Provider to the Mocks

**Yes.** If you add a new LLM client, you must add support for your provider in `conftest.py`.

- **`mock_response_factory`** – Base tests call it with `provider=self.get_provider_name()`. The factory has an explicit `if/elif` per provider and raises `ValueError("Unsupported provider: ...")` for anything else. Add an `elif provider == "yourprovider":` branch and set at least `mock_response.response_metadata` (e.g. to `metadata` or `{**metadata}`). Add `additional_kwargs` or `usage_metadata` only if your implementation reads them from the response.
- **`mock_llm_factory`** – Provider-agnostic (it just wraps whatever response you pass in). No conftest change needed.

Minimal addition in `conftest.py` inside `_create_mock_response`:

```python
elif provider == "yourprovider":
    mock_response.response_metadata = {**metadata}
```

If your implementation reads a specific shape (e.g. `response_metadata["model_name"]`), set those attributes on the mock so inherited metadata tests pass.

## Test Organization

```
tests/unit/llm_clients/
├── README.md              # This file
├── conftest.py            # Shared fixtures
├── test_helpers.py        # Reusable assertion functions
├── test_base_llm.py       # Base test classes
├── test_coverage.py       # Coverage validation tests
├── test_\*_llm.py          # solution-specific tests
├── test_llm_factory.py    # Factory tests
├── test_config.py         # Config tests
└── test_llm_interface.py  # Interface tests
```
