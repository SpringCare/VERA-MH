# /create-tests - Unified Test Creation Command

Create tests for VERA-MH following project conventions.

## Usage

### Focused Mode (Single Module)
`/create-tests <module_path> [--layer=unit|integration|e2e]`

Create tests for a specific module:
- `/create-tests llm_clients/claude_llm.py` - Ask which layer
- `/create-tests judge/score.py --layer=unit` - Create unit tests directly
- `/create-tests conversation_runner.py --layer=integration` - Create integration tests

### Coverage Analysis Mode (Multiple Modules)
`/create-tests`

Analyze coverage and systematically create tests for gaps.

---

## Workflow

### When module_path is provided (Focused Mode):

1. **Validate Input**
   - Check if module_path exists
   - If not found, search for similar files and suggest alternatives

2. **Determine Test Layer**
   - If --layer flag provided, use it
   - Otherwise, ask: "Which test layer? (unit/integration/e2e)"
   - Show guidance:
     - **unit**: Test individual functions/classes in isolation
     - **integration**: Test interactions between components
     - **e2e**: Test complete workflows

3. **Analyze Target Module**
   - Identify testable functions and classes
   - Detect async functions (need @pytest.mark.asyncio)
   - Identify LLM client usage (suggest MockLLM)
   - Detect file I/O operations (suggest tmp_path fixture)
   - Detect Pydantic models (suggest example instances)
   - Identify external dependencies (suggest appropriate mocking)

4. **Determine Test File Path**
   Based on layer:
   - Unit tests: `tests/unit/[module_path]/test_[filename].py`
   - Integration tests: `tests/integration/test_[feature].py`
   - E2E tests: `tests/e2e/test_[workflow].py`

5. **Create Test File**
   Include:
   - Proper imports (pytest, fixtures from conftest)
   - MockLLM import if needed
   - pytest.mark.asyncio for async tests
   - Appropriate fixtures (tmp_path, mock_llm, etc.)
   - Test stubs for each function with descriptive docstrings
   - Test markers (@pytest.mark.unit, @pytest.mark.integration, etc.)
   - Example test data and assertions

6. **Provide Guidance**
   - What to test (happy path, edge cases, errors)
   - How to use MockLLM for this project
   - How to handle async code
   - Relevant fixtures from conftest.py
   - Testing patterns specific to detected code patterns

7. **Verify**
   - Run: `pytest <new_test_file> -v` to ensure structure is valid
   - Report: "✅ Created [path] (N test stubs)"
   - If tests fail, show errors and offer to fix

### When no parameters (Coverage Analysis Mode):

1. **Run Coverage Analysis**
   - Execute: `uv run pytest --cov=vera_mh --cov-report=term-missing --cov-report=json`
   - Parse coverage.json report
   - Identify modules with coverage below quality target (75%+)
   - Note: Project has dual coverage targets:
     - **30%** = CI minimum (must pass for all code)
     - **75%+** = Quality target (goal for new/changed code)

2. **Prioritize Modules**
   Rank by importance:
   - **Critical**: Mental health safety code, prompt handling, data validation
   - **High**: LLM clients, conversation generation, evaluation/judge
   - **Medium**: Utilities, configuration, data loading
   - **Low**: Scripts, examples, tooling

3. **Analyze Coverage Gaps**
   For each low-coverage module:
   - Identify uncovered lines (from term-missing report)
   - Determine why not covered (untested functions, edge cases, error paths)
   - Assess complexity and risk
   - Note any existing tests that could be extended

4. **Present Coverage Report**
   Show user detailed breakdown:
   ```
   Coverage Report (Current: XX%, Quality Target: 75%):

   🔴 Critical - Needs Attention:
     module_name.py: XX% (↓XX% from quality target)
       Uncovered: Lines X-Y (description), Lines A-B (description)
       Risk: High/Medium/Low

   🟡 High Priority - Below Target:
     module_name.py: XX% (↓XX% from target)
       Uncovered: Lines X-Y (description)

   🟢 Good Coverage:
     module_name.py: XX%
   ```

5. **Get User Approval**
   Ask which modules to create tests for:
   - **all-critical** - All critical modules (recommended)
   - **all-below-target** - All modules below 75%
   - **select** - Choose specific modules interactively
   - **cancel** - Exit without creating tests

6. **Create Tests**
   For each approved module:
   - Use focused mode workflow (steps 3-7 from above)
   - Analyze the module's functions and uncovered lines
   - Create comprehensive test file
   - Include unit tests for all uncovered functions
   - Add edge case and error path tests
   - Use appropriate fixtures (MockLLM, tmp_path, etc.)
   - Focus on the uncovered line ranges identified

7. **Verify Improvement**
   - Run all new tests: `pytest <new_test_files> -v`
   - Re-run coverage: `uv run pytest --cov=vera_mh --cov-report=json`
   - Parse new coverage data
   - Show before/after comparison per module and overall
   - Report: "Coverage improved from XX% → YY% (+ZZ%)"
   - Highlight any modules that still need attention

---

## Examples

### Focused Mode Examples

```bash
# Create unit tests for a specific module
/create-tests llm_clients/claude_llm.py --layer=unit
→ Analyzing llm_clients/claude_llm.py...
→ Found: ClaudeLLM class with 8 methods (3 async)
→ Detected: LLM client usage, async operations, error handling
→ Creating: tests/unit/llm_clients/test_claude_llm.py
→ ✅ Created tests/unit/llm_clients/test_claude_llm.py (12 test stubs)

# Let Claude ask which layer
/create-tests judge/score_calculator.py
→ Which test layer? (unit/integration/e2e)
→ User: unit
→ Analyzing judge/score_calculator.py...
→ Found: calculate_score(), aggregate_scores(), 4 helper functions
→ Creating: tests/unit/judge/test_score_calculator.py
→ ✅ Created tests/unit/judge/test_score_calculator.py (8 test stubs)

# Create integration tests
/create-tests conversation_runner.py --layer=integration
→ Analyzing conversation_runner.py...
→ Found: ConversationRunner class, async run(), file I/O
→ Detected: LLM client usage, file operations, async workflows
→ Creating: tests/integration/test_conversation_runner.py
→ ✅ Created tests/integration/test_conversation_runner.py (6 test cases)
```

### Coverage Analysis Mode Example

```bash
/create-tests

→ Running coverage analysis...
→ Executing: uv run pytest --cov=vera_mh --cov-report=term-missing --cov-report=json

Coverage Report (Current: 65%, Quality Target: 75%):

🔴 Critical - Needs Attention:
  llm_clients/claude_llm.py: 45% (↓30% from quality target)
    Uncovered: Lines 78-95 (error handling), 120-135 (retry logic)
    Risk: High - Error handling for mental health content

  conversation_simulator.py: 58% (↓17%)
    Uncovered: Lines 45-52 (early termination), 89-103 (async errors)
    Risk: Medium - Conversation flow edge cases

🟡 High Priority - Below Target:
  judge/score_calculator.py: 68% (↓7%)
    Uncovered: Lines 156-170 (concurrent execution errors)
    Risk: Medium - Score calculation edge cases

🟢 Good Coverage:
  utils/prompt_loader.py: 88%
  utils/model_config_loader.py: 100%

Create tests for which modules?
(all-critical/all-below-target/select/cancel)
> all-critical

Creating tests for llm_clients/claude_llm.py...
→ Analyzing module structure...
→ Creating comprehensive test suite...
✅ Created tests/unit/llm_clients/test_claude_llm.py (15 test cases)

Creating tests for conversation_simulator.py...
→ Analyzing module structure...
→ Creating comprehensive test suite...
✅ Created tests/unit/test_conversation_simulator.py (12 test cases)

Running new tests to verify...
→ pytest tests/unit/llm_clients/test_claude_llm.py tests/unit/test_conversation_simulator.py -v
✅ All 27 new tests pass

Re-analyzing coverage...
→ uv run pytest --cov=vera_mh --cov-report=json

Coverage improved: 65% → 78% (+13%)

Module improvements:
  llm_clients/claude_llm.py: 45% → 87% (+42%)
  conversation_simulator.py: 58% → 81% (+23%)

🎉 Target achieved! Critical modules now above 75%.
```

---

## Test File Patterns

All test files should follow these patterns:

### File Structure
```python
"""Test module for [module_name].

Tests cover:
- [Function/class 1]: happy path, edge cases, errors
- [Function/class 2]: happy path, edge cases, errors
"""

import pytest
from vera_mh.module.filename import ClassOrFunction
from tests.mocks.mock_llm import MockLLM  # if needed

@pytest.mark.unit  # or @pytest.mark.integration, @pytest.mark.e2e
class TestClassName:
    """Test suite for ClassName"""

    def test_happy_path(self):
        """Test normal operation with valid inputs"""
        # Arrange
        # Act
        # Assert
        pass

    def test_edge_case_empty_input(self):
        """Test handling of empty input"""
        pass

    def test_error_handling_invalid_input(self):
        """Test error handling for invalid input"""
        with pytest.raises(ValueError):
            pass
```

### MockLLM Usage
```python
from tests.mocks.mock_llm import MockLLM

@pytest.mark.asyncio
async def test_llm_interaction(mock_llm):
    """Test LLM interaction with mocked responses"""
    llm = MockLLM(mock_responses=["Test response"])
    result = await llm.ainvoke("Test prompt")
    assert result == "Test response"
```

### File Operations
```python
def test_file_operations(tmp_path):
    """Test file reading/writing with temporary directory"""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Test your file operations
    result = read_file(test_file)
    assert result == "test content"
```

### Async Functions
```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function execution"""
    result = await async_function()
    assert result is not None
```

### Pydantic Models
```python
from vera_mh.models import ConversationConfig

def test_pydantic_model_validation():
    """Test Pydantic model validation"""
    # Valid case
    config = ConversationConfig(model="claude-3-7-sonnet", turns=5)
    assert config.model == "claude-3-7-sonnet"

    # Invalid case
    with pytest.raises(ValidationError):
        ConversationConfig(model="invalid", turns=-1)
```

### Fixtures from conftest.py
```python
def test_with_mock_llm(mock_llm):
    """Test using the mock_llm fixture from conftest.py"""
    # mock_llm is automatically available
    pass

def test_with_sample_config(sample_config):
    """Test using sample_config fixture"""
    # sample_config provides test configuration
    pass
```

---

## Testing Conventions

### Test Markers
- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, components interact)
- `@pytest.mark.e2e` - End-to-end tests (slowest, full workflows)
- `@pytest.mark.asyncio` - Async tests (required for async functions)

### Test Naming
- Test files: `test_[module_name].py`
- Test classes: `TestClassName` (matches class being tested)
- Test functions: `test_[function]_[scenario]` (descriptive)
  - Examples: `test_calculate_score_with_valid_input`
  - `test_parse_config_raises_on_invalid_json`

### Test Organization
- **Arrange-Act-Assert** pattern
- One assertion per test (when possible)
- Clear test docstrings explaining what's being tested
- Group related tests in classes
- Use fixtures for common setup

### Coverage Goals
- **30%** minimum (CI requirement)
- **75%** quality target (new/changed code)
- **100%** for critical safety code (mental health prompts, validation)

---

## Benefits of Unified Command

**For Daily Development:**
- ✅ Quick test creation: `/create-tests mymodule.py`
- ✅ One command to remember
- ✅ Natural workflow integration

**For Strategic Coverage:**
- ✅ Data-driven gap analysis
- ✅ Prioritized by business criticality
- ✅ Batch test creation
- ✅ Before/after metrics

**For Team Consistency:**
- ✅ Consistent test patterns
- ✅ Proper fixture usage
- ✅ Standard file organization
- ✅ Coverage-aware development
