"""Test helper functions for LLM client tests.

This module provides reusable assertion and validation functions that reduce
code duplication and improve test readability across all LLM client test files.

The helpers are organized into the following categories:

1. Metadata Assertions
   - assert_metadata_structure(): Validates LLM metadata fields
   - assert_iso_timestamp(): Validates ISO timestamp format
   - assert_metadata_copy_behavior(): Verifies copy behavior
   - assert_response_timing(): Validates timing fields

2. Response Assertions
   - assert_error_response(): Validates error message format
   - assert_error_metadata(): Validates error metadata structure

3. Mock Verification
   - verify_no_system_message_in_call(): Checks system message absence
   - verify_message_types_for_persona(): Validates persona role message flipping
"""

from datetime import datetime
from typing import Any, Dict, Optional

from llm_clients import Role
from llm_clients.llm_interface import LLMInterface

# ============================================================================
# Metadata Assertions
# ============================================================================


def assert_metadata_structure(
    llm: LLMInterface,
    expected_provider: str,
    expected_role: Optional[Role] = None,
    require_response_id: bool = False,
    require_usage: bool = False,
) -> Dict[str, Any]:
    """Assert that LLM metadata has expected structure and fields.

    Args:
        llm: LLM instance to check metadata on
        expected_provider: Expected provider name ("claude", "openai", "gemini", etc.)
        expected_role: Expected role (if None, doesn't check)
        require_response_id: Whether response_id must be non-None
        require_usage: Whether usage dict must have token counts

    Returns:
        The metadata dict for further assertions

    Raises:
        AssertionError: If metadata structure is invalid
    """
    metadata = llm.get_last_response_metadata()

    # Check required fields exist
    assert "model" in metadata, "Metadata missing 'model' field"
    assert "provider" in metadata, "Metadata missing 'provider' field"
    assert "timestamp" in metadata, "Metadata missing 'timestamp' field"

    # Check provider matches
    assert (
        metadata["provider"] == expected_provider
    ), f"Expected provider '{expected_provider}', got '{metadata['provider']}'"

    # Check role if provided
    if expected_role is not None:
        assert "role" in metadata, "Metadata missing 'role' field"
        assert (
            metadata["role"] == expected_role.value
        ), f"Expected role '{expected_role.value}', got '{metadata['role']}'"

    # Check response_id if required
    if require_response_id:
        assert metadata.get("response_id") is not None, "response_id should not be None"

    # Check usage if required
    if require_usage:
        assert "usage" in metadata, "Metadata missing 'usage' field"
        usage = metadata["usage"]
        assert isinstance(usage, dict), "usage should be a dict"
        assert len(usage) > 0, "usage dict should not be empty"

    return metadata


def assert_iso_timestamp(timestamp: str) -> None:
    """Assert that timestamp string is valid ISO format.

    Args:
        timestamp: Timestamp string to validate

    Raises:
        AssertionError: If timestamp is not valid ISO format
    """
    try:
        datetime.fromisoformat(timestamp)
    except (ValueError, TypeError) as e:
        raise AssertionError(f"Invalid ISO timestamp '{timestamp}': {e}") from e


def assert_metadata_copy_behavior(llm: LLMInterface) -> None:
    """Assert that get_last_response_metadata returns a copy.

    Verifies that:
    1. Multiple calls return equal but different objects
    2. Modifying returned dict doesn't affect internal state

    Args:
        llm: LLM instance to test

    Raises:
        AssertionError: If copy behavior is incorrect
    """
    # Set some test metadata
    llm.last_response_metadata = {"test": "value"}

    metadata1 = llm.get_last_response_metadata()
    metadata2 = llm.get_last_response_metadata()

    # Should be equal but not the same object
    assert metadata1 == metadata2, "Multiple calls should return equal dicts"
    assert metadata1 is not metadata2, "Should return different objects (copies)"

    # Modifying returned copy shouldn't affect internal state
    metadata1["modified"] = True
    assert (
        "modified" not in llm.last_response_metadata
    ), "Modification leaked to internal state"


def assert_response_timing(metadata: Dict[str, Any]) -> None:
    """Assert that metadata contains valid response timing information.

    Args:
        metadata: Metadata dict to check

    Raises:
        AssertionError: If timing information is invalid
    """
    assert "response_time_seconds" in metadata, "Missing response_time_seconds"
    response_time = metadata["response_time_seconds"]
    assert isinstance(
        response_time, (int, float)
    ), f"response_time_seconds should be numeric, got {type(response_time)}"
    assert (
        response_time >= 0
    ), f"response_time_seconds should be >= 0, got {response_time}"


def assert_error_metadata(
    llm: LLMInterface,
    expected_provider: str,
    expected_error_substring: str,
) -> None:
    """Assert that error metadata is properly structured.

    Args:
        llm: LLM instance to check metadata on
        expected_provider: Expected provider name
        expected_error_substring: Substring that should appear in error message

    Raises:
        AssertionError: If error metadata is invalid
    """
    metadata = llm.get_last_response_metadata()

    # Check error field exists and contains expected substring
    assert "error" in metadata, "Metadata missing 'error' field"
    assert expected_error_substring in metadata["error"], (
        f"Expected error to contain '{expected_error_substring}', "
        f"got: {metadata['error']}"
    )

    # Check other required fields
    assert metadata["response_id"] is None, "response_id should be None on error"
    assert metadata["provider"] == expected_provider
    assert "timestamp" in metadata
    assert metadata["usage"] == {}


# ============================================================================
# Response Assertions
# ============================================================================


def assert_error_response(response: str, expected_error_substring: str) -> None:
    """Assert that response is an error message with expected content.

    Args:
        response: Response string to check
        expected_error_substring: Substring expected in error message

    Raises:
        AssertionError: If response doesn't match error pattern
    """
    assert (
        "Error generating response" in response
    ), "Response should start with error prefix"
    assert (
        expected_error_substring in response
    ), f"Expected error to contain '{expected_error_substring}', got: {response}"


# ============================================================================
# Mock Verification
# ============================================================================


def verify_no_system_message_in_call(mock_llm) -> None:
    """Verify that no system message was included in ainvoke call.

    Args:
        mock_llm: Mock LLM instance to check

    Raises:
        AssertionError: If system message found
    """
    assert mock_llm.ainvoke.called, "ainvoke should have been called"
    call_args = mock_llm.ainvoke.call_args[0][0]

    from langchain_core.messages import SystemMessage

    # Check that first message is NOT a SystemMessage
    if len(call_args) > 0:
        first_msg = call_args[0]
        assert not isinstance(
            first_msg, SystemMessage
        ), "First message should not be SystemMessage when no system prompt"


def verify_message_types_for_persona(mock_llm, expected_message_count: int) -> None:
    """Verify message types are flipped correctly for persona role.

    For persona role:
    - Persona messages (odd turns) should be AIMessage
    - Provider messages (even turns) should be HumanMessage

    Args:
        mock_llm: Mock LLM instance to check
        expected_message_count: Expected number of messages (including SystemMessage)

    Raises:
        AssertionError: If message types are incorrect
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    assert mock_llm.ainvoke.called, "ainvoke should have been called"
    messages = mock_llm.ainvoke.call_args[0][0]

    assert (
        len(messages) == expected_message_count
    ), f"Expected {expected_message_count} messages, got {len(messages)}"

    # First message should be SystemMessage
    assert isinstance(
        messages[0], SystemMessage
    ), "First message should be SystemMessage"

    # Verify subsequent messages are correctly flipped
    # (This assumes a 3-turn conversation: persona, provider, persona)
    if len(messages) >= 4:
        assert isinstance(
            messages[1], AIMessage
        ), "Turn 1 (persona) should be AIMessage for persona role"
        assert isinstance(
            messages[2], HumanMessage
        ), "Turn 2 (provider) should be HumanMessage for persona role"
        assert isinstance(
            messages[3], AIMessage
        ), "Turn 3 (persona) should be AIMessage for persona role"
