"""Shared pytest fixtures for all unit tests.

This module provides fixtures that are used across multiple test directories.
"""

import pytest


@pytest.fixture
def mock_system_message():
    """Mock system message for basic tests.

    Returns a simple special-case turn 0 message that can be used in most tests
    where the specific message content doesn't matter.
    """
    return [{"turn": 0, "response": "Test"}]
