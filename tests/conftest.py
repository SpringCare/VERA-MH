from pathlib import Path

import pytest

from tests.mocks.mock_llm import MockLLM


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_llm() -> MockLLM:
    """Basic mock LLM with default responses."""
    return MockLLM(responses=["Test response 1", "Test response 2"])


@pytest.fixture
def mock_persona() -> MockLLM:
    """Mock LLM configured as a persona."""
    return MockLLM(
        name="mock-persona", responses=["Hello, I need help", "I'm feeling anxious"]
    )


@pytest.fixture
def mock_agent() -> MockLLM:
    """Mock LLM configured as a chatbot agent."""
    return MockLLM(name="mock-agent", responses=["How can I help you?", "Tell me more"])


@pytest.fixture
def sample_conversation() -> list[dict]:
    """Sample conversation history."""
    return [
        {"turn": 1, "speaker": "User", "response": "Hello"},
        {"turn": 2, "speaker": "Chatbot", "response": "Hi, how can I help?"},
    ]
