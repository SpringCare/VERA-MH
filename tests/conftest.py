from pathlib import Path

import pytest

from judge.rubric_config import RubricConfig
from tests.mocks.mock_llm import MockLLM


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
async def rubric_config_factory(fixtures_dir: Path):
    """Factory fixture for creating RubricConfig with custom rubric files."""

    async def _create_rubric_config(
        rubric_file: str = "rubric_single_row.tsv",
        rubric_prompt_beginning_file: str = "rubric_prompt_beginning.txt",
        question_prompt_file: str = "question_prompt.txt",
    ) -> RubricConfig:
        """Load a RubricConfig from test fixtures."""
        return await RubricConfig.load(
            rubric_folder=str(fixtures_dir),
            rubric_file=rubric_file,
            rubric_prompt_beginning_file=rubric_prompt_beginning_file,
            question_prompt_file=question_prompt_file,
        )

    return _create_rubric_config


@pytest.fixture
def mock_llm() -> MockLLM:
    """Basic mock LLM with default responses."""
    return MockLLM(responses=["Test response 1", "Test response 2"])


@pytest.fixture
def mock_persona() -> MockLLM:
    """Mock LLM configured as a persona."""
    from llm_clients.llm_interface import Role

    return MockLLM(
        name="mock-persona",
        role=Role.PERSONA,
        responses=["Hello, I need help", "I'm feeling anxious"],
    )


@pytest.fixture
def mock_agent() -> MockLLM:
    """Mock LLM configured as a chatbot agent."""
    from llm_clients.llm_interface import Role

    return MockLLM(
        name="mock-agent",
        role=Role.PROVIDER,
        responses=["How can I help you?", "Tell me more"],
    )


@pytest.fixture
def sample_conversation() -> list[dict]:
    """Sample conversation history."""
    return [
        {"turn": 1, "speaker": "persona", "response": "Hello"},
        {"turn": 2, "speaker": "provider", "response": "Hi, how can I help?"},
    ]
