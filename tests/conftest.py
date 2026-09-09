import os
import shutil
import tempfile
from pathlib import Path

import pytest

from judge.rubric_config import RubricConfig
from tests.mocks.mock_llm import MockLLM

_judge_logs_temp: Path | None = None
_adhoc_temp: Path | None = None


def pytest_sessionstart(session: pytest.Session) -> None:
    """Point judge file logs and adhoc outputs at temp dirs (cleaned in finish)."""
    global _judge_logs_temp, _adhoc_temp
    _judge_logs_temp = Path(tempfile.mkdtemp(prefix="vera_judge_logs_"))
    os.environ["VERA_JUDGE_LOGS_ROOT"] = str(_judge_logs_temp)
    _adhoc_temp = Path(tempfile.mkdtemp(prefix="vera_adhoc_"))
    os.environ["VERA_ADHOC_PARENT"] = str(_adhoc_temp)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Remove temp dirs; clear env so CLI uses normal defaults."""
    global _judge_logs_temp, _adhoc_temp
    os.environ.pop("VERA_JUDGE_LOGS_ROOT", None)
    os.environ.pop("VERA_ADHOC_PARENT", None)
    if _judge_logs_temp is not None and _judge_logs_temp.exists():
        shutil.rmtree(_judge_logs_temp, ignore_errors=True)
    _judge_logs_temp = None
    if _adhoc_temp is not None and _adhoc_temp.exists():
        shutil.rmtree(_adhoc_temp, ignore_errors=True)
    _adhoc_temp = None


_CREDENTIAL_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ENDPOINT_API_KEY",
)

_DUMMY_CREDENTIAL = "test-dummy-not-a-real-key"


@pytest.fixture(autouse=True)
def no_real_credentials(request: pytest.FixtureRequest, monkeypatch) -> None:
    """Replace real provider credentials with dummies outside live tests.

    llm_clients.config calls load_dotenv() at import time, so a developer's real
    keys are present in os.environ during the suite. A test missing the `live`
    marker would therefore make real, billable API calls locally and only fail in
    CI, where no keys exist. This also covers the gap in --disable-socket, which
    cannot see into the subprocesses the live tests spawn, since children inherit
    this environment.

    The values are overwritten rather than deleted on purpose: importlib.reload
    on the config module re-runs load_dotenv(), which would repopulate a deleted
    variable from .env but leaves an existing one alone.
    """
    if "live" in request.keywords:
        return
    for var in _CREDENTIAL_ENV_VARS:
        monkeypatch.setenv(var, _DUMMY_CREDENTIAL)


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
