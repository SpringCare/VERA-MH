"""Integration tests for ConversationRunner.

Tests the full workflow of running conversations including file I/O,
logging, and batch processing with real file operations.
"""

import asyncio
import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from generate_conversations.runner import ConversationRunner
from llm_clients.llm_interface import Role
from tests.mocks.mock_llm import MockLLM


def create_test_runner(
    persona_config: Dict[str, Any],
    agent_config: Dict[str, Any],
    run_id: str,
    **kwargs: Dict[str, Any],
) -> ConversationRunner:
    """Helper to create runner with copied configs (no shared mutation)."""
    runner = ConversationRunner(
        persona_model_config=copy.deepcopy(persona_config),
        agent_model_config=copy.deepcopy(agent_config),
        run_id=run_id,
        **kwargs,
    )
    return runner


@pytest.fixture
def basic_persona_config() -> Dict[str, Any]:
    """Basic persona model configuration."""
    return {
        "model": "mock-persona-model",
        "temperature": 0.7,
        "max_tokens": 1000,
    }


@pytest.fixture
def basic_agent_config() -> Dict[str, Any]:
    """Basic agent model configuration."""
    return {
        "model": "mock-agent-model",
        "name": "mock-agent",
        "system_prompt": "You are a helpful AI assistant.",
        "temperature": 0.5,
        "max_tokens": 500,
    }


@pytest.fixture
def mock_llm_factory():
    """Mock LLMFactory.create_llm to return MockLLM instances."""

    def create_mock_llm(*args, **kwargs: Dict[str, Any]) -> MockLLM:
        """Create different mock LLMs based on name.

        This accepts *args and **kwargs to handle any calling convention,
        including duplicate keyword arguments that might be in kwargs.
        """
        # Extract parameters, handling duplicates by preferring kwargs
        model_name = kwargs.get("model_name", kwargs.get("model", "mock-model"))
        name = kwargs.get("name", "mock-llm")
        system_prompt = kwargs.get("system_prompt", None)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)

        if "persona" in name.lower() or "mock-persona" in model_name:
            return MockLLM(
                name=name,
                model_name=model_name,
                responses=[
                    "Hello, I need help with anxiety",
                    "Can you help me?",
                    "Thank you for listening",
                ],
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            return MockLLM(
                name=name,
                model_name=model_name,
                responses=[
                    "I'm here to help",
                    "Tell me more about what's going on",
                    "You're welcome",
                ],
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    with patch(
        "generate_conversations.runner.LLMFactory.create_llm",
        side_effect=create_mock_llm,
    ) as mock:
        yield mock


@pytest.fixture
def test_personas_csv(tmp_path: Path) -> Path:
    """Create a test personas CSV file."""
    csv_content = """Name\tAge\tRace/Ethnicity\tPronouns\tBackground
TestPersona1\t30\tWhite\the/him\tTest background 1
TestPersona2\t25\tAsian\tshe/her\tTest background 2"""

    csv_path = tmp_path / "test_personas.tsv"
    csv_path.write_text(csv_content)

    # Also create template file
    template_content = "Persona: {Name}, Age: {Age}"
    template_path = tmp_path / "test_template.txt"
    template_path.write_text(template_content)

    return csv_path


@pytest.mark.integration
class TestConversationRunnerInit:
    """Test ConversationRunner initialization (non-async tests)."""

    def test_init_with_basic_config(
        self,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
    ) -> None:
        """Test initialization with basic configuration."""
        # Arrange
        run_id = "test_run_123"

        # Act
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id=run_id,
        )

        # Assert
        assert runner.persona_model_config == basic_persona_config
        assert runner.agent_model_config["model"] == basic_agent_config["model"]
        assert runner.run_id == run_id
        assert runner.max_turns == 6
        assert runner.runs_per_prompt == 3
        assert runner.folder_name == "conversations"
        assert runner.max_concurrent is None

    def test_init_with_custom_parameters(
        self,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
    ) -> None:
        """Test initialization with custom parameters."""
        # Arrange & Act
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id="custom_run",
            max_turns=10,
            runs_per_prompt=5,
            folder_name="test_conversations",
            max_concurrent=3,
        )

        # Assert
        assert runner.max_turns == 10
        assert runner.runs_per_prompt == 5
        assert runner.folder_name == "test_conversations"
        assert runner.max_concurrent == 3

    @pytest.mark.asyncio
    async def test_agent_system_prompt_from_config(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Agent create_llm call receives system_prompt from config."""
        custom_prompt = "You are a mental health support chatbot."
        agent_config = copy.deepcopy(basic_agent_config)
        agent_config["system_prompt"] = custom_prompt

        create_llm_calls = []
        real_create = mock_llm_factory.side_effect

        def recording_create_llm(*args: Any, **kwargs: Any) -> MockLLM:
            create_llm_calls.append(copy.deepcopy(kwargs))
            return real_create(*args, **kwargs)

        mock_llm_factory.side_effect = recording_create_llm

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=agent_config,
            run_id="test_run",
            folder_name=str(tmp_path / "conversations"),
        )
        persona_config = {
            "prompt": "Test persona prompt",
            "name": "TestPersona",
            "run": 1,
        }

        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            mock_logger.return_value = MagicMock()
            await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )

        agent_calls = [c for c in create_llm_calls if c.get("role") == Role.PROVIDER]
        assert len(agent_calls) == 1
        assert agent_calls[0]["system_prompt"] == custom_prompt

    @pytest.mark.asyncio
    async def test_agent_system_prompt_default(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """When agent config has no system_prompt, create_llm gets the fallback."""
        default_prompt = "You are a helpful AI assistant."
        agent_config = {
            "model": "mock-agent",
            "name": "test-agent",
        }
        create_llm_calls = []
        real_create = mock_llm_factory.side_effect

        def recording_create_llm(*args: Any, **kwargs: Any) -> MockLLM:
            create_llm_calls.append(copy.deepcopy(kwargs))
            return real_create(*args, **kwargs)

        mock_llm_factory.side_effect = recording_create_llm

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=agent_config,
            run_id="test_run",
            folder_name=str(tmp_path / "conversations"),
        )
        persona_config = {
            "prompt": "Test persona prompt",
            "name": "TestPersona",
            "run": 1,
        }

        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            mock_logger.return_value = MagicMock()
            await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )

        agent_calls = [c for c in create_llm_calls if c.get("role") == Role.PROVIDER]
        assert len(agent_calls) == 1
        assert agent_calls[0]["system_prompt"] == default_prompt


@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationRunnerSingle:
    """Test running single conversations."""

    async def test_run_single_conversation_creates_files(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that single conversation creates conversation and log."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        log_folder = tmp_path / "logging"
        run_id = "test_run_001"

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id=run_id,
            folder_name=str(conv_folder),
        )

        persona_config = {
            "prompt": "Test persona prompt",
            "name": "TestPersona",
            "run": 1,
        }

        # Act - patch setup_conversation_logger to use tmp_path
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = logging.getLogger("test_conversation")
            logger.handlers.clear()
            os.makedirs(log_folder / run_id, exist_ok=True)
            handler = logging.FileHandler(log_folder / run_id / "test.log", mode="w")
            logger.addHandler(handler)
            mock_logger.return_value = logger

            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=4,
                conversation_index=1,
                run_number=1,
            )

        # Assert - verify result structure
        assert result["index"] == 1
        assert result["llm1_model"] == "mock-persona-model"
        assert result["llm1_prompt"] == "TestPersona"
        assert result["run_number"] == 1
        assert len(result["sessions"]) == 1
        assert result["sessions"][0]["turns"] == 4
        assert result["duration"] > 0
        assert len(result["sessions"][0]["conversation"]) == 4
        assert Path(result["sessions"][0]["filename"]).exists()

    async def test_persona_speaks_first_false_first_turn_is_provider(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test persona_speaks_first=False: first turn is from the provider."""
        conv_folder = tmp_path / "conversations"
        run_id = "test_agent_first"

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id=run_id,
            folder_name=str(conv_folder),
            persona_speaks_first=False,
        )

        persona_config = {
            "prompt": "Test persona prompt",
            "name": "TestPersona",
            "run": 1,
        }

        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = logging.getLogger("test_agent_first")
            logger.handlers.clear()
            mock_logger.return_value = logger

            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=3,
                conversation_index=1,
                run_number=1,
            )

        conv = result["sessions"][0]["conversation"]
        assert len(conv) == 3
        assert conv[0]["speaker"] == "provider"
        assert conv[-1]["speaker"] == "provider"

    async def test_run_single_conversation_logging(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that conversation logging works correctly."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        log_folder = tmp_path / "logging"
        run_id = "test_run_002"

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id=run_id,
            folder_name=str(conv_folder),
        )

        persona_config = {
            "prompt": "Test persona prompt",
            "name": "TestPersona",
            "run": 1,
        }

        # Act - use real logging with tmp_path
        os.makedirs(log_folder / run_id, exist_ok=True)
        log_file = log_folder / run_id / "test_conversation.log"

        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger_setup:
            logger = logging.getLogger("test_conversation_logging")
            logger.handlers.clear()
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            mock_logger_setup.return_value = logger

            await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )

            # Cleanup logger handlers
            for h in logger.handlers[:]:
                h.close()
                logger.removeHandler(h)

        # Assert - verify log file exists and has content
        assert log_file.exists()
        log_content = log_file.read_text(encoding="utf-8")
        assert "CONVERSATION STARTED" in log_content
        assert "CONVERSATION COMPLETED" in log_content
        assert "TURN 1" in log_content

    async def test_filename_generation_format(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that filename follows correct naming convention."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        persona_model_config = {
            **basic_persona_config,
            "model": "claude-3-opus-20240229",
        }
        runner = ConversationRunner(
            persona_model_config=persona_model_config,
            agent_model_config=basic_agent_config,
            run_id="test_run",
            folder_name=str(conv_folder),
        )

        persona_config = {
            "prompt": "Test prompt",
            "name": "Persona",
            "run": 2,
        }

        # Act
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=2,
            )

        # Assert - verify filename format
        filename = Path(result["sessions"][0]["filename"]).name
        # Should contain: tag_personaname_modelshort_runN
        assert "Persona" in filename
        assert "claude-3-opus-20240229" in filename
        assert "run2" in filename
        assert filename.endswith(".txt")

    async def test_early_termination_tracking(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
    ) -> None:
        """Test that early termination is tracked correctly."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id="test_run",
            folder_name=str(conv_folder),
        )

        persona_config = {
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        # Create mock with early termination
        with patch("generate_conversations.runner.LLMFactory.create_llm") as mock:
            persona_mock = MockLLM(
                name="persona",
                model_name="mock-persona-model",
                responses=["Hello", "Goodbye, I'm done now"],
                temperature=0.7,
                max_tokens=1000,
            )
            agent_mock = MockLLM(
                name="agent",
                model_name="mock-agent-model",
                responses=["Hi", "Bye"],
                temperature=0.5,
                max_tokens=500,
            )

            def side_effect(*args, **kwargs):
                if "persona" in kwargs.get("name", "").lower():
                    return persona_mock
                return agent_mock

            mock.side_effect = side_effect

            # Patch conversation simulator to detect early termination
            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                logger = MagicMock()
                mock_logger.return_value = logger

                # Act
                result = await runner.run_single_conversation(
                    persona_config=persona_config,
                    max_turns=10,
                    conversation_index=1,
                    run_number=1,
                )

        assert "early_termination" in result["sessions"][0]

    async def test_conversation_metadata_completeness(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that all metadata fields are present in result."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id="test_run",
            folder_name=str(conv_folder),
        )

        persona_config = {
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        # Act
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=5,
                run_number=3,
            )

        # Assert - verify all required metadata fields
        required_fields = [
            "index",
            "llm1_model",
            "llm1_prompt",
            "run_number",
            "sessions",
            "duration",
            "skipped",
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        assert isinstance(result["sessions"], list)
        assert len(result["sessions"]) == 1
        assert isinstance(result["sessions"][0]["turns"], int)
        assert isinstance(result["sessions"][0]["conversation"], list)


@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationRunnerMultiple:
    """Test running multiple conversations."""

    async def test_run_multiple_conversations_from_personas(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test running multiple conversations from persona list."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=2,
            folder_name=str(conv_folder),
        )

        # Mock load_prompts_from_csv
        mock_personas = [
            {"Name": "Persona1", "prompt": "Prompt 1"},
            {"Name": "Persona2", "prompt": "Prompt 2"},
        ]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas

            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                logger = MagicMock()
                mock_logger.return_value = logger

                # Act
                results = await runner.run_conversations(
                    persona_names=["Persona1", "Persona2"]
                )

        # Assert - should run 2 personas * 2 runs each = 4 conversations
        assert len(results) == 4
        assert all(isinstance(r, dict) for r in results)
        assert all(r["sessions"] and r["sessions"][0]["conversation"] for r in results)

    async def test_concurrent_execution_limit(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that max_concurrent limits concurrent execution."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=3,
            folder_name=str(conv_folder),
            max_concurrent=2,
        )

        mock_personas = [
            {"Name": "Persona1", "prompt": "Prompt 1"},
            {"Name": "Persona2", "prompt": "Prompt 2"},
        ]

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent_seen = 0

        original_run = runner.run_single_conversation

        async def tracked_run(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.01)
            result = await original_run(*args, **kwargs)
            concurrent_count -= 1
            return result

        runner.run_single_conversation = tracked_run

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas

            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                logger = MagicMock()
                mock_logger.return_value = logger

                # Act
                results = await runner.run_conversations(persona_names=None)

        # Assert - should respect max_concurrent
        assert len(results) == 6  # 2 personas * 3 runs
        # Due to async nature, max_concurrent_seen might be 2 or less
        assert max_concurrent_seen <= 2

    async def test_no_concurrent_limit(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test running without concurrent limit."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=2,
            folder_name=str(conv_folder),
            max_concurrent=None,
        )

        mock_personas = [{"Name": "Persona1", "prompt": "Prompt 1"}]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas

            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                logger = MagicMock()
                mock_logger.return_value = logger

                # Act
                results = await runner.run_conversations(persona_names=None)

        # Assert
        assert len(results) == 2  # 1 persona * 2 runs

    async def test_conversation_index_increments(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that conversation indices increment correctly."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=2,
            folder_name=str(conv_folder),
        )

        mock_personas = [
            {"Name": "Persona1", "prompt": "Prompt 1"},
            {"Name": "Persona2", "prompt": "Prompt 2"},
        ]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas

            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                logger = MagicMock()
                mock_logger.return_value = logger

                # Act
                results = await runner.run_conversations(persona_names=None)

        # Assert - indices should be 1, 2, 3, 4
        indices = sorted([r["index"] for r in results])
        assert indices == [1, 2, 3, 4]

    async def test_resume_mode_skips_existing_persona_run(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Resume mode skips transcript when persona/run already exists."""
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir(parents=True, exist_ok=True)
        # Existing transcript for Persona1 run1
        (conv_folder / "abc123_Persona1_mock-persona-model_run1.txt").write_text(
            "user: hi\n\nchatbot: hello\n", encoding="utf-8"
        )

        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=2,
            folder_name=str(conv_folder),
            resume=True,
        )
        mock_personas = [{"Name": "Persona1", "prompt": "Prompt 1"}]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas
            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                mock_logger.return_value = MagicMock()
                results = await runner.run_conversations(persona_names=None)

        assert len(results) == 2
        assert sum(1 for r in results if r.get("skipped")) == 1
        assert sum(1 for r in results if not r.get("skipped")) == 1

    async def test_non_resume_mode_does_not_skip_existing_persona_run(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Without resume mode, existing files do not suppress new runs."""
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir(parents=True, exist_ok=True)
        (conv_folder / "abc123_Persona1_mock-persona-model_run1.txt").write_text(
            "user: hi\n\nchatbot: hello\n", encoding="utf-8"
        )

        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=2,
            folder_name=str(conv_folder),
            resume=False,
        )
        mock_personas = [{"Name": "Persona1", "prompt": "Prompt 1"}]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas
            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                mock_logger.return_value = MagicMock()
                results = await runner.run_conversations(persona_names=None)

        assert len(results) == 2
        assert all(not r.get("skipped") for r in results)

    async def test_resume_mode_skips_existing_persona_with_underscore_name(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Resume matching handles persona names that include underscores."""
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir(parents=True, exist_ok=True)
        # Simulates: "Mary Jane" -> "Mary_Jane" and model short like g5.2
        (conv_folder / "707d63_Mary_Jane_mock-persona-model_run1.txt").write_text(
            "user: hi\n\nchatbot: hello\n", encoding="utf-8"
        )

        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=1,
            folder_name=str(conv_folder),
            resume=True,
        )
        mock_personas = [{"Name": "Mary Jane", "prompt": "Prompt 1"}]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas
            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                mock_logger.return_value = MagicMock()
                results = await runner.run_conversations(persona_names=None)

        assert len(results) == 1
        assert results[0].get("skipped") is True

    async def test_resume_mode_skips_when_model_short_has_underscores(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Resume resolves persona when model_short contains underscores."""
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir(parents=True, exist_ok=True)
        (conv_folder / "abc123_Persona1_gpt_4_turbo_run1.txt").write_text(
            "user: hi\n\nchatbot: hello\n", encoding="utf-8"
        )

        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=2,
            folder_name=str(conv_folder),
            resume=True,
        )
        mock_personas = [{"Name": "Persona1", "prompt": "Prompt 1"}]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas
            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                mock_logger.return_value = MagicMock()
                results = await runner.run_conversations(persona_names=None)

        assert len(results) == 2
        assert sum(1 for r in results if r.get("skipped")) == 1
        assert sum(1 for r in results if not r.get("skipped")) == 1

    async def test_resume_mode_skips_multi_session_transcripts(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Resume recognizes *_runN_s{i}_{session_type}.txt transcript names."""
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir(parents=True, exist_ok=True)
        intake = conv_folder / "abc123_Persona1_mock-persona-model_run1_s1_intake.txt"
        coaching = (
            conv_folder / "def456_Persona1_mock-persona-model_run1_s2_coaching.txt"
        )
        intake.write_text("user: hi\n\nchatbot: hello\n", encoding="utf-8")
        coaching.write_text("user: hi\n\nchatbot: hello\n", encoding="utf-8")

        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=1,
            folder_name=str(conv_folder),
            resume=True,
            session_types=["intake", "coaching"],
        )
        mock_personas = [{"Name": "Persona1", "prompt": "Prompt 1"}]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas
            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                mock_logger.return_value = MagicMock()
                results = await runner.run_conversations(persona_names=None)

        assert len(results) == 1
        assert results[0].get("skipped") is True
        assert results[0].get("skip_reason") == "existing"

    async def test_resume_mode_runs_remaining_multi_session_only(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Resume with partial multi-session work runs only missing sessions."""
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir(parents=True, exist_ok=True)
        intake_path = (
            conv_folder / "abc123_Persona1_mock-persona-model_run1_s1_intake.txt"
        )
        intake_path.write_text("user: hi\n\nchatbot: hello\n", encoding="utf-8")

        agent_mock = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["reply"] * 10,
            model_name="mock-agent-model",
        )
        persona_mock = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["hi"] * 10,
            model_name="mock-persona-model",
        )

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config={"model": "mock-agent-model", "name": "agent"},
            run_id="test_partial_resume",
            folder_name=str(conv_folder),
            max_turns=2,
            runs_per_prompt=1,
            resume=True,
            session_types=["intake", "coaching"],
        )
        persona_config = {
            "model": "mock-persona-model",
            "prompt": "Test prompt",
            "name": "Persona1",
            "run": 1,
        }

        with (
            patch(
                "generate_conversations.runner.LLMFactory.create_llm",
                side_effect=[persona_mock, agent_mock],
            ),
            patch(
                "generate_conversations.runner.setup_conversation_logger",
                return_value=MagicMock(),
            ),
        ):
            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )

        assert len(result["sessions"]) == 2
        assert result["sessions"][0]["skipped"] is True
        assert result["sessions"][0]["skip_reason"] == "existing"
        assert result["sessions"][0]["filename"] == str(intake_path)
        assert result["sessions"][1]["skipped"] is False
        assert result["sessions"][1]["turns"] > 0

    async def test_agent_config_not_mutated_across_concurrent_conversations(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Ensure agent_model_config is not mutated when running multiple conversations.

        Each run_single_conversation creates an agent from the shared config; the config
        must remain intact so every conversation gets the same name and system_prompt.
        """
        expected_name = "shared-agent-name"
        expected_prompt = "Shared system prompt for all runs."

        create_llm_calls = []
        real_create = mock_llm_factory.side_effect

        def recording_create_llm(*args: Any, **kwargs: Any) -> MockLLM:
            create_llm_calls.append(copy.deepcopy(kwargs))
            return real_create(*args, **kwargs)

        mock_llm_factory.side_effect = recording_create_llm

        agent_config = copy.deepcopy(basic_agent_config)
        agent_config["name"] = expected_name
        agent_config["system_prompt"] = expected_prompt

        conv_folder = tmp_path / "conversations"
        runner = create_test_runner(
            basic_persona_config,
            agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=2,
            folder_name=str(conv_folder),
        )
        mock_personas = [
            {"Name": "Persona1", "prompt": "Prompt 1"},
            {"Name": "Persona2", "prompt": "Prompt 2"},
        ]

        with (
            patch("generate_conversations.runner.load_prompts_from_csv") as mock_load,
            patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger,
        ):
            mock_load.return_value = mock_personas
            mock_logger.return_value = MagicMock()
            await runner.run_conversations(persona_names=None)

        assert runner.agent_model_config["name"] == expected_name
        assert runner.agent_model_config["system_prompt"] == expected_prompt

        agent_calls = [
            c
            for c in create_llm_calls
            if c.get("model_name") == basic_agent_config["model"]
        ]
        assert (
            len(agent_calls) == 4
        ), "Expected 4 conversations => 4 agent create_llm calls"
        assert all(c.get("name") == expected_name for c in agent_calls)
        assert all(c.get("system_prompt") == expected_prompt for c in agent_calls)


@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationRunnerFileOperations:
    """Test file creation and organization."""

    async def test_conversation_folder_creation(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that conversation folder is created if it doesn't exist."""
        # Arrange
        conv_folder = tmp_path / "new_conversations"
        assert not conv_folder.exists()

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id="test_run",
            folder_name=str(conv_folder),
        )

        persona_config = {
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        # Act
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )

        # Assert
        assert conv_folder.exists()
        assert conv_folder.is_dir()

    async def test_multiple_files_created(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that multiple conversation files are created."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=3,
            folder_name=str(conv_folder),
        )

        mock_personas = [{"Name": "TestPersona", "prompt": "Test prompt"}]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas

            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                logger = MagicMock()
                mock_logger.return_value = logger

                # Act
                results = await runner.run_conversations(persona_names=None)

        # Assert - should create 3 conversation files (nested layout)
        assert len(results) == 3
        txt_files = list((conv_folder / "conversations").glob("*.txt"))
        assert len(txt_files) == 3

    async def test_filename_uniqueness(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that each conversation gets a unique filename."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=3,
            folder_name=str(conv_folder),
        )

        mock_personas = [{"Name": "TestPersona", "prompt": "Test prompt"}]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas

            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                logger = MagicMock()
                mock_logger.return_value = logger

                # Act
                results = await runner.run_conversations(persona_names=None)

        # Assert - all filenames should be unique
        filenames = [r["sessions"][0]["filename"] for r in results]
        assert len(filenames) == len(set(filenames))


@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationRunnerErrorHandling:
    """Test error handling and edge cases."""

    async def test_llm_errors_propagate_from_run_single_conversation(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
    ) -> None:
        """When the agent LLM fails, the run is skipped and no transcript is saved."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id="test_run",
            folder_name=str(conv_folder),
        )

        persona_config = {
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        # Create agent that will error; persona mock for the other create_llm call
        error_agent = MockLLM(
            name="error-agent",
            model_name="mock-error-model",
            responses=[],
            simulate_error=True,
            temperature=0.5,
            max_tokens=500,
        )
        persona_mock = MockLLM(
            name="persona",
            model_name="mock-persona-model",
            responses=["Hello"],
            temperature=0.7,
            max_tokens=1000,
        )

        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            with patch(
                "generate_conversations.runner.LLMFactory.create_llm"
            ) as mock_factory:
                # run_single_conversation creates persona first, then agent
                mock_factory.side_effect = [persona_mock, error_agent]

                # Act
                result = await runner.run_single_conversation(
                    persona_config=persona_config,
                    max_turns=2,
                    conversation_index=1,
                    run_number=1,
                )

                assert result.get("skipped") is True
                assert "Simulated API error" in result.get("error", "")
                txt_files = list((conv_folder / "conversations").glob("*.txt"))
                assert txt_files == []

    async def test_empty_persona_list(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test handling of empty persona list."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            folder_name=str(conv_folder),
        )

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = []

            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                logger = MagicMock()
                mock_logger.return_value = logger

                # Act
                results = await runner.run_conversations(persona_names=None)

        # Assert - should return empty list
        assert results == []

    async def test_logger_cleanup_called(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that logger cleanup is called after conversation."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id="test_run",
            folder_name=str(conv_folder),
        )

        persona_config = {
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        # Act
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            with patch("generate_conversations.runner.cleanup_logger") as mock_cleanup:
                await runner.run_single_conversation(
                    persona_config=persona_config,
                    max_turns=2,
                    conversation_index=1,
                    run_number=1,
                )

                # Assert: session logger cleaned after failure; finally may call again.
                assert mock_cleanup.call_count >= 1


@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationRunnerPerformance:
    """Test performance-related aspects."""

    async def test_duration_tracking(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that conversation duration is tracked accurately."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id="test_run",
            folder_name=str(conv_folder),
        )

        persona_config = {
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        # Act
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            start = time.time()
            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )
            end = time.time()

        # Assert
        assert result["duration"] > 0
        assert result["duration"] < (end - start) + 0.1  # Allow small margin

    async def test_batch_processing_speed(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that batch processing completes in reasonable time."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = create_test_runner(
            basic_persona_config,
            basic_agent_config,
            "test_run",
            max_turns=2,
            runs_per_prompt=2,
            folder_name=str(conv_folder),
        )

        mock_personas = [
            {"Name": "Persona1", "prompt": "Prompt 1"},
            {"Name": "Persona2", "prompt": "Prompt 2"},
        ]

        with patch("generate_conversations.runner.load_prompts_from_csv") as mock_load:
            mock_load.return_value = mock_personas

            with patch(
                "generate_conversations.runner.setup_conversation_logger"
            ) as mock_logger:
                logger = MagicMock()
                mock_logger.return_value = logger

                # Act
                start = time.time()
                results = await runner.run_conversations(persona_names=None)
                duration = time.time() - start

        # Assert - should complete quickly with mocks
        assert len(results) == 4
        assert duration < 2.0  # Should be fast with mocks


class SessionTrackingAgent(MockLLM):
    """MockLLM that records session lifecycle calls
    and supports first_speaker override."""

    def __init__(
        self,
        *args,
        first_speaker_role=None,
        extra_sessions=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setup_call_count = 0
        self.enter_session_calls: list = []
        self.prepare_sessions_calls: list = []
        self._first_speaker_role = first_speaker_role
        self._extra_sessions = extra_sessions or []

    async def setup(self):
        self.setup_call_count += 1

    async def enter_session(self, session_type: str):
        self.enter_session_calls.append(session_type)

    def prepare_sessions(self, session_types):
        self.prepare_sessions_calls.append(list(session_types))
        return self._extra_sessions + list(session_types)

    @property
    def first_speaker(self):
        return self._first_speaker_role


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultiSessionRunner:
    """Tests for multi-session runner behavior: setup hooks, first_speaker, sessions."""

    async def test_agent_setup_called_once_before_sessions(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
    ) -> None:
        """agent.setup() is called exactly once before the session loop."""
        conv_folder = tmp_path / "conversations"
        tracking_agent = SessionTrackingAgent(
            name="agent",
            role=Role.PROVIDER,
            responses=["reply"] * 10,
            model_name="mock-agent-model",
        )
        persona_mock = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["hi"] * 10,
            model_name="mock-persona-model",
        )

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config={"model": "mock-agent-model", "name": "agent"},
            run_id="test_setup",
            folder_name=str(conv_folder),
        )
        persona_config = {
            "model": "mock-persona-model",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        with (
            patch(
                "generate_conversations.runner.LLMFactory.create_llm",
                side_effect=[persona_mock, tracking_agent],
            ),
            patch(
                "generate_conversations.runner.setup_conversation_logger",
                return_value=MagicMock(),
            ),
        ):
            await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )

        assert tracking_agent.setup_call_count == 1

    async def test_enter_session_called_for_each_session(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
    ) -> None:
        """enter_session is called once per session."""
        conv_folder = tmp_path / "conversations"
        tracking_agent = SessionTrackingAgent(
            name="agent",
            role=Role.PROVIDER,
            responses=["reply"] * 10,
            model_name="mock-agent-model",
        )
        persona_mock = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["hi"] * 10,
            model_name="mock-persona-model",
        )

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config={"model": "mock-agent-model", "name": "agent"},
            run_id="test_reset",
            folder_name=str(conv_folder),
            session_types=["intake", "coaching"],
        )
        persona_config = {
            "model": "mock-persona-model",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        with (
            patch(
                "generate_conversations.runner.LLMFactory.create_llm",
                side_effect=[persona_mock, tracking_agent],
            ),
            patch(
                "generate_conversations.runner.setup_conversation_logger",
                return_value=MagicMock(),
            ),
        ):
            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )

        assert tracking_agent.enter_session_calls == ["intake", "coaching"]
        assert len(result["sessions"]) == 2

    async def test_prepare_sessions_result_used_as_session_list(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
    ) -> None:
        """prepare_sessions() return value is used as the actual session list."""
        conv_folder = tmp_path / "conversations"
        # Agent prepends "intake" to any session list
        tracking_agent = SessionTrackingAgent(
            name="agent",
            role=Role.PROVIDER,
            responses=["reply"] * 10,
            model_name="mock-agent-model",
            extra_sessions=["intake"],
        )
        persona_mock = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["hi"] * 10,
            model_name="mock-persona-model",
        )

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config={"model": "mock-agent-model", "name": "agent"},
            run_id="test_prepare",
            folder_name=str(conv_folder),
            session_types=["coaching"],
        )
        persona_config = {
            "model": "mock-persona-model",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        with (
            patch(
                "generate_conversations.runner.LLMFactory.create_llm",
                side_effect=[persona_mock, tracking_agent],
            ),
            patch(
                "generate_conversations.runner.setup_conversation_logger",
                return_value=MagicMock(),
            ),
        ):
            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )

        # prepare_sessions was called with ["coaching"], returned ["intake", "coaching"]
        assert tracking_agent.prepare_sessions_calls == [["coaching"]]
        assert tracking_agent.enter_session_calls == ["intake", "coaching"]
        assert len(result["sessions"]) == 2

    async def test_first_speaker_overrides_persona_speaks_first(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
    ) -> None:
        """agent.first_speaker returning PROVIDER forces provider-first even when
        runner.persona_speaks_first is True."""
        conv_folder = tmp_path / "conversations"
        tracking_agent = SessionTrackingAgent(
            name="agent",
            role=Role.PROVIDER,
            responses=["provider opening"] + ["reply"] * 10,
            model_name="mock-agent-model",
            first_speaker_role=Role.PROVIDER,
        )
        persona_mock = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["persona reply"] * 10,
            model_name="mock-persona-model",
        )

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config={"model": "mock-agent-model", "name": "agent"},
            run_id="test_first_speaker",
            folder_name=str(conv_folder),
            persona_speaks_first=True,  # runner default: persona first
        )
        persona_config = {
            "model": "mock-persona-model",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        with (
            patch(
                "generate_conversations.runner.LLMFactory.create_llm",
                side_effect=[persona_mock, tracking_agent],
            ),
            patch(
                "generate_conversations.runner.setup_conversation_logger",
                return_value=MagicMock(),
            ),
        ):
            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=3,
                conversation_index=1,
                run_number=1,
            )

        # Despite persona_speaks_first=True, agent.first_speaker=PROVIDER wins
        assert result["sessions"][0]["conversation"][0]["speaker"] == "provider"

    async def test_session_suffix_in_filenames_when_session_types_set(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
    ) -> None:
        """When session_types is explicitly set,
        filenames include session index and type."""
        conv_folder = tmp_path / "conversations"
        agent_mock = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["reply"] * 10,
            model_name="mock-agent-model",
        )
        persona_mock = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["hi"] * 10,
            model_name="mock-persona-model",
        )

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config={"model": "mock-agent-model", "name": "agent"},
            run_id="test_suffix",
            folder_name=str(conv_folder),
            session_types=["intake", "coaching"],
        )
        persona_config = {
            "model": "mock-persona-model",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        with (
            patch(
                "generate_conversations.runner.LLMFactory.create_llm",
                side_effect=[persona_mock, agent_mock],
            ),
            patch(
                "generate_conversations.runner.setup_conversation_logger",
                return_value=MagicMock(),
            ),
        ):
            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )

        assert len(result["sessions"]) == 2
        assert "1_intake" in result["sessions"][0]["filename"]
        assert "2_coaching" in result["sessions"][1]["filename"]
        assert "conversation" not in result

    async def test_multi_session_results_kept_separate(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
    ) -> None:
        """Each session has its own turns and transcript;
        job-level list is not merged."""
        conv_folder = tmp_path / "conversations"
        agent_mock = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["reply"] * 20,
            model_name="mock-agent-model",
        )
        persona_mock = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["hi"] * 20,
            model_name="mock-persona-model",
        )

        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config={"model": "mock-agent-model", "name": "agent"},
            run_id="test_concat",
            folder_name=str(conv_folder),
            session_types=["intake", "coaching"],
        )
        persona_config = {
            "model": "mock-persona-model",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        with (
            patch(
                "generate_conversations.runner.LLMFactory.create_llm",
                side_effect=[persona_mock, agent_mock],
            ),
            patch(
                "generate_conversations.runner.setup_conversation_logger",
                return_value=MagicMock(),
            ),
        ):
            result = await runner.run_single_conversation(
                persona_config=persona_config,
                max_turns=2,
                conversation_index=1,
                run_number=1,
            )

        assert len(result["sessions"]) == 2
        assert "turns" not in result
        for session in result["sessions"]:
            assert session["turns"] > 0
            assert len(session["conversation"]) == session["turns"]
