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
from tests.mocks.mock_llm import MockLLM


def create_test_runner(
    persona_config: Dict[str, Any],
    agent_config: Dict[str, Any],
    run_id: str,
    **kwargs: Dict[str, Any],
) -> ConversationRunner:
    """Helper to create runner and clean up agent_model_config.

    This removes system_prompt from agent_model_config to avoid duplicate
    keyword argument errors when run_conversations calls create_llm.
    """
    agent_cfg = copy.deepcopy(agent_config)
    runner = ConversationRunner(
        persona_model_config=copy.deepcopy(persona_config),
        agent_model_config=agent_cfg,
        run_id=run_id,
        **kwargs,
    )
    # Remove system_prompt to avoid duplicate kwarg in run_conversations
    runner.agent_model_config.pop("system_prompt", None)
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
    """Basic agent model configuration.

    Note: This includes system_prompt which will be extracted in __init__
    and used separately in run_conversations. The config will be mutated
    (name popped, system_prompt passed separately).
    """
    # Don't include system_prompt in the config dict since it causes
    # duplicate keyword argument errors when run_conversations calls
    # create_llm with both system_prompt as arg and **config
    # However, we do include it initially so __init__ can extract it
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
        # Note: agent_model_config will still have system_prompt
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

    def test_agent_system_prompt_from_config(
        self,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
    ) -> None:
        """Test that agent system prompt is extracted from config."""
        # Arrange
        custom_prompt = "You are a mental health support chatbot."
        basic_agent_config["system_prompt"] = custom_prompt

        # Act
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id="test_run",
        )

        # Assert
        assert runner.AGENT_SYSTEM_PROMPT == custom_prompt

    def test_agent_system_prompt_default(
        self,
        basic_persona_config: Dict[str, Any],
    ) -> None:
        """Test default agent system prompt when not in config."""
        # Arrange
        agent_config = {
            "model": "mock-agent",
            "name": "test-agent",
        }

        # Act
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=agent_config,
            run_id="test_run",
        )

        # Assert
        assert runner.AGENT_SYSTEM_PROMPT == "You are a helpful AI assistant."


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
            "model": "mock-persona-model",
            "prompt": "Test persona prompt",
            "name": "TestPersona",
            "run": 1,
        }

        agent = MockLLM(
            name="test-agent",
            model_name="mock-agent-model",
            responses=["I'm here to help", "Tell me more"],
            temperature=0.5,
            max_tokens=500,
        )

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
                agent=agent,
                max_turns=4,
                conversation_id=1,
                run_number=1,
            )

        # Assert - verify result structure
        assert result["id"] == 1
        assert result["llm1_model"] == "mock-persona-model"
        assert result["llm1_prompt"] == "TestPersona"
        assert result["run_number"] == 1
        assert result["turns"] == 4
        assert "filename" in result
        assert "log_file" in result
        assert result["duration"] > 0
        assert isinstance(result["conversation"], list)
        assert len(result["conversation"]) == 4

        # Verify conversation file exists
        assert Path(result["filename"]).exists()

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
            "model": "mock-persona-model",
            "prompt": "Test persona prompt",
            "name": "TestPersona",
            "run": 1,
        }

        agent = MockLLM(
            name="test-agent",
            model_name="mock-agent-model",
            responses=["Response 1"],
            temperature=0.5,
            max_tokens=500,
        )

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
                agent=agent,
                max_turns=2,
                conversation_id=1,
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
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id="test_run",
            folder_name=str(conv_folder),
        )

        persona_config = {
            "model": "claude-3-opus-20240229",
            "prompt": "Test prompt",
            "name": "Test Persona",
            "run": 2,
        }

        agent = MockLLM(
            name="test-agent",
            model_name="mock-agent-model",
            responses=["Response"],
            temperature=0.5,
            max_tokens=500,
        )

        # Act
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            result = await runner.run_single_conversation(
                persona_config=persona_config,
                agent=agent,
                max_turns=2,
                conversation_id=1,
                run_number=2,
            )

        # Assert - verify filename format
        filename = Path(result["filename"]).name
        # Should contain: tag_personaname_modelshort_runN
        assert "Test_Persona" in filename
        assert "c3-opus-20240229" in filename
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
            "model": "mock-persona",
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
                    agent=agent_mock,
                    max_turns=10,
                    conversation_id=1,
                    run_number=1,
                )

        # Assert - check if early termination flag is present
        assert "early_termination" in result
        # Note: actual termination depends on simulator's signal detection

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
            "model": "mock-persona",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        agent = MockLLM(
            name="test-agent",
            model_name="mock-agent-model",
            responses=["Response"],
            temperature=0.5,
            max_tokens=500,
        )

        # Act
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            result = await runner.run_single_conversation(
                persona_config=persona_config,
                agent=agent,
                max_turns=2,
                conversation_id=5,
                run_number=3,
            )

        # Assert - verify all required metadata fields
        required_fields = [
            "id",
            "llm1_model",
            "llm1_prompt",
            "run_number",
            "turns",
            "filename",
            "log_file",
            "duration",
            "early_termination",
            "conversation",
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Verify types
        assert isinstance(result["id"], int)
        assert isinstance(result["turns"], int)
        assert isinstance(result["duration"], float)
        assert isinstance(result["early_termination"], bool)
        assert isinstance(result["conversation"], list)


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
        assert all("conversation" in r for r in results)

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

    async def test_conversation_id_increments(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
        mock_llm_factory,
    ) -> None:
        """Test that conversation IDs increment correctly."""
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

        # Assert - IDs should be 1, 2, 3, 4
        ids = sorted([r["id"] for r in results])
        assert ids == [1, 2, 3, 4]


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
            "model": "mock-persona",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        agent = MockLLM(
            name="test-agent",
            model_name="mock-agent-model",
            responses=["Response"],
            temperature=0.5,
            max_tokens=500,
        )

        # Act
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            await runner.run_single_conversation(
                persona_config=persona_config,
                agent=agent,
                max_turns=2,
                conversation_id=1,
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

        # Assert - should create 3 conversation files
        assert len(results) == 3
        txt_files = list(conv_folder.glob("*.txt"))
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
        filenames = [r["filename"] for r in results]
        assert len(filenames) == len(set(filenames))


@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationRunnerErrorHandling:
    """Test error handling and edge cases."""

    async def test_handles_llm_errors_gracefully(
        self,
        tmp_path: Path,
        basic_persona_config: Dict[str, Any],
        basic_agent_config: Dict[str, Any],
    ) -> None:
        """Test that LLM errors are handled gracefully."""
        # Arrange
        conv_folder = tmp_path / "conversations"
        runner = ConversationRunner(
            persona_model_config=basic_persona_config,
            agent_model_config=basic_agent_config,
            run_id="test_run",
            folder_name=str(conv_folder),
        )

        persona_config = {
            "model": "mock-persona",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        # Create agent that will error
        error_agent = MockLLM(
            name="error-agent",
            model_name="mock-error-model",
            responses=[],
            simulate_error=True,
            temperature=0.5,
            max_tokens=500,
        )

        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            with patch(
                "generate_conversations.runner.LLMFactory.create_llm"
            ) as mock_factory:
                mock_factory.return_value = MockLLM(
                    name="persona",
                    model_name="mock-persona-model",
                    responses=["Hello"],
                    temperature=0.7,
                    max_tokens=1000,
                )

                # Act & Assert - should raise the error
                with pytest.raises(Exception) as exc_info:
                    await runner.run_single_conversation(
                        persona_config=persona_config,
                        agent=error_agent,
                        max_turns=2,
                        conversation_id=1,
                        run_number=1,
                    )

                assert "Simulated API error" in str(exc_info.value)

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
            "model": "mock-persona",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        agent = MockLLM(
            name="test-agent",
            model_name="mock-agent-model",
            responses=["Response"],
            temperature=0.5,
            max_tokens=500,
        )

        # Act
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            with patch("generate_conversations.runner.cleanup_logger") as mock_cleanup:
                await runner.run_single_conversation(
                    persona_config=persona_config,
                    agent=agent,
                    max_turns=2,
                    conversation_id=1,
                    run_number=1,
                )

                # Assert
                mock_cleanup.assert_called_once_with(logger)


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
            "model": "mock-persona",
            "prompt": "Test prompt",
            "name": "TestPersona",
            "run": 1,
        }

        agent = MockLLM(
            name="test-agent",
            model_name="mock-agent-model",
            responses=["Response"],
            temperature=0.5,
            max_tokens=500,
        )

        # Act
        with patch(
            "generate_conversations.runner.setup_conversation_logger"
        ) as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            start = time.time()
            result = await runner.run_single_conversation(
                persona_config=persona_config,
                agent=agent,
                max_turns=2,
                conversation_id=1,
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
