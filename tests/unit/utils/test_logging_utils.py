"""Unit tests for utils/logging_utils.py"""

import logging
from pathlib import Path

import pytest

from llm_clients import Role
from tests.mocks.mock_llm import MockLLM
from utils.logging_utils import (
    cleanup_logger,
    log_conversation_end,
    log_conversation_start,
    log_conversation_turn,
    log_error,
    setup_conversation_logger,
)


@pytest.mark.unit
class TestSetupConversationLogger:
    """Test suite for setup_conversation_logger function"""

    def test_creates_logger_with_default_settings(self, tmp_path):
        """Test logger creation with default settings

        Arrange: Create log filename and run_id
        Act: Set up logger with tmp_path
        Assert: Logger is created with correct name and level
        """
        log_filename = "test_conversation"
        run_id = "run_001"
        log_folder = str(tmp_path / "logging")

        logger = setup_conversation_logger(
            log_filename=log_filename,
            run_id=run_id,
            log_folder=log_folder,
        )

        assert logger is not None
        assert logger.name == log_filename
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)

    def test_creates_log_folder_if_not_exists(self, tmp_path):
        """Test that logger creates log folder if it doesn't exist

        Arrange: Use non-existent log folder path
        Act: Set up logger
        Assert: Log folder is created
        """
        log_folder = str(tmp_path / "new_logging_folder")
        run_id = "run_002"

        setup_conversation_logger(
            log_filename="test",
            run_id=run_id,
            log_folder=log_folder,
        )

        assert Path(log_folder).exists()
        assert Path(log_folder, run_id).exists()

    def test_creates_log_file_with_correct_path(self, tmp_path):
        """Test that logger creates log file in correct location

        Arrange: Set up log path and filename
        Act: Create logger and write log message
        Assert: Log file exists in expected location
        """
        log_filename = "conversation_123"
        run_id = "run_003"
        log_folder = str(tmp_path / "logging")

        logger = setup_conversation_logger(
            log_filename=log_filename,
            run_id=run_id,
            log_folder=log_folder,
        )

        logger.info("Test message")

        expected_log_path = Path(log_folder) / run_id / f"{log_filename}.log"
        assert expected_log_path.exists()

    def test_custom_log_level(self, tmp_path):
        """Test logger creation with custom log level

        Arrange: Create logger with DEBUG level
        Act: Set up logger
        Assert: Logger has correct log level
        """
        logger = setup_conversation_logger(
            log_filename="test",
            run_id="run_004",
            log_folder=str(tmp_path / "logging"),
            level=logging.DEBUG,
        )

        assert logger.level == logging.DEBUG

    def test_logger_formatter_configuration(self, tmp_path):
        """Test that logger handler has correct formatter

        Arrange: Set up logger
        Act: Get handler formatter
        Assert: Formatter has expected format string
        """
        logger = setup_conversation_logger(
            log_filename="test",
            run_id="run_005",
            log_folder=str(tmp_path / "logging"),
        )

        handler = logger.handlers[0]
        formatter = handler.formatter

        assert formatter is not None
        assert "%(asctime)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt

    def test_clears_existing_handlers(self, tmp_path):
        """Test that setting up same logger clears old handlers

        Arrange: Create logger with one handler
        Act: Set up logger again with same name
        Assert: Old handlers are removed, only new handler exists
        """
        log_filename = "test_handler_clear"
        run_id = "run_006"
        log_folder = str(tmp_path / "logging")

        logger1 = setup_conversation_logger(
            log_filename=log_filename,
            run_id=run_id,
            log_folder=log_folder,
        )
        handler_count_1 = len(logger1.handlers)

        logger2 = setup_conversation_logger(
            log_filename=log_filename,
            run_id=run_id,
            log_folder=log_folder,
        )
        handler_count_2 = len(logger2.handlers)

        assert handler_count_1 == 1
        assert handler_count_2 == 1
        assert logger1.name == logger2.name

    def test_log_file_encoding_utf8(self, tmp_path):
        """Test that log file supports UTF-8 encoding

        Arrange: Set up logger
        Act: Log message with non-ASCII characters
        Assert: Message is written correctly to file
        """
        log_filename = "utf8_test"
        run_id = "run_007"
        log_folder = str(tmp_path / "logging")

        logger = setup_conversation_logger(
            log_filename=log_filename,
            run_id=run_id,
            log_folder=log_folder,
        )

        unicode_message = "Testing unicode: 你好, مرحبا, Здравствуйте"
        logger.info(unicode_message)

        log_file_path = Path(log_folder) / run_id / f"{log_filename}.log"
        content = log_file_path.read_text(encoding="utf-8")

        assert unicode_message in content


@pytest.mark.unit
class TestLogConversationStart:
    """Test suite for log_conversation_start function"""

    def test_logs_conversation_start_basic(self, tmp_path):
        """Test basic conversation start logging

        Arrange: Set up logger and mock LLMs
        Act: Log conversation start
        Assert: Log file contains start message and configuration
        """
        logger = setup_conversation_logger(
            log_filename="start_test",
            run_id="run_008",
            log_folder=str(tmp_path / "logging"),
        )

        llm1 = MockLLM(
            name="llm1",
            model_name="claude-3-opus",
            temperature=0.7,
            max_tokens=1000,
        )
        llm2 = MockLLM(
            name="llm2",
            model_name="gpt-4o",
            temperature=0.8,
            max_tokens=2000,
        )

        log_conversation_start(
            logger=logger,
            llm1_model_str="claude-3-opus",
            llm1_prompt="You are a helpful assistant",
            llm2_name="User",
            llm2_model_str="gpt-4o",
            max_turns=10,
            persona_speaks_first=True,
            llm1_model=llm1,
            llm2_model=llm2,
        )

        log_file = Path(tmp_path / "logging" / "run_008" / "start_test.log")
        content = log_file.read_text()

        assert "CONVERSATION STARTED" in content
        assert "claude-3-opus" in content
        assert "gpt-4o" in content
        assert "Max Turns: 10" in content
        assert "Persona speaks first: True" in content

    def test_logs_llm_configuration(self, tmp_path):
        """Test that LLM configuration is logged correctly

        Arrange: Set up logger with custom LLM configurations
        Act: Log conversation start
        Assert: Temperature and max_tokens are logged
        """
        logger = setup_conversation_logger(
            log_filename="config_test",
            run_id="run_009",
            log_folder=str(tmp_path / "logging"),
        )

        llm1 = MockLLM(temperature=0.5, max_tokens=500)
        llm2 = MockLLM(temperature=0.9, max_tokens=1500)

        log_conversation_start(
            logger=logger,
            llm1_model_str="model1",
            llm1_prompt="prompt1",
            llm2_name="model2",
            llm2_model_str="model2",
            max_turns=5,
            persona_speaks_first=False,
            llm1_model=llm1,
            llm2_model=llm2,
        )

        log_file = Path(tmp_path / "logging" / "run_009" / "config_test.log")
        content = log_file.read_text()

        assert "temperature: 0.5" in content
        assert "max_tokens: 500" in content
        assert "temperature: 0.9" in content
        assert "max_tokens: 1500" in content
        assert "Persona speaks first: False" in content

    def test_logs_with_empty_logging_dict(self, tmp_path):
        """Test logging with empty logging dictionary

        Arrange: Set up logger
        Act: Call with empty logging dict
        Assert: Function completes without error
        """
        logger = setup_conversation_logger(
            log_filename="empty_dict_test",
            run_id="run_010",
            log_folder=str(tmp_path / "logging"),
        )

        llm1 = MockLLM()
        llm2 = MockLLM()

        log_conversation_start(
            logger=logger,
            llm1_model_str="model1",
            llm1_prompt="prompt1",
            llm2_name="model2",
            llm2_model_str="model2",
            max_turns=5,
            persona_speaks_first=True,
            llm1_model=llm1,
            llm2_model=llm2,
            logging={},
        )

        log_file = Path(tmp_path / "logging" / "run_010" / "empty_dict_test.log")
        assert log_file.exists()


@pytest.mark.unit
class TestLogConversationTurn:
    """Test suite for log_conversation_turn function"""

    def test_logs_basic_turn(self, tmp_path):
        """Test logging a basic conversation turn

        Arrange: Set up logger
        Act: Log conversation turn
        Assert: Log contains turn number, speaker, input, response
        """
        logger = setup_conversation_logger(
            log_filename="turn_test",
            run_id="run_011",
            log_folder=str(tmp_path / "logging"),
        )

        log_conversation_turn(
            logger=logger,
            turn_number=1,
            speaker=Role.PERSONA.value,
            input_message="Hello, how are you?",
            response="I'm doing well, thanks!",
        )

        log_file = Path(tmp_path / "logging" / "run_011" / "turn_test.log")
        content = log_file.read_text()

        assert f"TURN 1 - {Role.PERSONA.value.upper()}" in content
        assert "Input: Hello, how are you?" in content
        assert "Response: I'm doing well, thanks!" in content

    def test_logs_turn_with_early_termination(self, tmp_path):
        """Test logging turn with early termination

        Arrange: Set up logger
        Act: Log turn with early_termination=True
        Assert: Log contains early termination warning
        """
        logger = setup_conversation_logger(
            log_filename="early_term_test",
            run_id="run_012",
            log_folder=str(tmp_path / "logging"),
        )

        log_conversation_turn(
            logger=logger,
            turn_number=5,
            speaker=Role.PROVIDER.value,
            input_message="Let's end this",
            response="Goodbye",
            early_termination=True,
        )

        log_file = Path(tmp_path / "logging" / "run_012" / "early_term_test.log")
        content = log_file.read_text()

        assert "EARLY TERMINATION" in content
        assert f"detected by {Role.PROVIDER.value.upper()}" in content

    def test_logs_turn_with_metadata(self, tmp_path):
        """Test logging turn with metadata dictionary

        Arrange: Set up logger
        Act: Log turn with logging dict containing response_id
        Assert: Log contains response_id
        """
        logger = setup_conversation_logger(
            log_filename="metadata_test",
            run_id="run_013",
            log_folder=str(tmp_path / "logging"),
        )

        metadata = {"response_id": "resp_123", "tokens": 150}

        log_conversation_turn(
            logger=logger,
            turn_number=2,
            speaker=Role.PROVIDER.value,
            input_message="What's the weather?",
            response="It's sunny today",
            logging=metadata,
        )

        log_file = Path(tmp_path / "logging" / "run_013" / "metadata_test.log")
        content = log_file.read_text()

        assert "response_id: resp_123" in content
        assert "tokens" in content

    def test_logs_multiple_turns(self, tmp_path):
        """Test logging multiple turns in sequence

        Arrange: Set up logger
        Act: Log multiple turns
        Assert: All turns are logged in order
        """
        logger = setup_conversation_logger(
            log_filename="multi_turn_test",
            run_id="run_014",
            log_folder=str(tmp_path / "logging"),
        )

        for i in range(1, 4):
            log_conversation_turn(
                logger=logger,
                turn_number=i,
                speaker=Role.PERSONA.value if i % 2 == 1 else Role.PROVIDER.value,
                input_message=f"Input {i}",
                response=f"Response {i}",
            )

        log_file = Path(tmp_path / "logging" / "run_014" / "multi_turn_test.log")
        content = log_file.read_text()

        assert f"TURN 1 - {Role.PERSONA.value.upper()}" in content
        assert f"TURN 2 - {Role.PROVIDER.value.upper()}" in content
        assert f"TURN 3 - {Role.PERSONA.value.upper()}" in content


@pytest.mark.unit
class TestLogConversationEnd:
    """Test suite for log_conversation_end function"""

    def test_logs_basic_end(self, tmp_path):
        """Test logging conversation end with basic info

        Arrange: Set up logger
        Act: Log conversation end
        Assert: Log contains completion message and turn count
        """
        logger = setup_conversation_logger(
            log_filename="end_test",
            run_id="run_015",
            log_folder=str(tmp_path / "logging"),
        )

        log_conversation_end(
            logger=logger,
            total_turns=10,
            early_termination=False,
        )

        log_file = Path(tmp_path / "logging" / "run_015" / "end_test.log")
        content = log_file.read_text()

        assert "CONVERSATION COMPLETED" in content
        assert "Total Turns: 10" in content
        assert "Early Termination: False" in content

    def test_logs_end_with_duration(self, tmp_path):
        """Test logging conversation end with duration

        Arrange: Set up logger
        Act: Log end with total_time
        Assert: Log contains duration in seconds
        """
        logger = setup_conversation_logger(
            log_filename="duration_test",
            run_id="run_016",
            log_folder=str(tmp_path / "logging"),
        )

        log_conversation_end(
            logger=logger,
            total_turns=5,
            early_termination=True,
            total_time=123.456,
        )

        log_file = Path(tmp_path / "logging" / "run_016" / "duration_test.log")
        content = log_file.read_text()

        assert "Duration: 123.46 seconds" in content

    def test_logs_end_with_early_termination(self, tmp_path):
        """Test logging with early termination flag

        Arrange: Set up logger
        Act: Log end with early_termination=True
        Assert: Log shows early termination
        """
        logger = setup_conversation_logger(
            log_filename="early_end_test",
            run_id="run_017",
            log_folder=str(tmp_path / "logging"),
        )

        log_conversation_end(
            logger=logger,
            total_turns=3,
            early_termination=True,
        )

        log_file = Path(tmp_path / "logging" / "run_017" / "early_end_test.log")
        content = log_file.read_text()

        assert "Early Termination: True" in content

    def test_logs_end_without_duration(self, tmp_path):
        """Test logging end without duration provided

        Arrange: Set up logger
        Act: Log end without total_time
        Assert: Log does not contain duration line
        """
        logger = setup_conversation_logger(
            log_filename="no_duration_test",
            run_id="run_018",
            log_folder=str(tmp_path / "logging"),
        )

        log_conversation_end(
            logger=logger,
            total_turns=7,
            early_termination=False,
            total_time=None,
        )

        log_file = Path(tmp_path / "logging" / "run_018" / "no_duration_test.log")
        content = log_file.read_text()

        assert "Duration:" not in content


@pytest.mark.unit
class TestLogError:
    """Test suite for log_error function"""

    def test_logs_error_message_only(self, tmp_path):
        """Test logging error message without exception

        Arrange: Set up logger
        Act: Log error message
        Assert: Error message is in log
        """
        logger = setup_conversation_logger(
            log_filename="error_test",
            run_id="run_019",
            log_folder=str(tmp_path / "logging"),
        )

        log_error(logger=logger, error_message="Something went wrong")

        log_file = Path(tmp_path / "logging" / "run_019" / "error_test.log")
        content = log_file.read_text()

        assert "ERROR: Something went wrong" in content

    def test_logs_error_with_exception(self, tmp_path):
        """Test logging error with exception object

        Arrange: Set up logger and create exception
        Act: Log error with exception
        Assert: Exception details are logged
        """
        logger = setup_conversation_logger(
            log_filename="exception_test",
            run_id="run_020",
            log_folder=str(tmp_path / "logging"),
        )

        try:
            raise ValueError("Invalid value provided")
        except ValueError as e:
            log_error(
                logger=logger,
                error_message="Value error occurred",
                exception=e,
            )

        log_file = Path(tmp_path / "logging" / "run_020" / "exception_test.log")
        content = log_file.read_text()

        assert "ERROR: Value error occurred" in content
        assert "Exception: Invalid value provided" in content
        assert "Exception Type: ValueError" in content


@pytest.mark.unit
class TestCleanupLogger:
    """Test suite for cleanup_logger function"""

    def test_cleanup_removes_handlers(self, tmp_path):
        """Test that cleanup removes all handlers

        Arrange: Set up logger with handlers
        Act: Call cleanup_logger
        Assert: Logger has no handlers
        """
        logger = setup_conversation_logger(
            log_filename="cleanup_test",
            run_id="run_021",
            log_folder=str(tmp_path / "logging"),
        )

        assert len(logger.handlers) > 0

        cleanup_logger(logger)

        assert len(logger.handlers) == 0

    def test_cleanup_closes_handlers(self, tmp_path):
        """Test that cleanup closes file handlers properly

        Arrange: Set up logger and write to it
        Act: Clean up logger
        Assert: File can be deleted (not locked)
        """
        log_filename = "close_test"
        run_id = "run_022"
        log_folder = str(tmp_path / "logging")

        logger = setup_conversation_logger(
            log_filename=log_filename,
            run_id=run_id,
            log_folder=log_folder,
        )

        logger.info("Test message")

        cleanup_logger(logger)

        log_file = Path(log_folder) / run_id / f"{log_filename}.log"
        log_file.unlink()

        assert not log_file.exists()

    def test_cleanup_empty_logger(self, tmp_path):
        """Test cleanup on logger with no handlers

        Arrange: Get logger with no handlers
        Act: Call cleanup_logger
        Assert: No error occurs
        """
        logger = logging.getLogger("empty_logger_test")
        logger.handlers.clear()

        cleanup_logger(logger)

        assert len(logger.handlers) == 0


@pytest.mark.unit
class TestMultipleLoggers:
    """Test suite for multiple logger interactions"""

    def test_multiple_loggers_independent(self, tmp_path):
        """Test that multiple loggers don't interfere

        Arrange: Create two loggers with different names
        Act: Log to both
        Assert: Each has separate log file with correct content
        """
        logger1 = setup_conversation_logger(
            log_filename="logger1",
            run_id="run_023",
            log_folder=str(tmp_path / "logging"),
        )

        logger2 = setup_conversation_logger(
            log_filename="logger2",
            run_id="run_023",
            log_folder=str(tmp_path / "logging"),
        )

        logger1.info("Message from logger1")
        logger2.info("Message from logger2")

        log1_file = Path(tmp_path / "logging" / "run_023" / "logger1.log")
        log2_file = Path(tmp_path / "logging" / "run_023" / "logger2.log")

        content1 = log1_file.read_text()
        content2 = log2_file.read_text()

        assert "Message from logger1" in content1
        assert "Message from logger2" not in content1

        assert "Message from logger2" in content2
        assert "Message from logger1" not in content2

    def test_multiple_loggers_different_run_ids(self, tmp_path):
        """Test loggers with different run_ids create separate folders

        Arrange: Create loggers with different run_ids
        Act: Log messages
        Assert: Separate folders and files are created
        """
        logger1 = setup_conversation_logger(
            log_filename="conv1",
            run_id="run_024",
            log_folder=str(tmp_path / "logging"),
        )

        logger2 = setup_conversation_logger(
            log_filename="conv2",
            run_id="run_025",
            log_folder=str(tmp_path / "logging"),
        )

        logger1.info("Run 024 message")
        logger2.info("Run 025 message")

        run1_folder = Path(tmp_path / "logging" / "run_024")
        run2_folder = Path(tmp_path / "logging" / "run_025")

        assert run1_folder.exists()
        assert run2_folder.exists()
        assert (run1_folder / "conv1.log").exists()
        assert (run2_folder / "conv2.log").exists()


@pytest.mark.unit
class TestFullConversationWorkflow:
    """Test suite for complete conversation logging workflow"""

    def test_complete_conversation_logging_flow(self, tmp_path):
        """Test complete conversation logging from start to end

        Arrange: Set up logger and mock LLMs
        Act: Log start, turns, and end
        Assert: All events are logged correctly in sequence
        """
        log_filename = "full_conversation"
        run_id = "run_026"
        log_folder = str(tmp_path / "logging")

        logger = setup_conversation_logger(
            log_filename=log_filename,
            run_id=run_id,
            log_folder=log_folder,
        )

        llm1 = MockLLM(
            name="assistant",
            model_name="claude-3-opus",
            temperature=0.7,
            max_tokens=1000,
        )
        llm2 = MockLLM(
            name="user",
            model_name="gpt-4o",
            temperature=0.8,
            max_tokens=2000,
        )

        log_conversation_start(
            logger=logger,
            llm1_model_str="claude-3-opus",
            llm1_prompt="You are helpful",
            llm2_name="User",
            llm2_model_str="gpt-4o",
            max_turns=3,
            persona_speaks_first=True,
            llm1_model=llm1,
            llm2_model=llm2,
        )

        log_conversation_turn(
            logger=logger,
            turn_number=1,
            speaker=Role.PERSONA.value,
            input_message="Hello",
            response="Hi there!",
        )

        log_conversation_turn(
            logger=logger,
            turn_number=2,
            speaker=Role.PROVIDER.value,
            input_message="Hi there!",
            response="How can I help?",
        )

        log_conversation_end(
            logger=logger,
            total_turns=2,
            early_termination=False,
            total_time=45.67,
        )

        cleanup_logger(logger)

        log_file = Path(log_folder) / run_id / f"{log_filename}.log"
        content = log_file.read_text()

        assert "CONVERSATION STARTED" in content
        assert "Persona speaks first: True" in content
        assert f"TURN 1 - {Role.PERSONA.value.upper()}" in content
        assert f"TURN 2 - {Role.PROVIDER.value.upper()}" in content
        assert "CONVERSATION COMPLETED" in content
        assert "Duration: 45.67 seconds" in content

        assert len(logger.handlers) == 0
