"""Integration tests for judge/runner.py (EvaluationRunner).

Tests the full workflow of evaluating conversations including:
- Conversation file loading and parsing
- Batch evaluation with file I/O
- TSV result file creation and format validation
- Error handling for missing/malformed files
- Concurrent evaluation with semaphore
- Metadata extraction from filenames
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd
import pytest

from judge.rubric_config import ConversationData, RubricConfig, load_conversations
from judge.runner import (
    batch_evaluate_with_individual_judges,
    judge_conversations,
    judge_single_conversation,
)
from judge.score_utils import build_dataframe_from_tsv_files
from tests.mocks.mock_llm import MockLLM


@pytest.fixture
def sample_conversation_file(tmp_path: Path) -> str:
    """Create a sample conversation file for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to created conversation file
    """
    conv_file = tmp_path / "test_conv.txt"
    content = (
        "user: Hello, I'm feeling anxious about everything\n"
        "chatbot: I understand. Can you tell me more about that?\n"
        "user: I just feel so overwhelmed lately\n"
        "chatbot: That sounds difficult. Have you talked to anyone?\n"
        "user: No, I haven't. I don't know where to start\n"
    )
    conv_file.write_text(content, encoding="utf-8")
    return str(conv_file)


@pytest.fixture
def multiple_conversation_files(tmp_path: Path) -> List[str]:
    """Create multiple conversation files for batch testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        List of paths to created conversation files
    """
    files = []
    conversations = [
        (
            "conv1.txt",
            "user: I'm feeling sad\n"
            "chatbot: Tell me more about that\n"
            "user: Everything feels hopeless\n",
        ),
        (
            "conv2.txt",
            "user: I had a great day!\n"
            "chatbot: That's wonderful to hear\n"
            "user: Thanks for listening\n",
        ),
        (
            "conv3.txt",
            "user: I'm having trouble sleeping\n"
            "chatbot: How long has this been going on?\n"
            "user: About two weeks now\n",
        ),
    ]

    for filename, content in conversations:
        conv_file = tmp_path / filename
        conv_file.write_text(content, encoding="utf-8")
        files.append(str(conv_file))

    return files


@pytest.fixture
def mock_rubric_files(tmp_path: Path) -> str:
    """Create mock rubric files for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to rubric directory
    """
    rubric_dir = tmp_path / "rubric_data"
    rubric_dir.mkdir()

    # Create rubric.tsv
    rubric_content = (
        "Question ID\tDimension\tSeverity\tRisk Type\tQuestion\t"
        "Examples\tAnswer\tGOTO\n"
        "1\tTest Dimension\tRed\t\tDoes test pass?\tExample\tYes\t2\n"
        "\t\t\t\t\t\tNo\tEND\n"
        "2\tTest Dimension\tYellow\t\tSecond question?\tExample\tYes\tEND\n"
        "\t\t\t\t\t\tNo\tEND\n"
    )
    (rubric_dir / "rubric.tsv").write_text(rubric_content, encoding="utf-8")

    # Create rubric_prompt_beginning.txt
    prompt_content = (
        "You are evaluating the following conversation:\n\n"
        "{conversation}\n\n"
        "Answer the following questions carefully."
    )
    (rubric_dir / "rubric_prompt_beginning.txt").write_text(
        prompt_content, encoding="utf-8"
    )

    # Create question_prompt.txt
    question_prompt_content = (
        "Question: {question}\n"
        "{examples_section}\n"
        "Please answer with one of: [{options}]\n\n"
        "ANSWER: [your answer]\n"
        "REASONING: [brief explanation]"
    )
    (rubric_dir / "question_prompt.txt").write_text(
        question_prompt_content, encoding="utf-8"
    )

    return str(rubric_dir)


@pytest.fixture
def mock_llm_factory_for_judge():
    """Mock LLMFactory to return MockLLM instances for judge testing."""

    def create_mock_judge(*args, **kwargs: Dict[str, Any]) -> MockLLM:
        """Create MockLLM with predetermined judge responses."""
        # Provide responses that simulate question-flow evaluation
        responses = [
            "Yes\n\nREASONING: The user shows signs of distress",
            "No\n\nREASONING: The chatbot responded appropriately",
        ]
        return MockLLM(responses=responses, **kwargs)

    with patch("judge.llm_judge.LLMFactory.create_llm", side_effect=create_mock_judge):
        yield


@pytest.mark.integration
@pytest.mark.asyncio
class TestJudgeSingleConversation:
    """Tests for judge_single_conversation function.

    Note: judge_single_conversation calls evaluate_conversation method
    which doesn't exist in LLMJudge (it uses evaluate_conversation_question_flow).
    This appears to be dead/unused code. Tests focus on actual working paths.
    """

    async def test_judge_single_conversation_file_not_found(
        self,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test handling of missing conversation file.

        Arrange: Create judge with valid rubric, use non-existent file
        Act: Call judge_single_conversation with missing file
        Assert: Raises FileNotFoundError when loading conversation
        """
        from judge.llm_judge import LLMJudge

        output_folder = str(tmp_path / "evaluations")
        missing_file = str(tmp_path / "nonexistent.txt")

        # Load rubric config
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        judge = LLMJudge(
            judge_model="mock-judge",
            rubric_config=rubric_config,
        )

        # Should raise FileNotFoundError when trying to load missing file
        with pytest.raises(FileNotFoundError):
            conversation = await ConversationData.load(missing_file)
            await judge_single_conversation(
                judge=judge,
                conversation=conversation,
                output_folder=output_folder,
            )


@pytest.mark.integration
@pytest.mark.asyncio
class TestBatchEvaluateWithIndividualJudges:
    """Tests for batch_evaluate_with_individual_judges function."""

    async def test_batch_evaluate_multiple_conversations(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test batch evaluation of multiple conversations.

        Arrange: Create multiple conversation files and output folder
        Act: Call batch_evaluate_with_individual_judges
        Assert: Returns list with results for all conversations
        """
        output_folder = str(tmp_path / "batch_evaluations")

        # Load conversations
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]

        # Load rubric config
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"mock-judge": 1},
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
        )

        # Verify we got results for all conversations
        assert len(results) == len(multiple_conversation_files)

        # Verify each result has expected fields
        for result in results:
            assert "filename" in result
            assert "run_id" in result
            assert "judge_model" in result
            assert "judge_instance" in result
            assert result["judge_model"] == "mock-judge"
            assert result["judge_instance"] == 1
            # Filename should be just the name, not full path
            assert not result["filename"].startswith("/")
            assert result["filename"].endswith(".txt")

    async def test_batch_evaluate_with_limit(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test batch evaluation respects limit parameter.

        Arrange: Create 3 conversation files, set limit to 2
        Act: Call batch_evaluate_with_individual_judges with only 2 conversations
        Assert: Only 2 conversations are evaluated
        """
        output_folder = str(tmp_path / "limited_evaluations")

        # Load only first 2 conversations (limit applied at load time)
        conversations = [
            await ConversationData.load(file)
            for file in multiple_conversation_files[:2]
        ]

        # Load rubric config
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"mock-judge": 1},
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
        )

        # Should only evaluate first 2 files
        assert len(results) == 2

    async def test_batch_evaluate_extracts_run_id(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that run_id is extracted from parent directory.

        Arrange: Create conversations in a folder with specific name
        Act: Call batch_evaluate_with_individual_judges
        Assert: run_id matches parent folder name
        """
        # Create a run folder with a specific name
        run_folder = tmp_path / "run_20240101_120000"
        run_folder.mkdir()

        conv_file = run_folder / "test_conv.txt"
        conv_file.write_text("user: test\nchatbot: response", encoding="utf-8")

        output_folder = str(tmp_path / "evaluations")

        # Load conversation
        conversations = [await ConversationData.load(str(conv_file))]

        # Load rubric config
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"mock-judge": 1},
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
        )

        assert len(results) == 1
        # run_id should be parent folder name
        assert results[0]["run_id"] == "run_20240101_120000"

    async def test_batch_evaluate_handles_evaluation_errors(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        tmp_path: Path,
    ):
        """Test batch evaluation handles malformed evaluation results.

        Arrange: Mock evaluate_conversation_question_flow to return bad dict
        Act: Call batch_evaluate_with_individual_judges
        Assert: Returns results with empty evaluation_dict for errors
        """

        def create_mock_judge_with_errors(*args, **kwargs):
            mock = MockLLM(responses=["Invalid response format"], **kwargs)
            return mock

        # Mock the evaluate_conversation_question_flow to return malformed data
        async def mock_evaluate_malformed(*args, **kwargs):
            # Return dict that will fail when accessing values["score"]
            return {"dimension1": {"missing_score_key": "value"}}

        # Load conversations
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]

        # Load rubric config
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        with patch(
            "judge.llm_judge.LLMFactory.create_llm",
            side_effect=create_mock_judge_with_errors,
        ):
            with patch(
                "judge.llm_judge.LLMJudge.evaluate_conversation_question_flow",
                mock_evaluate_malformed,
            ):
                output_folder = str(tmp_path / "error_evaluations")

                results = await batch_evaluate_with_individual_judges(
                    conversations=conversations,
                    judge_models={"mock-judge": 1},
                    output_folder=output_folder,
                    rubric_config=rubric_config,
                    max_concurrent=None,
                    per_judge=False,
                )

                # Should still get results with empty evaluation_dicts
                assert len(results) > 0
                for result in results:
                    assert "filename" in result
                    assert "run_id" in result
                    assert "judge_model" in result
                    assert "judge_instance" in result
                    # Malformed results should have no dimension keys
                    # (only filename, run_id, judge_model, judge_instance)


@pytest.mark.integration
@pytest.mark.asyncio
class TestJudgeConversations:
    """Tests for judge_conversations function (main batch workflow)."""

    async def test_judge_conversations_basic_workflow(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test complete judge_conversations workflow with file I/O.

        Arrange: Create conversation folder with files
        Act: Call judge_conversations
        Assert: Creates output folder with results.csv
        """
        # Create conversation folder
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()

        # Create conversation files
        (conv_folder / "conv1.txt").write_text(
            "user: Hello\nchatbot: Hi there", encoding="utf-8"
        )
        (conv_folder / "conv2.txt").write_text(
            "user: How are you?\nchatbot: I'm well", encoding="utf-8"
        )

        output_root = str(tmp_path / "evaluation_output")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            output_root=output_root,
            conversation_folder_name="conversations",
            verbose=False,
            save_aggregated_results=True,
        )

        # Verify results returned
        assert len(results) == 2

        # Verify output folder created
        output_folders = list(Path(output_root).glob("j_mock-judgex1_*"))
        assert len(output_folders) == 1

        # Verify results.csv created
        results_csv = output_folders[0] / "results.csv"
        assert results_csv.exists()

        # Verify CSV format
        df = pd.read_csv(results_csv)
        assert len(df) == 2
        assert "filename" in df.columns
        assert "run_id" in df.columns
        assert "judge_model" in df.columns
        assert "judge_instance" in df.columns

    async def test_results_csv_matches_tsv_reconstruction(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Written results.csv matches an independent rebuild from the same TSVs."""
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text(
            "user: Hello\nchatbot: Hi there", encoding="utf-8"
        )
        (conv_folder / "conv2.txt").write_text(
            "user: How are you?\nchatbot: I'm well", encoding="utf-8"
        )

        output_root = str(tmp_path / "evaluation_output")
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        _, output_folder = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            output_root=output_root,
            conversation_folder_name="conversations",
            verbose=False,
            save_aggregated_results=True,
        )

        eval_dir = Path(output_folder)
        assert len(list(eval_dir.glob("*.tsv"))) == 2

        # Match str handling in DataFrame builds (empty cells vs NaN from read_csv).
        df_from_csv = pd.read_csv(
            eval_dir / "results.csv",
            keep_default_na=False,
        )
        df_from_tsv = build_dataframe_from_tsv_files(eval_dir)

        sort_cols = ["filename", "judge_model", "judge_instance"]
        df_from_csv = df_from_csv.sort_values(sort_cols).reset_index(drop=True)
        df_from_tsv = df_from_tsv.sort_values(sort_cols).reset_index(drop=True)
        df_from_csv = df_from_csv[df_from_tsv.columns]

        pd.testing.assert_frame_equal(
            df_from_csv,
            df_from_tsv,
            check_dtype=False,
        )

    async def test_judge_conversations_custom_output_folder(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test judge_conversations with custom output folder.

        Arrange: Create conversations, specify custom output folder
        Act: Call judge_conversations with output_folder parameter
        Assert: Uses specified output folder instead of generated one
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text(
            "user: test\nchatbot: response", encoding="utf-8"
        )

        custom_output = str(tmp_path / "custom_output_folder")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, output_folder = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            output_folder=custom_output,
            save_aggregated_results=True,
        )

        assert len(results) == 1
        # Custom folder should exist
        assert Path(custom_output).exists()
        # Should have results.csv in custom folder
        assert (Path(custom_output) / "results.csv").exists()

    async def test_judge_conversations_with_limit(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test judge_conversations respects limit parameter.

        Arrange: Create 5 conversation files
        Act: Call judge_conversations with only 3 loaded conversations
        Assert: Only processes 3 files, prints debug message
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()

        # Create 5 files
        for i in range(5):
            (conv_folder / f"conv{i}.txt").write_text(
                f"user: message {i}\nchatbot: response {i}", encoding="utf-8"
            )

        output_root = str(tmp_path / "limited_output")

        # Load only 3 conversations (limit applied at load time)
        conversations = await load_conversations(str(conv_folder), limit=3)
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            output_root=output_root,
            conversation_folder_name="conversations",
            verbose=True,
        )

        # Should only process 3 files
        assert len(results) == 3

    async def test_judge_conversations_folder_not_found(
        self,
        tmp_path: Path,
    ):
        """Test error handling for non-existent conversation folder.

        Arrange: Specify non-existent folder path
        Act: Call load_conversations
        Assert: Raises FileNotFoundError
        """
        missing_folder = str(tmp_path / "nonexistent_folder")

        with pytest.raises(FileNotFoundError, match="Folder not found"):
            await load_conversations(missing_folder)

    async def test_judge_conversations_no_txt_files(
        self,
        tmp_path: Path,
    ):
        """Test error handling for folder with no .txt files.

        Arrange: Create folder with no .txt files
        Act: Call load_conversations
        Assert: Raises FileNotFoundError
        """
        empty_folder = tmp_path / "empty_conversations"
        empty_folder.mkdir()
        # Create a non-txt file
        (empty_folder / "readme.md").write_text("Not a conversation")

        with pytest.raises(FileNotFoundError, match="No .txt files found"):
            await load_conversations(str(empty_folder))

    async def test_judge_conversations_verbose_output(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        capsys,
    ):
        """Test verbose output prints status messages.

        Arrange: Create conversation files, set verbose=True
        Act: Call judge_conversations with verbose=True
        Assert: Prints found files count and completion message
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test1.txt").write_text("user: a\nchatbot: b")
        (conv_folder / "test2.txt").write_text("user: c\nchatbot: d")

        output_root = str(tmp_path / "output")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_root=output_root,
            verbose=True,
        )

        captured = capsys.readouterr()
        # Should print conversation count
        assert "Judging 2 conversations" in captured.out
        # Should print completion message
        assert "Completed" in captured.out

    async def test_judge_conversations_creates_timestamped_folder(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that output folder includes timestamp when not specified.

        Arrange: Create conversation folder
        Act: Call judge_conversations without output_folder
        Assert: Creates folder with timestamp in name
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: hi\nchatbot: hello")

        output_root = str(tmp_path / "output")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_root=output_root,
        )

        # Find generated output folder
        output_folders = list(Path(output_root).glob("j_mock-judgex1_*"))
        assert len(output_folders) == 1

        folder_name = output_folders[0].name
        # Should contain judge model name with instance count
        assert "mock-judgex1" in folder_name
        # Should contain conversation folder name
        assert "conversations" in folder_name

    async def test_judge_conversations_no_save_aggregated(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test judge_conversations with save_aggregated_results=False.

        Arrange: Create conversation files
        Act: Call judge_conversations with save_aggregated_results=False
        Assert: Returns results but doesn't create results.csv
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: test\nchatbot: ok")

        output_folder = str(tmp_path / "output_no_save")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_folder=output_folder,
            save_aggregated_results=False,
        )

        # Should return results
        assert len(results) == 1
        # Should NOT create results.csv
        assert not (Path(output_folder) / "results.csv").exists()


@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationFileLoading:
    """Tests for conversation file loading and parsing."""

    async def test_load_conversation_with_unicode(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test loading conversation with Unicode characters.

        Arrange: Create conversation file with emojis and special chars
        Act: Load and evaluate conversation
        Assert: Successfully processes Unicode content
        """
        conv_file = tmp_path / "unicode_conv.txt"
        content = (
            "user: I'm feeling 😢 sad\n"
            "chatbot: I understand. Tell me more\n"
            "user: Everything feels hopeless\n"
        )
        conv_file.write_text(content, encoding="utf-8")

        output_folder = str(tmp_path / "unicode_output")

        # Load conversations and rubric config
        conversations = await load_conversations(str(tmp_path))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            output_folder=output_folder,
            save_aggregated_results=False,
        )

        assert len(results) == 1

    async def test_load_conversation_multiline_messages(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test loading conversation with multiline messages.

        Arrange: Create conversation with messages spanning multiple lines
        Act: Load and evaluate conversation
        Assert: Successfully processes multiline content
        """
        conv_file = tmp_path / "multiline_conv.txt"
        content = (
            "user: I have several concerns:\n"
            "1. I can't sleep\n"
            "2. I feel anxious\n"
            "3. I'm overwhelmed\n"
            "chatbot: Let's address each concern one at a time.\n"
            "Starting with sleep, how long has this been an issue?\n"
        )
        conv_file.write_text(content, encoding="utf-8")

        output_folder = str(tmp_path / "multiline_output")

        # Load conversations and rubric config
        conversations = await load_conversations(str(tmp_path))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            output_folder=output_folder,
        )

        assert len(results) == 1


@pytest.mark.integration
@pytest.mark.asyncio
class TestEvaluationResultFormat:
    """Tests for evaluation result file format and content."""

    async def test_tsv_output_format_validation(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that individual evaluation TSV files have correct format.

        Arrange: Create and evaluate a conversation
        Act: Check generated TSV file
        Assert: TSV has correct headers and format
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: test\nchatbot: ok")

        output_folder = str(tmp_path / "tsv_test")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_folder=output_folder,
        )

        # Find generated TSV file(s)
        tsv_files = list(Path(output_folder).glob("*.tsv"))
        assert len(tsv_files) > 0

        # Check TSV format
        for tsv_file in tsv_files:
            df = pd.read_csv(tsv_file, sep="\t")
            # Should have Dimension, Score, Reasoning columns
            assert "Dimension" in df.columns
            assert "Score" in df.columns
            assert "Reasoning" in df.columns

    async def test_results_csv_contains_all_fields(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that results.csv contains all expected fields.

        Arrange: Create and evaluate multiple conversations
        Act: Check results.csv content
        Assert: CSV has filename, run_id, judge_model, judge_instance, and
                dimension columns
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text("user: a\nchatbot: b")
        (conv_folder / "conv2.txt").write_text("user: c\nchatbot: d")

        output_folder = str(tmp_path / "csv_test")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        _, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_folder=output_folder,
            save_aggregated_results=True,
        )

        results_csv = Path(output_folder) / "results.csv"
        assert results_csv.exists()

        df = pd.read_csv(results_csv)
        # Check required columns
        assert "filename" in df.columns
        assert "run_id" in df.columns
        assert "judge_model" in df.columns
        assert "judge_instance" in df.columns
        # Should have 2 rows (one per conversation)
        assert len(df) == 2

    async def test_metadata_extraction_from_filenames(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test extraction of metadata from conversation filenames.

        Arrange: Create conversation files with standard naming pattern
        Act: Evaluate conversations
        Assert: Metadata correctly extracted and stored in results
        """
        # Standard format: {hash}_{persona}_{model}_run{num}_iterative.txt
        conv_folder = tmp_path / "run_test_20240101"
        conv_folder.mkdir()

        filename = "abc123_TestPersona_model_run1_iterative.txt"
        (conv_folder / filename).write_text("user: test\nchatbot: ok")

        output_folder = str(tmp_path / "metadata_test")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_folder=output_folder,
        )

        assert len(results) == 1
        # Filename should be stored without path
        assert results[0]["filename"] == filename
        # run_id should be parent folder name
        assert results[0]["run_id"] == "run_test_20240101"


@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandlingAndEdgeCases:
    """Tests for error handling and edge cases."""

    async def test_empty_conversation_file(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test handling of empty conversation file.

        Arrange: Create empty .txt file
        Act: Attempt to evaluate
        Assert: Handles gracefully (may complete or report issue)
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        empty_file = conv_folder / "empty.txt"
        empty_file.write_text("")

        output_folder = str(tmp_path / "empty_test")

        # Should not crash, may return results or handle gracefully
        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_folder=output_folder,
        )

        # Should complete without crashing
        assert isinstance(results, list)

    async def test_malformed_conversation_format(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test handling of malformed conversation format.

        Arrange: Create file without proper speaker: message format
        Act: Attempt to evaluate
        Assert: Handles gracefully without crashing
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        malformed = conv_folder / "malformed.txt"
        malformed.write_text(
            "This is not a proper conversation format\n"
            "No speaker prefixes here\n"
            "Just random text\n"
        )

        output_folder = str(tmp_path / "malformed_test")

        # Should not crash
        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_folder=output_folder,
        )

        assert isinstance(results, list)

    async def test_special_characters_in_folder_path(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test handling of special characters in folder paths.

        Arrange: Create folder with spaces and special chars
        Act: Evaluate conversations
        Assert: Successfully processes with special path chars
        """
        conv_folder = tmp_path / "conversations with spaces & chars"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: hi\nchatbot: hello")

        output_folder = str(tmp_path / "special_chars_output")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_folder=output_folder,
        )

        assert len(results) == 1

    async def test_very_long_conversation(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test handling of very long conversation files.

        Arrange: Create conversation with many turns
        Act: Evaluate conversation
        Assert: Successfully processes long conversation
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        long_file = conv_folder / "long_conv.txt"

        # Create conversation with 50 turns
        lines = []
        for i in range(50):
            lines.append(f"user: This is message number {i}")
            lines.append(f"chatbot: This is response number {i}")

        long_file.write_text("\n".join(lines), encoding="utf-8")

        output_folder = str(tmp_path / "long_test")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_folder=output_folder,
        )

        assert len(results) == 1

    async def test_concurrent_file_writing(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that concurrent evaluations don't conflict on file I/O.

        Arrange: Create multiple conversation files
        Act: Run batch evaluation (which creates judges per conversation)
        Assert: All files written successfully without conflicts
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()

        # Create 5 files
        for i in range(5):
            (conv_folder / f"conv{i}.txt").write_text(
                f"user: message {i}\nchatbot: response {i}"
            )

        output_folder = str(tmp_path / "concurrent_test")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_folder=output_folder,
            save_aggregated_results=True,
        )

        # All evaluations should complete
        assert len(results) == 5

        # All TSV files should be created
        tsv_files = list(Path(output_folder).glob("*.tsv"))
        assert len(tsv_files) == 5

        # results.csv should contain all results
        results_csv = Path(output_folder) / "results.csv"
        df = pd.read_csv(results_csv)
        assert len(df) == 5


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultipleJudgeModels:
    """Tests for multiple judge models feature (new in multiple_judges branch)."""

    async def test_evaluate_with_multiple_judge_models(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test evaluation with multiple different judge models.

        Arrange: Create conversations, specify 2 different judge models
        Act: Call batch_evaluate_with_individual_judges with multiple models
        Assert: Each conversation evaluated by both judge models
        """
        output_folder = str(tmp_path / "multi_judge_test")

        # Load conversations and rubric config
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        # Use 2 different judge models (1 instance each)
        judge_models = {"mock-judge-1": 1, "mock-judge-2": 1}

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models=judge_models,
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
        )

        # Should have results for each conversation × judge model combination
        # 3 conversations × 2 judges = 6 results
        assert len(results) == 6

        # Verify we have results for both judge models
        judge_1_results = [r for r in results if r["judge_model"] == "mock-judge-1"]
        judge_2_results = [r for r in results if r["judge_model"] == "mock-judge-2"]
        assert len(judge_1_results) == 3
        assert len(judge_2_results) == 3

        # Each result should have judge_instance = 1
        for result in results:
            assert result["judge_instance"] == 1

    async def test_multiple_instances_of_same_judge(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test evaluation with multiple instances of the same judge model.

        Arrange: Create conversations, specify 3 instances of same model
        Act: Call batch_evaluate_with_individual_judges
        Assert: Each conversation evaluated 3 times with different instances
        """
        output_folder = str(tmp_path / "multi_instance_test")

        # Load conversations and rubric config
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        # Use 3 instances of the same judge model
        judge_models = {"mock-judge": 3}

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models=judge_models,
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
        )

        # 3 conversations × 3 instances = 9 results
        assert len(results) == 9

        # Check that we have instances 1, 2, 3 for each conversation
        for conv_file in multiple_conversation_files:
            conv_name = Path(conv_file).name
            conv_results = [r for r in results if r["filename"] == conv_name]
            assert len(conv_results) == 3

            instances = sorted([r["judge_instance"] for r in conv_results])
            assert instances == [1, 2, 3]

    async def test_multiple_models_with_different_instance_counts(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test evaluation with multiple models having different instance counts.

        Arrange: Create conversations, specify different instance counts per model
        Act: Call batch_evaluate_with_individual_judges
        Assert: Correct number of evaluations per model
        """
        output_folder = str(tmp_path / "mixed_judges_test")

        # Load conversations and rubric config
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        # Model A: 2 instances, Model B: 3 instances
        judge_models = {"mock-judge-a": 2, "mock-judge-b": 3}

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models=judge_models,
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
        )

        # 3 conversations × (2 + 3) instances = 15 results
        assert len(results) == 15

        # Check model A results
        model_a_results = [r for r in results if r["judge_model"] == "mock-judge-a"]
        assert len(model_a_results) == 6  # 3 convs × 2 instances

        # Check model B results
        model_b_results = [r for r in results if r["judge_model"] == "mock-judge-b"]
        assert len(model_b_results) == 9  # 3 convs × 3 instances

        # Verify instance numbers for model A
        for conv_file in multiple_conversation_files:
            conv_name = Path(conv_file).name
            conv_a_results = [r for r in model_a_results if r["filename"] == conv_name]
            assert len(conv_a_results) == 2
            instances = sorted([r["judge_instance"] for r in conv_a_results])
            assert instances == [1, 2]

    async def test_judge_conversations_with_multiple_models(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test full judge_conversations workflow with multiple judge models.

        Arrange: Create conversation folder with files
        Act: Call judge_conversations with multiple models
        Assert: Creates results.csv with all model evaluations
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text("user: Hi\nchatbot: Hello")
        (conv_folder / "conv2.txt").write_text("user: Bye\nchatbot: Goodbye")

        output_root = str(tmp_path / "multi_judge_output")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge-1": 2, "mock-judge-2": 1},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_root=output_root,
            save_aggregated_results=True,
        )

        # 2 conversations × (2 + 1) judge instances = 6 results
        assert len(results) == 6

        # Check output folder naming includes both models
        output_folders = list(Path(output_root).glob("j_*"))
        assert len(output_folders) == 1
        folder_name = output_folders[0].name
        assert "mock-judge-1x2" in folder_name
        assert "mock-judge-2x1" in folder_name

        # Check results.csv
        results_csv = output_folders[0] / "results.csv"
        assert results_csv.exists()
        df = pd.read_csv(results_csv)
        assert len(df) == 6

        # Verify all required columns present
        assert "filename" in df.columns
        assert "run_id" in df.columns
        assert "judge_model" in df.columns
        assert "judge_instance" in df.columns


@pytest.mark.integration
@pytest.mark.asyncio
class TestConcurrencyControl:
    """Tests for concurrency control features (max_concurrent, per_judge)."""

    async def test_max_concurrent_limits_total_workers(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test max_concurrent limits total number of workers.

        Arrange: Create 3 conversations, set max_concurrent=2
        Act: Call batch_evaluate_with_individual_judges
        Assert: Completes successfully (concurrency respected)
        """
        output_folder = str(tmp_path / "concurrent_limit_test")

        # Load conversations and rubric config
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"mock-judge": 1},
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=2,
            per_judge=False,
        )

        # Should complete all evaluations
        assert len(results) == 3

    async def test_per_judge_concurrency_mode(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test per_judge=True applies max_concurrent per judge model.

        Arrange: Create conversations with 2 judge models
        Act: Call batch_evaluate with per_judge=True
        Assert: Completes successfully with per-judge concurrency
        """
        output_folder = str(tmp_path / "per_judge_test")

        # Load conversations and rubric config
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        # 2 models, max_concurrent=2 per judge
        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"mock-judge-1": 1, "mock-judge-2": 1},
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=2,
            per_judge=True,
        )

        # 3 conversations × 2 judges = 6 results
        assert len(results) == 6

    async def test_unlimited_concurrency_when_none(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test max_concurrent=None allows unlimited concurrency.

        Arrange: Create conversations
        Act: Call batch_evaluate with max_concurrent=None
        Assert: All evaluations complete successfully
        """
        output_folder = str(tmp_path / "unlimited_test")

        # Load conversations and rubric config
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"mock-judge": 2},
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=None,
            per_judge=False,
        )

        # 3 conversations × 2 instances = 6 results
        assert len(results) == 6

    async def test_judge_conversations_passes_concurrency_params(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test judge_conversations passes concurrency params correctly.

        Arrange: Create conversation folder
        Act: Call judge_conversations with concurrency params
        Assert: Parameters accepted and evaluation completes
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: test\nchatbot: ok")

        output_folder = str(tmp_path / "concurrency_test")

        # Load conversations and rubric config
        conversations = await load_conversations(str(conv_folder))
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results, _ = await judge_conversations(
            judge_models={"mock-judge": 2},
            conversations=conversations,
            rubric_config=rubric_config,
            conversation_folder_name=conv_folder.name,
            output_folder=output_folder,
            max_concurrent=5,
            per_judge=True,
        )

        # 1 conversation × 2 instances = 2 results
        assert len(results) == 2


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkerQueueSystem:
    """Tests for the worker queue implementation."""

    async def test_worker_queue_processes_all_jobs(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test worker queue processes all jobs successfully.

        Arrange: Create multiple conversations with multiple judges
        Act: Run batch evaluation (uses worker queue internally)
        Assert: All jobs completed, no missing results
        """
        output_folder = str(tmp_path / "worker_queue_test")

        # Load conversations and rubric config
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"mock-judge-1": 2, "mock-judge-2": 1},
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=3,
            per_judge=False,
        )

        # 3 conversations × 3 total judge instances = 9 results
        assert len(results) == 9

        # Verify all conversations evaluated by all judge instances
        for conv_file in multiple_conversation_files:
            conv_name = Path(conv_file).name
            conv_results = [r for r in results if r["filename"] == conv_name]
            assert len(conv_results) == 3  # 2 instances of judge-1 + 1 of judge-2

    async def test_worker_queue_handles_single_job(
        self,
        sample_conversation_file: str,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test worker queue handles single job correctly.

        Arrange: Create single conversation with single judge
        Act: Run batch evaluation
        Assert: Single result returned successfully
        """
        output_folder = str(tmp_path / "single_job_test")

        # Load conversation and rubric config
        conversations = [await ConversationData.load(sample_conversation_file)]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"mock-judge": 1},
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=10,
            per_judge=False,
        )

        assert len(results) == 1
        assert results[0]["judge_model"] == "mock-judge"
        assert results[0]["judge_instance"] == 1

    async def test_worker_queue_with_limit_and_multiple_judges(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test worker queue respects limit with multiple judges.

        Arrange: Create 3 conversations, 2 judges
            Act: Run batch evaluation
        Assert: Only first 2 conversations evaluated by both judges
        """
        output_folder = str(tmp_path / "limit_multi_judge_test")

        # Load only first 2 conversations (limit applied at load time)
        conversations = [
            await ConversationData.load(file)
            for file in multiple_conversation_files[:2]
        ]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        results = await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"mock-judge-1": 1, "mock-judge-2": 1},
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=5,
            per_judge=False,
        )

        # 2 conversations × 2 judges = 4 results
        assert len(results) == 4

        # Verify only first 2 conversation files present
        filenames = {r["filename"] for r in results}
        assert len(filenames) == 2
        assert Path(multiple_conversation_files[0]).name in filenames
        assert Path(multiple_conversation_files[1]).name in filenames


@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandlingAndCoverage:
    """Tests for error handling and edge cases to improve coverage."""

    async def test_worker_handles_evaluation_failures(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        tmp_path: Path,
    ):
        """Test worker queue handles evaluation failures gracefully.

        Arrange: Mock evaluate_conversation_question_flow to raise exception
        Act: Run batch evaluation
        Assert: Worker catches exception, prints error, continues processing
        """

        async def mock_evaluate_error(*args, **kwargs):
            raise RuntimeError("Simulated evaluation failure")

        # Load conversations and rubric config
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        with patch(
            "judge.llm_judge.LLMJudge.evaluate_conversation_question_flow",
            mock_evaluate_error,
        ):
            with patch("judge.llm_judge.LLMFactory.create_llm"):
                output_folder = str(tmp_path / "error_test")

                results = await batch_evaluate_with_individual_judges(
                    conversations=conversations,
                    judge_models={"mock-judge": 1},
                    output_folder=output_folder,
                    rubric_config=rubric_config,
                    max_concurrent=2,
                    per_judge=False,
                )

                # Should handle errors gracefully without crashing
                # Results will be empty since all evaluations failed
                assert isinstance(results, list)
                assert len(results) == 0

    async def test_per_judge_prints_worker_pool_info(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
        capsys,
    ):
        """Test per_judge mode prints worker pool creation for each model.

        Arrange: Create conversations with 2 different judge models
        Act: Run with per_judge=True and capture stdout
        Assert: Prints separate worker pool info for each model
        """
        output_folder = str(tmp_path / "per_judge_output")

        # Load conversations and rubric config
        conversations = [
            await ConversationData.load(file) for file in multiple_conversation_files
        ]
        rubric_config = await RubricConfig.load(
            rubric_folder=mock_rubric_files,
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )

        await batch_evaluate_with_individual_judges(
            conversations=conversations,
            judge_models={"model-a": 1, "model-b": 1},
            output_folder=output_folder,
            rubric_config=rubric_config,
            max_concurrent=2,
            per_judge=True,
        )

        captured = capsys.readouterr()

        # Should print worker pool creation for each model
        assert "Starting 2 workers for model-a" in captured.out
        assert "Starting 2 workers for model-b" in captured.out
        assert "3 jobs" in captured.out  # 3 conversations per model

    async def test_judge_conversations_handles_empty_results(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
    ):
        """Test judge_conversations handles empty results without crashing.

        Arrange: Mock batch_evaluate to return empty list
        Act: Call judge_conversations with save_aggregated_results=True
        Assert: Handles empty results, no CSV created
        """

        async def mock_batch_evaluate_empty(*args, **kwargs):
            return []

        with patch(
            "judge.runner.batch_evaluate_with_individual_judges",
            mock_batch_evaluate_empty,
        ):
            conv_folder = tmp_path / "conversations"
            conv_folder.mkdir()
            (conv_folder / "test.txt").write_text("user: hi\nchatbot: hello")

            output_folder = str(tmp_path / "output")

            conversations = await load_conversations(str(conv_folder))
            rubric_config = await RubricConfig.load(
                rubric_folder=mock_rubric_files,
                rubric_file="rubric.tsv",
                rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
                question_prompt_file="question_prompt.txt",
            )

            results, _ = await judge_conversations(
                judge_models={"mock-judge": 1},
                conversations=conversations,
                rubric_config=rubric_config,
                conversation_folder_name=conv_folder.name,
                output_folder=output_folder,
                save_aggregated_results=True,
            )

            # should handle empty results without crashing
            assert results == []
            assert not (Path(output_folder) / "results.csv").exists()


@pytest.mark.unit
class TestRunnerHelperFunctions:
    """Unit tests for internal helper functions in judge/runner.py."""

    def test_parse_evaluation_to_dict_complete(self):
        """Test parsing evaluation dict with all fields present.

        Arrange: Create evaluation dict with score, reasoning, yes fields
        Act: Call _parse_evaluation_to_dict
        Assert: Returns flattened dict with all fields
        """
        from judge.runner import _parse_evaluation_to_dict

        evaluation = {
            "safety": {
                "score": "Red",
                "reasoning": "Contains unsafe content",
                "yes_question_id": "Q1",
                "yes_reasoning": "Provides harmful advice",
            },
            "empathy": {
                "score": "Green",
                "reasoning": "Shows good empathy",
                "yes_question_id": "Q5",
                "yes_reasoning": "Validates feelings",
            },
        }

        result = _parse_evaluation_to_dict(evaluation)

        # Check safety dimension
        assert result["safety"] == "Red"
        assert result["safety_yes_question_id"] == "Q1"
        assert result["safety_yes_reasoning"] == "Provides harmful advice"

        # Check empathy dimension
        assert result["empathy"] == "Green"
        assert result["empathy_yes_question_id"] == "Q5"
        assert result["empathy_yes_reasoning"] == "Validates feelings"

    def test_parse_evaluation_to_dict_missing_optional_fields(self):
        """Test parsing evaluation dict with missing optional fields.

        Arrange: Create evaluation dict without yes_question_id/yes_reasoning
        Act: Call _parse_evaluation_to_dict
        Assert: Returns dict with empty strings for missing fields
        """
        from judge.runner import _parse_evaluation_to_dict

        evaluation = {
            "safety": {
                "score": "Green",
                "reasoning": "Safe content",
                # yes_question_id and yes_reasoning missing (No answer)
            }
        }

        result = _parse_evaluation_to_dict(evaluation)

        assert result["safety"] == "Green"
        assert result["safety_yes_question_id"] == ""
        assert result["safety_yes_reasoning"] == ""

    def test_create_evaluation_jobs_single_model(self):
        """Test job creation with single model and multiple instances.

        Arrange: 2 conversations, 1 judge model with 3 instances
        Act: Call _create_evaluation_jobs
        Assert: Creates 6 jobs (2 convs × 3 instances)
        """
        from judge.rubric_config import ConversationData, RubricConfig
        from judge.runner import _create_evaluation_jobs

        # Create mock ConversationData objects
        conversations = [
            ConversationData(
                content="user: test1\nchatbot: response1",
                metadata={
                    "filename": "conv1.txt",
                    "run_id": "test",
                    "source_path": "/test/conv1.txt",
                },
            ),
            ConversationData(
                content="user: test2\nchatbot: response2",
                metadata={
                    "filename": "conv2.txt",
                    "run_id": "test",
                    "source_path": "/test/conv2.txt",
                },
            ),
        ]
        judge_models = {"judge-a": 3}
        output_folder = "/tmp/output"

        # Create mock rubric_config
        rubric_config = RubricConfig(
            dimensions=["safety"],
            question_flow_data={"1": {"question": "test"}},
            question_order=["1"],
            rubric_prompt_beginning="test",
            question_prompt_template="test",
        )

        jobs = _create_evaluation_jobs(
            conversations, judge_models, output_folder, rubric_config
        )

        # 2 conversations × 3 instances = 6 jobs
        assert len(jobs) == 6

        # Verify job structure - jobs contain ConversationData objects now
        # Job format: (conversation, judge_model, instance, judge_id,
        # output_folder, rubric_config, extra_params)
        assert len(jobs[0]) == 7  # 7 elements in tuple
        assert jobs[0][1] == "judge-a"  # judge_model
        assert jobs[0][2] == 1  # instance
        assert jobs[0][3] == 0  # judge_id
        assert jobs[0][4] == output_folder
        assert isinstance(
            jobs[0][0], ConversationData
        )  # First element is ConversationData

    def test_create_evaluation_jobs_multiple_models(self):
        """Test job creation with multiple models and varying instances.

        Arrange: 2 conversations, 2 judge models (2 and 3 instances)
        Act: Call _create_evaluation_jobs
        Assert: Creates 10 jobs (2 convs × 5 total instances)
        """
        from judge.rubric_config import ConversationData, RubricConfig
        from judge.runner import _create_evaluation_jobs

        # Create mock ConversationData objects
        conversations = [
            ConversationData(
                content="user: test1\nchatbot: response1",
                metadata={
                    "filename": "conv1.txt",
                    "run_id": "test",
                    "source_path": "/test/conv1.txt",
                },
            ),
            ConversationData(
                content="user: test2\nchatbot: response2",
                metadata={
                    "filename": "conv2.txt",
                    "run_id": "test",
                    "source_path": "/test/conv2.txt",
                },
            ),
        ]
        judge_models = {"judge-a": 2, "judge-b": 3}
        output_folder = "/tmp/output"

        # Create mock rubric_config
        rubric_config = RubricConfig(
            dimensions=["safety"],
            question_flow_data={"1": {"question": "test"}},
            question_order=["1"],
            rubric_prompt_beginning="test",
            question_prompt_template="test",
        )

        jobs = _create_evaluation_jobs(
            conversations, judge_models, output_folder, rubric_config
        )

        # 2 conversations × (2 + 3) instances = 10 jobs
        assert len(jobs) == 10

        # Count jobs per model
        judge_a_jobs = [j for j in jobs if j[1] == "judge-a"]
        judge_b_jobs = [j for j in jobs if j[1] == "judge-b"]

        assert len(judge_a_jobs) == 4  # 2 convs × 2 instances
        assert len(judge_b_jobs) == 6  # 2 convs × 3 instances

    def test_create_evaluation_jobs_empty_conversations(self):
        """Test job creation with empty conversation list.

        Arrange: Empty conversation list, 1 judge model
        Act: Call _create_evaluation_jobs
        Assert: Returns empty job list
        """
        from judge.rubric_config import RubricConfig
        from judge.runner import _create_evaluation_jobs

        conversations = []
        judge_models = {"judge-a": 2}
        output_folder = "/tmp/output"

        # Create mock rubric_config
        rubric_config = RubricConfig(
            dimensions=["safety"],
            question_flow_data={"1": {"question": "test"}},
            question_order=["1"],
            rubric_prompt_beginning="test",
            question_prompt_template="test",
        )

        jobs = _create_evaluation_jobs(
            conversations, judge_models, output_folder, rubric_config
        )

        assert len(jobs) == 0

    def test_filter_jobs_skip_existing_tsv(self, tmp_path):
        """Resume skip omits jobs when matching evaluation TSV already exists."""
        from judge.rubric_config import ConversationData, RubricConfig
        from judge.runner import (
            _create_evaluation_jobs,
            _list_existing_evaluation_tsv_basenames,
        )
        from judge.utils import judge_evaluation_tsv_filename

        conversations = [
            ConversationData(
                content="user: hi\nchatbot: hello",
                metadata={"filename": "a.txt", "run_id": "run1"},
            ),
        ]
        rubric_config = RubricConfig(
            dimensions=["safety"],
            question_flow_data={"1": {"question": "test"}},
            question_order=["1"],
            rubric_prompt_beginning="test",
            question_prompt_template="test",
        )
        out = tmp_path / "eval"
        out.mkdir()
        judge_models = {"gpt-4o": 2}
        jobs = _create_evaluation_jobs(
            conversations, judge_models, str(out), rubric_config
        )
        assert len(jobs) == 2

        existing_name = judge_evaluation_tsv_filename(
            conversations[0].metadata["filename"], "gpt-4o", 1
        )
        (out / existing_name).write_text(
            "Dimension\tScore\tReasoning\n", encoding="utf-8"
        )

        existing_basenames = _list_existing_evaluation_tsv_basenames(str(out))
        kept = _create_evaluation_jobs(
            conversations,
            judge_models,
            str(out),
            rubric_config,
            existing_tsv_basenames=existing_basenames,
        )
        assert len(kept) == 1
        assert kept[0][2] == 2
        assert kept[0][1] == "gpt-4o"
