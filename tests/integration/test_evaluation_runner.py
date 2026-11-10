"""Integration tests for judging: judge/run.py (`run_judging`), which runs one
`LLMJudge` job per conversation, judge model, and instance on the job loop.

Tests the full workflow of evaluating conversations including:
- Conversation file loading and parsing
- Batch evaluation with file I/O
- TSV result file creation and format validation
- Error handling for missing/malformed files
- Concurrency caps, overall and per judge model
- Metadata extraction from filenames
- events.jsonl and resuming a run with `into`
"""

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pandas as pd
import pytest

from tests.mocks.mock_llm import MockLLM
from tests.mocks.run_record import minimal_run_record
from vera_mh.judge.answers import (
    answer_column,
    answers_tsv_path,
    conversation_answers_dir,
    explanation_column,
)
from vera_mh.judge.llm_judge import LLMJudge
from vera_mh.judge.rubric_config import (
    RubricConfig,
    load_conversations,
)
from vera_mh.judge.run import run_judging
from vera_mh.judge.utils import judge_evaluation_tsv_filename
from vera_mh.utils.config_schema import JudgingConfig, ModelSpec, RubricFiles
from vera_mh.utils.judge_results import build_dataframe_from_tsv_files


def judging_config(
    conversations: str,
    rubric_folder: str,
    output: str = "unused-with-into",
    judge_models: Optional[Dict[str, int]] = None,
    max_concurrent: Optional[int] = None,
    per_judge: bool = False,
) -> JudgingConfig:
    """A judging config for the conversations in `conversations`, judged with
    the rubric files in `rubric_folder` (see the `mock_rubric_files` fixture).

    `output` is where a new run folder goes; a call that passes `into` writes
    there instead and never reads it.
    """
    rubric = Path(rubric_folder).resolve()
    return JudgingConfig(
        models=[
            ModelSpec(name=name, repeats=count, extra_params={})
            for name, count in (judge_models or {"mock-judge": 1}).items()
        ],
        rubrics=[
            RubricFiles(
                rubric_file=str(rubric / "rubric.tsv"),
                rubric_prompt_beginning_file=str(
                    rubric / "rubric_prompt_beginning.txt"
                ),
                question_prompt_file=str(rubric / "question_prompt.txt"),
            )
        ],
        max_concurrent=max_concurrent,
        per_judge=per_judge,
        conversations=[conversations],
        output=output,
    )


async def evaluate_all(
    conversations_folder: str,
    judge_models: Dict[str, int],
    output_root: str,
    rubric_folder: str,
    max_concurrent: Optional[int] = None,
    per_judge: bool = False,
    sample: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Judge the conversations in `conversations_folder` with `run_judging`, in a
    new run folder under `output_root`, and return the rows of its results.csv
    (none when no evaluation succeeded, as then results.csv isn't written)."""
    output_folder = await run_judging(
        judging_config(
            conversations_folder,
            rubric_folder,
            output=output_root,
            judge_models=judge_models,
            max_concurrent=max_concurrent,
            per_judge=per_judge,
        ),
        sample=sample,
        run_record=minimal_run_record(),
    )
    return read_results(output_folder)


def read_results(output_folder: str) -> List[Dict[str, Any]]:
    """The rows of the run's results.csv, or none if it wasn't written."""
    results_csv = Path(output_folder) / "results.csv"
    if not results_csv.exists():
        return []
    return pd.read_csv(results_csv, keep_default_na=False).to_dict("records")


def read_events(output_folder: str) -> List[Dict[str, Any]]:
    """The events the run wrote to `events.jsonl` in its output folder."""
    lines = (Path(output_folder) / "events.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in lines.splitlines()]


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

    with patch(
        "vera_mh.judge.llm_judge.LLMFactory.create_llm", side_effect=create_mock_judge
    ):
        yield


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkerPoolEvaluation:
    """Tests for judging a folder of conversations through `run_judging`
    (`evaluate_all`)."""

    async def test_batch_evaluate_multiple_conversations(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test batch evaluation of multiple conversations.

        Arrange: Create multiple conversation files and output folder
        Act: Call evaluate_all
        Assert: Returns list with results for all conversations
        """
        output_folder = str(tmp_path / "batch_evaluations")

        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models={"mock-judge": 1},
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
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
        Act: Call evaluate_all with sample=2
        Assert: Only 2 conversations are evaluated
        """
        output_folder = str(tmp_path / "limited_evaluations")

        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models={"mock-judge": 1},
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
            max_concurrent=None,
            per_judge=False,
            sample=2,
        )

        # Should only evaluate 2 of the 3 files
        assert len(results) == 2
        assert len({r["filename"] for r in results}) == 2

    async def test_batch_evaluate_extracts_run_id(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that run_id is extracted from parent directory.

        Arrange: Create conversations in a folder with specific name
        Act: Call evaluate_all
        Assert: run_id matches parent folder name
        """
        # Create a run folder with a specific name
        run_folder = tmp_path / "run_20240101_120000"
        run_folder.mkdir()

        conv_file = run_folder / "test_conv.txt"
        conv_file.write_text("user: test\nchatbot: response", encoding="utf-8")

        output_folder = str(tmp_path / "evaluations")

        results = await evaluate_all(
            conversations_folder=str(run_folder),
            judge_models={"mock-judge": 1},
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
            max_concurrent=None,
            per_judge=False,
        )

        assert len(results) == 1
        # run_id should be the conversation folder's name
        assert results[0]["run_id"] == "run_20240101_120000"


@pytest.mark.integration
@pytest.mark.asyncio
class TestRunJudging:
    """Tests for run_judging (main batch workflow)."""

    async def test_run_judging_basic_workflow(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test complete run_judging workflow with file I/O.

        Arrange: Create conversation folder with files
        Act: Call run_judging
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

        folder = await run_judging(
            judging_config(
                str(conv_folder),
                mock_rubric_files,
                output=output_root,
                judge_models={"mock-judge": 1},
            ),
            run_record=minimal_run_record(),
        )
        results = read_results(folder)

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

    async def test_run_judging_writes_per_question_answers(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """The run writes answers/answers.tsv beside results.csv.

        Arrange: two conversations
        Act: run_judging
        Assert: one wide answers row per conversation, rubric-ordered question
        columns, and nothing stray in the evaluation folder's own *.tsv glob
        """
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

        await run_judging(
            judging_config(
                str(conv_folder),
                mock_rubric_files,
                output=output_root,
                judge_models={"mock-judge": 1},
            ),
            run_record=minimal_run_record(),
        )

        output_folder = next(Path(output_root).glob("j_mock-judgex1_*"))

        answers_tsv = answers_tsv_path(output_folder)
        assert answers_tsv.exists()

        df = pd.read_csv(answers_tsv, sep="\t", keep_default_na=False)
        assert len(df) == len(conversations)
        for question_id in rubric_config.question_order:
            assert answer_column(question_id) in df.columns
            assert explanation_column(question_id) in df.columns

        # Every conversation was asked the first rubric question.
        first_question = answer_column(rubric_config.question_order[0])
        assert all(answer != "" for answer in df[first_question])

        # A per-conversation file exists for each evaluation TSV.
        eval_tsvs = {p.name for p in output_folder.glob("*.tsv")}
        per_conversation = {
            p.name for p in conversation_answers_dir(output_folder).glob("*.tsv")
        }
        assert per_conversation == eval_tsvs
        assert "answers.tsv" not in eval_tsvs

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

        output_folder = await run_judging(
            judging_config(
                str(conv_folder),
                mock_rubric_files,
                output=output_root,
                judge_models={"mock-judge": 1},
            ),
            run_record=minimal_run_record(),
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

    async def test_run_judging_custom_output_folder(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test run_judging with custom output folder.

        Arrange: Create conversations, specify custom output folder
        Act: Call run_judging with `into`
        Assert: Uses specified output folder instead of generated one
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text(
            "user: test\nchatbot: response", encoding="utf-8"
        )

        custom_output = str(tmp_path / "custom_output_folder")

        output_folder = await run_judging(
            judging_config(
                str(conv_folder), mock_rubric_files, judge_models={"mock-judge": 1}
            ),
            into=custom_output,
        )

        assert output_folder == custom_output
        assert len(read_results(output_folder)) == 1
        # Custom folder should exist
        assert Path(custom_output).exists()
        # Should have results.csv in custom folder
        assert (Path(custom_output) / "results.csv").exists()

    async def test_run_judging_with_limit(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test run_judging respects `sample`.

        Arrange: Create 5 conversation files
        Act: Call run_judging with sample=3
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

        folder = await run_judging(
            judging_config(
                str(conv_folder),
                mock_rubric_files,
                output=output_root,
                judge_models={"mock-judge": 1},
            ),
            sample=3,
            run_record=minimal_run_record(),
        )
        results = read_results(folder)

        # Should only process 3 files
        assert len(results) == 3

    async def test_run_judging_folder_not_found(
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

    async def test_run_judging_no_txt_files(
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

    async def test_run_judging_prints_status(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        capsys,
    ):
        """Test that judging prints status messages.

        Arrange: Create conversation files
        Act: Call run_judging
        Assert: Prints found files count and completion message
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test1.txt").write_text("user: a\nchatbot: b")
        (conv_folder / "test2.txt").write_text("user: c\nchatbot: d")

        output_root = str(tmp_path / "output")

        await run_judging(
            judging_config(
                str(conv_folder),
                mock_rubric_files,
                output=output_root,
                judge_models={"mock-judge": 1},
            ),
            run_record=minimal_run_record(),
        )

        captured = capsys.readouterr()
        # Should print conversation count
        assert "Loaded 2 conversations" in captured.out
        # Should print the loop's summary and where the evaluations went
        assert "judge finished: 2 done, 0 failed, 0 skipped" in captured.out
        assert "Evaluations in" in captured.out
        assert "failed; rerun" not in captured.out

    async def test_run_judging_creates_timestamped_folder(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test that output folder includes timestamp when not specified.

        Arrange: Create conversation folder
        Act: Call run_judging without output_folder
        Assert: Creates folder with timestamp in name
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: hi\nchatbot: hello")

        output_root = str(tmp_path / "output")

        await run_judging(
            judging_config(
                str(conv_folder),
                mock_rubric_files,
                output=output_root,
                judge_models={"mock-judge": 1},
            ),
            run_record=minimal_run_record(),
        )

        # Find generated output folder
        output_folders = list(Path(output_root).glob("j_mock-judgex1_*"))
        assert len(output_folders) == 1

        folder_name = output_folders[0].name
        # Should contain judge model name with instance count
        assert "mock-judgex1" in folder_name
        # Should contain conversation folder name
        assert "conversations" in folder_name


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

        folder = await run_judging(
            judging_config(
                str(tmp_path), mock_rubric_files, judge_models={"mock-judge": 1}
            ),
            into=output_folder,
        )
        results = read_results(folder)

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

        folder = await run_judging(
            judging_config(
                str(tmp_path), mock_rubric_files, judge_models={"mock-judge": 1}
            ),
            into=output_folder,
        )
        results = read_results(folder)

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

        await run_judging(
            judging_config(
                str(conv_folder), mock_rubric_files, judge_models={"mock-judge": 1}
            ),
            into=output_folder,
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

        await run_judging(
            judging_config(
                str(conv_folder), mock_rubric_files, judge_models={"mock-judge": 1}
            ),
            into=output_folder,
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

        output_root = str(tmp_path / "metadata_test")

        folder = await run_judging(
            judging_config(
                str(conv_folder),
                mock_rubric_files,
                output=output_root,
                judge_models={"mock-judge": 1},
            ),
            run_record=minimal_run_record(),
        )
        results = read_results(folder)

        assert len(results) == 1
        # Filename should be stored without path
        assert results[0]["filename"] == filename
        # run_id comes from the run folder, which is named after the
        # conversation folder
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

        folder = await run_judging(
            judging_config(
                str(conv_folder), mock_rubric_files, judge_models={"mock-judge": 1}
            ),
            into=output_folder,
        )
        results = read_results(folder)

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

        folder = await run_judging(
            judging_config(
                str(conv_folder), mock_rubric_files, judge_models={"mock-judge": 1}
            ),
            into=output_folder,
        )
        results = read_results(folder)

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

        folder = await run_judging(
            judging_config(
                str(conv_folder), mock_rubric_files, judge_models={"mock-judge": 1}
            ),
            into=output_folder,
        )
        results = read_results(folder)

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

        folder = await run_judging(
            judging_config(
                str(conv_folder), mock_rubric_files, judge_models={"mock-judge": 1}
            ),
            into=output_folder,
        )
        results = read_results(folder)

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

        folder = await run_judging(
            judging_config(
                str(conv_folder), mock_rubric_files, judge_models={"mock-judge": 1}
            ),
            into=output_folder,
        )
        results = read_results(folder)

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
        Act: Call evaluate_all with multiple models
        Assert: Each conversation evaluated by both judge models
        """
        output_folder = str(tmp_path / "multi_judge_test")

        # Use 2 different judge models (1 instance each)
        judge_models = {"mock-judge-1": 1, "mock-judge-2": 1}

        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models=judge_models,
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
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
        Act: Call evaluate_all
        Assert: Each conversation evaluated 3 times with different instances
        """
        output_folder = str(tmp_path / "multi_instance_test")

        # Use 3 instances of the same judge model
        judge_models = {"mock-judge": 3}

        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models=judge_models,
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
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
        Act: Call evaluate_all
        Assert: Correct number of evaluations per model
        """
        output_folder = str(tmp_path / "mixed_judges_test")

        # Model A: 2 instances, Model B: 3 instances
        judge_models = {"mock-judge-a": 2, "mock-judge-b": 3}

        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models=judge_models,
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
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

    async def test_run_judging_with_multiple_models(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test full run_judging workflow with multiple judge models.

        Arrange: Create conversation folder with files
        Act: Call run_judging with multiple models
        Assert: Creates results.csv with all model evaluations
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text("user: Hi\nchatbot: Hello")
        (conv_folder / "conv2.txt").write_text("user: Bye\nchatbot: Goodbye")

        output_root = str(tmp_path / "multi_judge_output")

        folder = await run_judging(
            judging_config(
                str(conv_folder),
                mock_rubric_files,
                output=output_root,
                judge_models={"mock-judge-1": 2, "mock-judge-2": 1},
            ),
            run_record=minimal_run_record(),
        )
        results = read_results(folder)

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
        """Test max_concurrent caps the number of evaluations running at once.

        Arrange: Create 3 conversations, set max_concurrent=2
        Act: Call evaluate_all
        Assert: Completes successfully (concurrency respected)
        """
        output_folder = str(tmp_path / "concurrent_limit_test")

        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models={"mock-judge": 1},
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
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
        Act: Call evaluate_all with per_judge=True
        Assert: Completes successfully with per-judge concurrency
        """
        output_folder = str(tmp_path / "per_judge_test")

        # 2 models, max_concurrent=2 per judge
        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models={"mock-judge-1": 1, "mock-judge-2": 1},
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
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
        Act: Call evaluate_all with max_concurrent=None
        Assert: All evaluations complete successfully
        """
        output_folder = str(tmp_path / "unlimited_test")

        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models={"mock-judge": 2},
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
            max_concurrent=None,
            per_judge=False,
        )

        # 3 conversations × 2 instances = 6 results
        assert len(results) == 6

    async def test_run_judging_passes_concurrency_params(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Test run_judging passes concurrency params correctly.

        Arrange: Create conversation folder
        Act: Call run_judging with concurrency params
        Assert: Parameters accepted and evaluation completes
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "test.txt").write_text("user: test\nchatbot: ok")

        output_folder = str(tmp_path / "concurrency_test")

        folder = await run_judging(
            judging_config(
                str(conv_folder),
                mock_rubric_files,
                judge_models={"mock-judge": 2},
                max_concurrent=5,
                per_judge=True,
            ),
            into=output_folder,
        )
        results = read_results(folder)

        # 1 conversation × 2 instances = 2 results
        assert len(results) == 2


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkerQueueSystem:
    """Tests for running every evaluation job on the job loop."""

    async def test_worker_queue_processes_all_jobs(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
        tmp_path: Path,
    ):
        """Test the job loop runs every evaluation successfully.

        Arrange: Create multiple conversations with multiple judges
        Act: Run batch evaluation (LLMJudge jobs on the job loop)
        Assert: All jobs completed, no missing results
        """
        output_folder = str(tmp_path / "worker_queue_test")

        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models={"mock-judge-1": 2, "mock-judge-2": 1},
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
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
        """Test the job loop handles a single evaluation correctly.

        Arrange: Create single conversation with single judge
        Act: Run batch evaluation
        Assert: Single result returned successfully
        """
        output_folder = str(tmp_path / "single_job_test")

        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models={"mock-judge": 1},
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
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
        """Test the job loop respects `sample` with multiple judges.

        Arrange: Create 3 conversations, 2 judges
        Act: Run batch evaluation with sample=2
        Assert: Only 2 conversations evaluated, by both judges
        """
        output_folder = str(tmp_path / "limit_multi_judge_test")

        results = await evaluate_all(
            conversations_folder=str(tmp_path),
            judge_models={"mock-judge-1": 1, "mock-judge-2": 1},
            output_root=output_folder,
            rubric_folder=mock_rubric_files,
            max_concurrent=5,
            per_judge=False,
            sample=2,
        )

        # 2 conversations × 2 judges = 4 results
        assert len(results) == 4

        # Verify only 2 of the conversation files present, each judged by both
        filenames = {r["filename"] for r in results}
        assert len(filenames) == 2
        assert filenames <= {Path(f).name for f in multiple_conversation_files}
        for filename in filenames:
            judges = {r["judge_model"] for r in results if r["filename"] == filename}
            assert judges == {"mock-judge-1", "mock-judge-2"}


@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandlingAndCoverage:
    """Tests for error handling and edge cases to improve coverage."""

    async def test_worker_handles_evaluation_failures(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        tmp_path: Path,
        capsys,
    ):
        """Test the job loop handles evaluation failures gracefully.

        Arrange: Mock evaluate_conversation_question_flow to raise exception
        Act: Run judging
        Assert: Every evaluation fails without crashing the run, no TSV or
            results.csv is written, and the rerun hint is printed
        """

        async def mock_evaluate_error(*args, **kwargs):
            raise RuntimeError("Simulated evaluation failure")

        with patch(
            "vera_mh.judge.llm_judge.LLMJudge.evaluate_conversation_question_flow",
            mock_evaluate_error,
        ):
            with patch("vera_mh.judge.llm_judge.LLMFactory.create_llm"):
                output_folder = str(tmp_path / "error_test")

                folder = await run_judging(
                    judging_config(
                        str(tmp_path),
                        mock_rubric_files,
                        judge_models={"mock-judge": 1},
                        max_concurrent=2,
                    ),
                    into=output_folder,
                )

        assert not list(Path(folder).glob("*.tsv"))
        assert not (Path(folder) / "results.csv").exists()
        assert read_results(folder) == []

        captured = capsys.readouterr()
        assert "judge finished: 0 done, 3 failed, 0 skipped" in captured.out
        assert (
            f"3 evaluation(s) failed; rerun with `--into {output_folder}`"
            in captured.out
        )


def recording_run(running: Counter, peaks: Counter):
    """A stand-in for `LLMJudge.run` that records how many evaluations run at
    once, per judge model (`running[model]`) and in total (`running["total"]`),
    and the most of each seen at once in `peaks`. It writes no TSV."""

    async def run(self: LLMJudge) -> None:
        for key in (self.judge_model, "total"):
            running[key] += 1
            peaks[key] = max(peaks[key], running[key])
        await asyncio.sleep(0.02)
        for key in (self.judge_model, "total"):
            running[key] -= 1

    return run


@pytest.mark.integration
@pytest.mark.asyncio
class TestJobLoopConcurrency:
    """`max_concurrent` caps evaluations in total, or per judge model with
    `per_judge`."""

    async def test_per_judge_caps_each_model_and_runs_models_side_by_side(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        tmp_path: Path,
    ):
        """Arrange: 3 conversations, 2 judge models, max_concurrent=1, per_judge
        Act: run_judging
        Assert: each model runs at most 1 evaluation at once, and the two
            models' evaluations overlap
        """
        running: Counter = Counter()
        peaks: Counter = Counter()
        with patch.object(LLMJudge, "run", recording_run(running, peaks)):
            await run_judging(
                judging_config(
                    str(tmp_path),
                    mock_rubric_files,
                    judge_models={"model-a": 1, "model-b": 1},
                    max_concurrent=1,
                    per_judge=True,
                ),
                into=str(tmp_path / "per_judge_output"),
            )

        assert peaks["model-a"] == 1
        assert peaks["model-b"] == 1
        assert peaks["total"] == 2

    async def test_max_concurrent_caps_all_models_together(
        self,
        multiple_conversation_files: List[str],
        mock_rubric_files: str,
        tmp_path: Path,
    ):
        """Arrange: 3 conversations, 2 judge models, max_concurrent=1
        Act: run_judging without per_judge
        Assert: at most 1 evaluation runs at once across both models
        """
        running: Counter = Counter()
        peaks: Counter = Counter()
        with patch.object(LLMJudge, "run", recording_run(running, peaks)):
            await run_judging(
                judging_config(
                    str(tmp_path),
                    mock_rubric_files,
                    judge_models={"model-a": 1, "model-b": 1},
                    max_concurrent=1,
                    per_judge=False,
                ),
                into=str(tmp_path / "shared_cap_output"),
            )

        assert peaks["total"] == 1
        assert peaks["model-a"] == 1
        assert peaks["model-b"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
class TestEventsAndResume:
    """The run's events.jsonl, and continuing a run with `into`."""

    async def test_events_jsonl_records_each_evaluation(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
        mock_llm_factory_for_judge,
    ):
        """Arrange: 2 conversations, one judge model with 2 instances
        Act: run_judging
        Assert: events.jsonl in the run folder has one job_finished per
            evaluation and one stage_finished
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text("user: a\nchatbot: b")
        (conv_folder / "conv2.txt").write_text("user: c\nchatbot: d")

        folder = await run_judging(
            judging_config(
                str(conv_folder),
                mock_rubric_files,
                output=str(tmp_path / "output"),
                judge_models={"mock-judge": 2},
            ),
            run_record=minimal_run_record(),
        )

        assert (Path(folder) / "events.jsonl").is_file()
        types = Counter(event["type"] for event in read_events(folder))
        assert types["job_finished"] == 4
        assert types["stage_finished"] == 1
        assert types["job_failed"] == 0

    async def test_into_skips_evaluations_already_on_disk(
        self,
        tmp_path: Path,
        mock_rubric_files: str,
    ):
        """Arrange: an existing run folder holding the TSV of instance 1 of 2
        Act: run_judging with `into` that folder
        Assert: only instance 2 calls the LLM, and results.csv has both rows
        """
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "a.txt").write_text("user: hi\nchatbot: hello")

        out = tmp_path / "eval"
        out.mkdir()
        existing = out / judge_evaluation_tsv_filename("a.txt", "mock-judge", 1)
        existing.write_text(
            "Dimension\tScore\tReasoning\nTest Dimension\tBest Practice\tok\n",
            encoding="utf-8",
        )

        created: List[MockLLM] = []

        def create_mock_judge(*args, **kwargs: Any) -> MockLLM:
            llm = MockLLM(
                responses=["Yes\n\nREASONING: a", "No\n\nREASONING: b"], **kwargs
            )
            created.append(llm)
            return llm

        with patch(
            "vera_mh.judge.llm_judge.LLMFactory.create_llm",
            side_effect=create_mock_judge,
        ):
            folder = await run_judging(
                judging_config(
                    str(conv_folder),
                    mock_rubric_files,
                    judge_models={"mock-judge": 2},
                ),
                into=str(out),
            )

        assert len(created) == 1
        types = Counter(event["type"] for event in read_events(folder))
        assert types["job_skipped"] == 1
        assert types["job_finished"] == 1

        results = read_results(folder)
        assert sorted(r["judge_instance"] for r in results) == [1, 2]
        assert {r["filename"] for r in results} == {"a.txt"}
