"""Data structures and loaders for rubric and conversation configuration.

This module provides pre-loaded data structures that eliminate the need for
file caching. All files are read once at startup and passed through the API
as data structures rather than file paths.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import pandas as pd

from judge.question_navigator import QuestionNavigator
from utils.rubric_manifest import load_manifest

# Rubric TSV column names - single source of truth for rubric structure
COL_QUESTION_ID = "Question ID"
COL_DIMENSION = "Dimension"
COL_SEVERITY = "Severity"
COL_RISK_TYPE = "Risk Type"
COL_QUESTION = "Question"
COL_EXAMPLES = "Examples"
COL_ANSWER = "Answer"
COL_GOTO = "GOTO"
COL_AUTO_ANSWER = "Auto Answer"

# Rubric TSV columns to ignore
IGNORE_COLUMNS = {"Human notes", "Notes for Interpretability of GOTO logic"}

# Dimension names
DETECTS_POTENTIAL_RISK = "Detects Potential Risk"
CONFIRMS_RISK = "Confirms Risk"
GUIDES_TO_HUMAN_CARE = "Guides to Human Care"
SUPPORTIVE_CONVERSATION = "Supportive Conversation"
FOLLOWS_AI_BOUNDARIES = "Follows AI Boundaries"

# Expected dimension names
EXPECTED_DIMENSION_NAMES = {
    DETECTS_POTENTIAL_RISK,
    CONFIRMS_RISK,
    GUIDES_TO_HUMAN_CARE,
    SUPPORTIVE_CONVERSATION,
    FOLLOWS_AI_BOUNDARIES,
}


@dataclass
class RubricConfig:
    """Parsed rubric configuration data.

    Contains all rubric-related data loaded from files, eliminating the need
    for file paths and caching in downstream components.
    """

    dimensions: List[str]
    question_flow_data: Dict[str, Dict[str, Any]]
    question_order: List[str]
    rubric_prompt_beginning: str
    question_prompt_template: str

    @classmethod
    async def load(
        cls,
        rubric_folder: str = "data",
        rubric_file: str = "rubric.tsv",
        rubric_prompt_beginning_file: str = "rubric_prompt_beginning.txt",
        question_prompt_file: str = "question_prompt.txt",
        sep: str = "\t",
    ) -> "RubricConfig":
        """Load all rubric data from files asynchronously.

        Args:
            rubric_folder: Folder containing rubric files
            rubric_file: Rubric TSV filename
            rubric_prompt_beginning_file: System prompt template filename
            question_prompt_file: Question prompt template filename
            sep: Separator for TSV file (default: tab)

        Returns:
            Loaded RubricConfig with all data

        Raises:
            FileNotFoundError: If any required file doesn't exist
        """
        rubric_path = Path(rubric_folder) / rubric_file
        rubric_prompt_beginning_path = (
            Path(rubric_folder) / rubric_prompt_beginning_file
        )
        question_prompt_path = Path(rubric_folder) / question_prompt_file

        # Validate files exist
        if not rubric_path.exists():
            raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
        if not rubric_prompt_beginning_path.exists():
            raise FileNotFoundError(
                f"Rubric prompt file not found: {rubric_prompt_beginning_path}"
            )
        if not question_prompt_path.exists():
            raise FileNotFoundError(
                f"Question prompt file not found: {question_prompt_path}"
            )

        # Load all files in parallel
        rubric_df_task = asyncio.to_thread(
            pd.read_csv, str(rubric_path), sep=sep, dtype=str
        )
        rubric_prompt_task = cls._read_file(rubric_prompt_beginning_path)
        question_prompt_task = cls._read_file(question_prompt_path)

        (
            rubric_df,
            rubric_prompt_beginning,
            question_prompt_template,
        ) = await asyncio.gather(
            rubric_df_task, rubric_prompt_task, question_prompt_task
        )

        # Parse rubric structure
        question_flow_data, question_order = cls._parse_rubric(rubric_df)
        cls._validate_navigation(question_flow_data, question_order)
        dimensions = cls._extract_dimensions(rubric_df)

        return cls(
            dimensions=dimensions,
            question_flow_data=question_flow_data,
            question_order=question_order,
            rubric_prompt_beginning=rubric_prompt_beginning,
            question_prompt_template=question_prompt_template,
        )

    @classmethod
    async def load_bundle(cls, manifest_path: str) -> "RubricConfig":
        """Load a rubric from a rubric bundle manifest.

        A rubric bundle manifest is a JSON file describing what a rubric
        *is* -- its files and (informational only) intended personas -- as
        opposed to how to run it, which belongs in a run config, not here.

        Manifest shape:
            {
              "rubric_file": "rubric.tsv",
              "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
              "question_prompt_file": "question_prompt.txt",
              "personas": ["personas.tsv"]
            }

        `personas` is informational only -- it documents which personas this
        rubric is intended/validated for; it is not read by this loader.
        File paths in the manifest are relative to the manifest's own folder.

        Args:
            manifest_path: Path to the rubric bundle manifest JSON file

        Returns:
            Loaded RubricConfig with all data

        Raises:
            FileNotFoundError: If the manifest or any file it references
                doesn't exist
            ValueError: If the manifest is missing a required key
        """
        manifest_file = Path(manifest_path)
        manifest = await load_manifest(manifest_path)

        return await cls.load(
            rubric_folder=str(manifest_file.parent),
            rubric_file=manifest["rubric_file"],
            rubric_prompt_beginning_file=manifest["rubric_prompt_beginning_file"],
            question_prompt_file=manifest["question_prompt_file"],
        )

    @staticmethod
    async def _read_file(file_path: Path) -> str:
        """Read text file asynchronously.

        Args:
            file_path: Path to file

        Returns:
            File contents as string
        """
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            return await f.read()

    @staticmethod
    def _extract_dimensions(rubric_df: pd.DataFrame) -> List[str]:
        """Extract unique dimensions from rubric DataFrame.

        Args:
            rubric_df: Loaded rubric DataFrame

        Returns:
            List of unique dimension names
        """
        dimensions = [
            d.strip()
            for d in rubric_df[COL_DIMENSION].dropna().unique()
            if d and str(d).strip() != "nan"
        ]
        return dimensions

    @staticmethod
    def _parse_rubric(
        rubric_df: pd.DataFrame,
    ) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Parse the rubric DataFrame into a navigable data structure.

        The rubric has questions with potential multi-row answer options.
        Questions have a Question ID, and subsequent rows with blank Question ID
        contain answer options for that question.

        Args:
            rubric_df: Loaded rubric DataFrame

        Returns:
            Tuple of (questions_dict, question_order_list):
            - questions_dict: Dictionary mapping Question ID to question data
            - question_order_list: Ordered list of Question IDs
        """
        questions = {}
        question_order = []
        current_question_id = None
        current_question_data = None

        for _, row in rubric_df.iterrows():
            question_id_raw = (
                row[COL_QUESTION_ID] if pd.notna(row[COL_QUESTION_ID]) else None
            )
            question_id = RubricConfig._clean_identifier(question_id_raw)

            # If this row has a Question ID, it's a new question
            if question_id and question_id != "nan":
                if question_id in questions or question_id == current_question_id:
                    raise ValueError(f"Duplicate Question ID: {question_id!r}")

                # Save previous question if exists
                if current_question_id and current_question_data:
                    questions[current_question_id] = current_question_data

                dimension = (
                    str(row[COL_DIMENSION]).strip()
                    if pd.notna(row[COL_DIMENSION])
                    else ""
                )
                if not dimension or dimension == "nan":
                    raise ValueError(
                        f"Question {question_id!r} must declare a Dimension "
                        "on its primary row"
                    )

                auto_answer_raw = (
                    row[COL_AUTO_ANSWER]
                    if COL_AUTO_ANSWER in rubric_df.columns
                    else None
                )
                auto_answer = RubricConfig._parse_auto_answer(
                    auto_answer_raw, question_id
                )

                # Read severity from the question row
                severity = (
                    str(row[COL_SEVERITY]).strip()
                    if pd.notna(row[COL_SEVERITY])
                    else ""
                )
                severity = (
                    severity if severity and severity not in ["nan", ""] else None
                )

                # Start new question
                current_question_id = question_id
                question_order.append(question_id)
                current_question_data = {
                    "dimension": dimension,
                    "risk_type": str(row[COL_RISK_TYPE]).strip()
                    if pd.notna(row[COL_RISK_TYPE])
                    else "",
                    "question": str(row[COL_QUESTION]).strip()
                    if pd.notna(row[COL_QUESTION])
                    else "",
                    "examples": str(row[COL_EXAMPLES]).strip()
                    if pd.notna(row[COL_EXAMPLES])
                    else "",
                    "severity": severity,
                    "auto_answer": auto_answer,
                    "answers": [],
                }

                # Check if this row also has an answer (single-row question)
                answer = (
                    str(row[COL_ANSWER]).strip() if pd.notna(row[COL_ANSWER]) else ""
                )
                if answer and answer != "nan":
                    goto = RubricConfig._clean_identifier(
                        row[COL_GOTO] if pd.notna(row[COL_GOTO]) else None
                    )
                    current_question_data["answers"].append(
                        {
                            "option": answer,
                            "goto": goto if goto and goto != "nan" else None,
                        }
                    )

            # This is a continuation row with an answer option
            elif current_question_data is not None:
                answer = (
                    str(row[COL_ANSWER]).strip() if pd.notna(row[COL_ANSWER]) else ""
                )
                if answer and answer != "nan":
                    goto = RubricConfig._clean_identifier(
                        row[COL_GOTO] if pd.notna(row[COL_GOTO]) else None
                    )
                    current_question_data["answers"].append(
                        {
                            "option": answer,
                            "goto": goto if goto and goto != "nan" else None,
                        }
                    )

        # Save last question
        if current_question_id and current_question_data:
            questions[current_question_id] = current_question_data

        # Add default Yes/No answers for questions without explicit answers
        # (empty Answer column). Mark as implicit_yes_no so navigator uses
        # empty-Answer logic: Yes -> first question of next dimension,
        # No -> next row. Severity is still assigned from the question row.
        for question_id in question_order:
            question_data = questions[question_id]
            if question_data["auto_answer"] and len(question_data["answers"]) != 1:
                raise ValueError(
                    f"Question {question_id!r} has Auto Answer=true but must "
                    "declare exactly one explicit answer"
                )
            if len(question_data["answers"]) == 0:
                question_data["implicit_yes_no"] = True
                question_data["answers"] = [
                    {"option": "Yes", "goto": None},
                    {"option": "No", "goto": None},
                ]

        return questions, question_order

    @staticmethod
    def _clean_identifier(value: Any) -> str:
        """Return a question ID or GOTO target as a stripped opaque string."""
        if value is None or pd.isna(value):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    @staticmethod
    def _parse_auto_answer(value: Any, question_id: str) -> bool:
        """Parse the optional Auto Answer cell on a primary question row."""
        if value is None or pd.isna(value) or not str(value).strip():
            return False

        normalized = str(value).strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise ValueError(
            f"Question {question_id!r} has invalid Auto Answer value {value!r}; "
            "expected true, false, or blank"
        )

    @staticmethod
    def _validate_navigation(
        questions: Dict[str, Dict[str, Any]], question_order: List[str]
    ) -> None:
        """Validate navigation targets and reject every reachable graph cycle."""
        navigator = QuestionNavigator(questions, question_order)
        edges: Dict[str, List[str]] = {
            question_id: [] for question_id in question_order
        }

        for question_id in question_order:
            for answer in questions[question_id]["answers"]:
                next_question_id, _ = navigator.get_next_question(
                    question_id, answer["option"]
                )
                if next_question_id is None:
                    continue
                if next_question_id not in questions:
                    raise ValueError(
                        f"Question {question_id!r} answer {answer['option']!r} "
                        f"targets missing question {next_question_id!r}"
                    )
                edges[question_id].append(next_question_id)

        state: Dict[str, int] = {}
        stack: List[str] = []
        stack_positions: Dict[str, int] = {}

        def visit(question_id: str) -> None:
            state[question_id] = 1
            stack_positions[question_id] = len(stack)
            stack.append(question_id)

            for next_question_id in edges[question_id]:
                if state.get(next_question_id) == 1:
                    cycle_start = stack_positions[next_question_id]
                    cycle = stack[cycle_start:] + [next_question_id]
                    raise ValueError(
                        "Rubric navigation contains a cycle: " + " -> ".join(cycle)
                    )
                if state.get(next_question_id, 0) == 0:
                    visit(next_question_id)

            stack.pop()
            stack_positions.pop(question_id)
            state[question_id] = 2

        for question_id in question_order:
            if state.get(question_id, 0) == 0:
                visit(question_id)


@dataclass
class ConversationData:
    """Single conversation data with metadata.

    Contains conversation content and metadata, eliminating the need
    to pass file paths through the evaluation pipeline.
    """

    content: str
    metadata: Dict[str, str]  # filename, run_id, source_path

    @classmethod
    async def load(cls, file_path: str) -> "ConversationData":
        """Load a single conversation file asynchronously.

        Args:
            file_path: Path to conversation file

        Returns:
            ConversationData with content and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Conversation file not found: {file_path}")

        # Read file content
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()

        # Extract metadata from path
        metadata = {
            "filename": path.name,
            "run_id": path.parent.name,
            "source_path": str(path),
        }

        return cls(content=content, metadata=metadata)


async def load_conversations(
    folder: str, limit: Optional[int] = None
) -> List[ConversationData]:
    """Load all conversation files from a folder in parallel.

    Args:
        folder: Folder containing conversation .txt files
        limit: Optional limit on number of conversations to load

    Returns:
        List of ConversationData objects

    Raises:
        FileNotFoundError: If folder doesn't exist or contains no .txt files
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Find all conversation files
    conversation_files = list(folder_path.glob("*.txt"))
    if not conversation_files:
        raise FileNotFoundError(f"No .txt files found in: {folder}")

    # Apply limit if specified
    if limit is not None:
        conversation_files = conversation_files[:limit]

    # Load all conversations in parallel
    tasks = [ConversationData.load(str(f)) for f in conversation_files]
    conversations = await asyncio.gather(*tasks)

    return conversations
