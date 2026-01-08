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
        rubric_df_task = asyncio.to_thread(pd.read_csv, str(rubric_path), sep=sep)
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
        dimensions = cls._extract_dimensions(rubric_df)

        return cls(
            dimensions=dimensions,
            question_flow_data=question_flow_data,
            question_order=question_order,
            rubric_prompt_beginning=rubric_prompt_beginning,
            question_prompt_template=question_prompt_template,
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
            for d in rubric_df["Dimension"].dropna().unique()
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

        for idx, row in rubric_df.iterrows():
            question_id_raw = (
                row["Question ID"] if pd.notna(row["Question ID"]) else None
            )
            # Convert to string and clean up (remove .0 from floats)
            if question_id_raw is not None:
                question_id = (
                    str(int(question_id_raw))
                    if isinstance(question_id_raw, (int, float))
                    else str(question_id_raw).strip()
                )
            else:
                question_id = ""

            # If this row has a Question ID, it's a new question
            if question_id and question_id != "nan":
                # Save previous question if exists
                if current_question_id and current_question_data:
                    questions[current_question_id] = current_question_data

                # Read severity from the question row
                severity = (
                    str(row["Severity"]).strip() if pd.notna(row["Severity"]) else ""
                )
                severity = (
                    severity if severity and severity not in ["nan", ""] else None
                )

                # Start new question
                current_question_id = question_id
                question_order.append(question_id)
                current_question_data = {
                    "dimension": str(row["Dimension"]).strip()
                    if pd.notna(row["Dimension"])
                    else "",
                    "risk_type": str(row["Risk Type"]).strip()
                    if pd.notna(row["Risk Type"])
                    else "",
                    "question": str(row["Question"]).strip()
                    if pd.notna(row["Question"])
                    else "",
                    "examples": str(row["Examples"]).strip()
                    if pd.notna(row["Examples"])
                    else "",
                    "severity": severity,
                    "answers": [],
                }

                # Check if this row also has an answer (single-row question)
                answer = str(row["Answer"]).strip() if pd.notna(row["Answer"]) else ""
                if answer and answer != "nan":
                    goto_raw = row["GOTO"] if pd.notna(row["GOTO"]) else None
                    goto = (
                        str(int(goto_raw))
                        if goto_raw and isinstance(goto_raw, (int, float))
                        else (str(goto_raw).strip() if goto_raw else None)
                    )
                    current_question_data["answers"].append(
                        {
                            "option": answer,
                            "goto": goto if goto and goto != "nan" else None,
                        }
                    )

            # This is a continuation row with an answer option
            elif current_question_data is not None:
                answer = str(row["Answer"]).strip() if pd.notna(row["Answer"]) else ""
                if answer and answer != "nan":
                    goto_raw = row["GOTO"] if pd.notna(row["GOTO"]) else None
                    goto = (
                        str(int(goto_raw))
                        if goto_raw and isinstance(goto_raw, (int, float))
                        else (str(goto_raw).strip() if goto_raw else None)
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
        for question_id in question_order:
            question_data = questions[question_id]
            if len(question_data["answers"]) == 0:
                question_data["answers"] = [
                    {"option": "Yes", "goto": None},
                    {"option": "No", "goto": None},
                ]

        return questions, question_order


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
