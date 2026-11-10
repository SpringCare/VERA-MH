"""Data structures and loaders for rubric and conversation configuration.

This module provides pre-loaded data structures that eliminate the need for
file caching. All files are read once at startup and passed through the API
as data structures rather than file paths.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiofiles
import pandas as pd

from vera_mh.judge.question_navigator import QuestionNavigator
from vera_mh.utils.rubric_manifest import load_manifest_rubric_paths

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

# Asks one rubric question and returns (answer, reasoning): a judge LLM in a
# real run (`JudgeLLM.answer`), a script in tests.
Ask = Callable[[str, Dict[str, Any]], Awaitable[tuple[str, str]]]


@dataclass
class RubricWalk:
    """The answers one walk through a rubric collected."""

    # Answers by dimension, for scoring (`judge/scoring.py`). A NOT_RELEVANT>>
    # route replaces a dimension's list, so this isn't a full record.
    dimension_answers: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Every question asked, in order, never rewritten: the per-question
    # answers file is written from it.
    question_log: List[Dict[str, Any]] = field(default_factory=list)
    # The question whose END or ASSIGN_END stopped the walk, or None if it ran
    # out of questions.
    stopped_at: Optional[str] = None


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

    def question_prompt(self, question_data: Dict[str, Any]) -> str:
        """The prompt asking one question: the rubric's question template
        filled with the question, its examples, and its answer options."""
        examples = question_data.get("examples", "")
        return self.question_prompt_template.format(
            question=question_data["question"],
            examples_section=f"\n{examples}\n" if examples else "",
            options=", ".join(answer_options(question_data)),
        )

    async def walk(
        self, ask: Ask, start_question_id: Optional[str] = None
    ) -> RubricWalk:
        """Ask the rubric's questions in the order its answers route them,
        until END/ASSIGN_END or no question is left.

        For each question:
        1. Ask it with `ask`, unless the rubric declares its single answer
           deterministic (`auto_answer`)
        2. Store the answer for scoring, and log it
        3. Go where the answer routes: the next question, NOT_RELEVANT>>
           (this dimension is Not Relevant, continue), ASSIGN_END, or END

        Args:
            ask: Asks one question, returning (answer, reasoning)
            start_question_id: Question to start with (default: the first)
        """
        if start_question_id is None:
            if not self.question_order:
                raise ValueError("No questions found in rubric")
            start_question_id = self.question_order[0]
        navigator = QuestionNavigator(
            question_flow_data=self.question_flow_data,
            question_order=self.question_order,
        )
        walk = RubricWalk()
        current_question_id: Optional[str] = start_question_id
        visited_questions = set()
        current_dimension = None

        while current_question_id:
            # Safety check: prevent infinite loops
            # Rubrics are validated at load time; retain this as defense in depth.
            if current_question_id in visited_questions:
                break
            visited_questions.add(current_question_id)

            # Get question data from rubric
            question_data = navigator.get_question_data(current_question_id)
            if not question_data:
                break

            # Step 1: Ask the question, unless the rubric explicitly declares
            # its single answer deterministic.
            if question_data.get("auto_answer"):
                answer_text = question_data["answers"][0]["option"]
                reasoning = "Automatically selected by rubric"
            else:
                answer_text, reasoning = await ask(current_question_id, question_data)

            # Update current dimension if this question has one
            dimension = question_data.get("dimension")
            if dimension:
                current_dimension = dimension

            # Step 2: Store answer for this dimension
            store_answer(
                walk.dimension_answers,
                question_data,
                current_question_id,
                answer_text,
                dimension or current_dimension,
                reasoning,
            )
            walk.question_log.append(
                {
                    "question_id": current_question_id,
                    "dimension": dimension or current_dimension,
                    "answer": answer_text,
                    "severity": question_data.get("severity"),
                    "reasoning": reasoning,
                }
            )

            # Step 3: Determine next question
            next_question_id, goto_value = navigator.get_next_question(
                current_question_id, answer_text
            )

            # Step 4: Handle special GOTO values (END conditions and NOT_RELEVANT>>)
            # NOT_RELEVANT>>{ID}: Mark current dimension as Not Relevant
            # and jump to {ID}
            if goto_value and goto_value.startswith("NOT_RELEVANT>>"):
                mark_not_relevant(
                    current_question_id,
                    answer_text,
                    dimension or current_dimension,
                    walk.dimension_answers,
                )
                current_question_id = next_question_id
                continue

            # ASSIGN_END is a special case of END that assigns severity
            # to current dimension, and marks the others Not Relevant.
            # Scoring tells it apart from a regular END by the answers.
            if goto_value == "ASSIGN_END":
                mark_others_not_relevant(
                    self.dimensions,
                    current_question_id,
                    answer_text,
                    dimension or current_dimension,
                    walk.dimension_answers,
                )
                walk.stopped_at = current_question_id
                return walk

            if goto_value == "END":
                # Regular END: scoring marks every dimension Not Relevant
                walk.stopped_at = current_question_id
                return walk

            # Step 5: Continue to next question
            current_question_id = next_question_id

        return walk

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
        return await cls.from_paths(
            rubric_file=str(Path(rubric_folder) / rubric_file),
            rubric_prompt_beginning_file=str(
                Path(rubric_folder) / rubric_prompt_beginning_file
            ),
            question_prompt_file=str(Path(rubric_folder) / question_prompt_file),
            sep=sep,
        )

    @classmethod
    async def from_paths(
        cls,
        *,
        rubric_file: str,
        rubric_prompt_beginning_file: str,
        question_prompt_file: str,
        sep: str = "\t",
    ) -> "RubricConfig":
        """Load rubric data from three already-resolved file paths.

        This is the resolved-value entry point: callers pass complete paths and
        this does no joining, defaulting, or manifest reading. `load` and
        `load_bundle` are the convenience wrappers that resolve their inputs
        down to these three paths.

        Args:
            rubric_file: Path to the rubric TSV
            rubric_prompt_beginning_file: Path to the system prompt template
            question_prompt_file: Path to the question prompt template
            sep: Separator for the rubric TSV (default: tab)

        Returns:
            Loaded RubricConfig with all data

        Raises:
            FileNotFoundError: If any required file doesn't exist
        """
        rubric_path = Path(rubric_file)
        rubric_prompt_beginning_path = Path(rubric_prompt_beginning_file)
        question_prompt_path = Path(question_prompt_file)

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
        # Validate the complete in-memory navigation graph once; this does not
        # reread the rubric for each question.
        cls._validate_navigation(question_flow_data, question_order)
        cls._validate_severity_ordering(question_flow_data, question_order)
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
        """Load a rubric's configuration and metadata
        from a rubric bundle manifest.

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
        return await cls.from_paths(**await load_manifest_rubric_paths(manifest_path))

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
        """Validate navigation targets by exooring all possible paths,
        and reject every reachable graph cycle."""
        navigator = QuestionNavigator(questions, question_order)
        edges: Dict[str, List[str]] = {
            question_id: [] for question_id in question_order
        }

        for question_id in question_order:
            for answer in questions[question_id]["answers"]:
                # ASSIGN_END assigns this question's Severity to the current
                # dimension, and severity only means anything when the finding
                # is present -- which for a rubric question is a "Yes". Putting
                # it on any other option would ask the scorer to penalize a
                # dimension for an answer that reported no problem, so it is
                # rejected at load rather than reinterpreted at scoring time.
                if (
                    answer.get("goto") == "ASSIGN_END"
                    and answer["option"].strip().lower() != "yes"
                ):
                    raise ValueError(
                        f"Question {question_id!r} routes answer "
                        f"{answer['option']!r} to ASSIGN_END, which is only "
                        f"valid on a 'Yes' answer. Use END to stop without "
                        f"assigning severity."
                    )
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

    @staticmethod
    def _validate_severity_ordering(
        questions: Dict[str, Dict[str, Any]], question_order: List[str]
    ) -> None:
        """Reject a dimension that asks a Red question after a Yellow one.

        Within a dimension the more severe questions come first, so the first
        "Yes" a dimension collects is also its worst finding. Scoring relies on
        that: a "Yes" ends the dimension, and
        `_calculate_score_from_severity` reports only the highest tier it was
        given. A Yellow asked before a Red could therefore end the dimension on
        the Yellow and never ask the Red at all, under-reporting the severity.

        Checked per *reachable path*, not per row order, because a dimension may
        legitimately hold parallel branches that are never both visited -- SI's
        `Guides to Human Care` splits on Q9 into a not-immediate-risk branch
        (Q10-Q15) and an immediate-risk branch (Q16-Q22), each internally
        Red-then-Yellow. Row order alone shows Q16 (Red) after Q13 (Yellow) and
        would reject a correct rubric.

        State is (question, dimensions that have shown a Yellow so far) and is
        memoized, so this is linear in questions times dimension subsets rather
        than exponential in path count.
        """
        if not question_order:
            return

        navigator = QuestionNavigator(questions, question_order)
        # Carries the Yellow question id per dimension so the error can name it.
        start: tuple[str, frozenset] = (question_order[0], frozenset())
        stack = [start]
        seen: set[tuple[str, frozenset]] = set()
        first_yellow: Dict[tuple[str, str], str] = {}

        while stack:
            question_id, yellow_dims = stack.pop()
            if (question_id, yellow_dims) in seen:
                continue
            seen.add((question_id, yellow_dims))

            data = questions[question_id]
            dimension = data.get("dimension", "")
            severity = (data.get("severity") or "").strip().lower()

            if severity == "red" and dimension in yellow_dims:
                earlier = first_yellow.get((question_id, dimension), "")
                raise ValueError(
                    f"Dimension {dimension!r} asks Red question "
                    f"{question_id!r} after Yellow question {earlier!r} on the "
                    f"same path; within a dimension Red questions must come "
                    f"before Yellow ones"
                )

            next_yellow = yellow_dims
            if severity == "yellow" and dimension not in yellow_dims:
                next_yellow = yellow_dims | {dimension}

            for answer in data.get("answers", []):
                next_question_id, _ = navigator.get_next_question(
                    question_id, answer["option"]
                )
                if next_question_id is None or next_question_id not in questions:
                    continue
                if next_yellow is not yellow_dims:
                    first_yellow.setdefault((next_question_id, dimension), question_id)
                stack.append((next_question_id, next_yellow))


def answer_options(question_data: Dict[str, Any]) -> List[str]:
    """A question's answer options, or Yes/No when the rubric lists none."""
    answers = question_data.get("answers", [])
    return [ans["option"] for ans in answers] if answers else ["Yes", "No"]


def store_answer(
    dimension_answers: Dict[str, List[Dict[str, Any]]],
    question_data: Dict[str, Any],
    question_id: str,
    answer: str,
    dimension: Optional[str],
    reasoning: str,
) -> None:
    """Store an answer under its dimension, for scoring."""
    if not dimension:
        return

    if dimension not in dimension_answers:
        dimension_answers[dimension] = []

    dimension_answers[dimension].append(
        {
            "question_id": question_id,
            "question": question_data["question"],
            "answer": answer,
            "severity": question_data.get("severity"),
            "reasoning": reasoning,
        }
    )


def mark_not_relevant(
    question_id: str,
    answer: str,
    current_dimension: Optional[str],
    dimension_answers: Dict[str, List[Dict[str, Any]]],
) -> None:
    """NOT_RELEVANT>>: mark only the current dimension as Not Relevant,
    replacing its answers."""
    if current_dimension:
        dimension_answers[current_dimension] = [
            {
                "question_id": question_id,
                "question": (
                    f"Marked Not Relevant due to NOT_RELEVANT>> "
                    f"at Question {question_id}"
                ),
                "answer": "Not Relevant",
                "severity": None,
                "reasoning": (
                    f"Question {question_id} answered '{answer}' "
                    f"triggered NOT_RELEVANT>> - dimension marked Not Relevant"
                ),
            }
        ]


def mark_others_not_relevant(
    dimensions: List[str],
    question_id: str,
    answer: str,
    current_dimension: Optional[str],
    dimension_answers: Dict[str, List[Dict[str, Any]]],
) -> None:
    """ASSIGN_END: mark every other dimension not yet answered Not Relevant."""
    for dimension in dimensions:
        if dimension != current_dimension and dimension not in dimension_answers:
            dimension_answers[dimension] = [
                {
                    "question_id": question_id,
                    "question": (
                        f"Marked Not Relevant due to ASSIGN_END "
                        f"at Question {question_id}"
                    ),
                    "answer": "Not Relevant",
                    "severity": None,
                    "reasoning": (
                        f"Question {question_id} answered '{answer}' "
                        f"triggered ASSIGN_END - dimension marked Not Relevant"
                    ),
                }
            ]


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
