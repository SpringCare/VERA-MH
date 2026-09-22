"""Per-question answer artifacts for judge runs.

The evaluation TSV that ``LLMJudge`` writes is dimension-level: five rows of
``Dimension / Score / Reasoning``. The individual rubric answers behind those
scores are collapsed away, so "what did the judge answer for question 9?" is
only recoverable by grepping the run log.

This module persists those answers as data, in two artifacts under an
``answers/`` subfolder of the evaluation folder:

``answers/conversations/<evaluation tsv stem>.tsv``
    One file per judged conversation, long format -- one row per question the
    flow actually visited, in visit order.

``answers/answers.tsv``
    The run-level aggregate, wide format -- one row per conversation, with a
    ``Q{id}`` and ``Q{id} explanation`` column pair per rubric question, in
    rubric order. Questions the flow skipped are blank, so a row also shows
    which branch the conversation took.

Both live in a subfolder because the evaluation folder's own ``*.tsv`` glob
(``build_results_csv_from_tsv_files``, and the resume scan in
``judge.runner``) treats every TSV directly inside it as an evaluation.
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from .score_utils import (
    extract_conversation_filename_from_tsv,
    parse_judge_metadata_from_evaluation_tsv_filename,
)
from .utils import extract_persona_name_from_filename

ANSWERS_DIR_NAME = "answers"
CONVERSATIONS_DIR_NAME = "conversations"
ANSWERS_TSV_NAME = "answers.tsv"

COL_QUESTION_ID = "Question ID"
COL_ANSWER = "Answer"
COL_EXPLANATION = "Explanation"

IDENTITY_COLUMNS = [
    "filename",
    "persona_name",
    "run_id",
    "judge_model",
    "judge_instance",
    "judge_id",
]


def answers_dir(evaluation_folder: Path | str) -> Path:
    """Folder holding this run's per-question answer artifacts."""
    return Path(evaluation_folder) / ANSWERS_DIR_NAME


def conversation_answers_dir(evaluation_folder: Path | str) -> Path:
    """Folder holding the per-conversation answer files."""
    return answers_dir(evaluation_folder) / CONVERSATIONS_DIR_NAME


def answers_tsv_path(evaluation_folder: Path | str) -> Path:
    """Path of the run-level wide answers TSV."""
    return answers_dir(evaluation_folder) / ANSWERS_TSV_NAME


def answer_column(question_id: str) -> str:
    """Wide-format column holding the answer to ``question_id``."""
    return f"Q{question_id}"


def explanation_column(question_id: str) -> str:
    """Wide-format column holding the explanation for ``question_id``."""
    return f"Q{question_id} explanation"


def _flatten(text: Any) -> str:
    """Collapse a value to a single TSV-safe cell."""
    if text is None:
        return ""
    return str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def write_conversation_answers(
    output_file: Path, question_log: Sequence[Dict[str, Any]], sep: str = "\t"
) -> None:
    """Write one conversation's visited questions in visit order.

    ``question_log`` is the append-only record of questions asked, not
    ``dimension_answers`` -- the latter is rewritten in place when a
    ``NOT_RELEVANT>>`` goto fires, which drops the answers it replaces.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{COL_QUESTION_ID}{sep}{COL_ANSWER}{sep}{COL_EXPLANATION}\n")
        for entry in question_log:
            question_id = _flatten(entry.get("question_id"))
            answer = _flatten(entry.get("answer"))
            explanation = _flatten(entry.get("reasoning"))
            f.write(f"{question_id}{sep}{answer}{sep}{explanation}\n")


def build_answers_dataframe(
    evaluation_folder: Path | str, question_order: Sequence[str]
) -> pd.DataFrame:
    """Assemble the run-level wide answers table.

    One row per per-conversation answer file, one ``Q{id}`` /
    ``Q{id} explanation`` column pair per rubric question in ``question_order``.
    Returns an empty (but correctly-columned) frame when the run has no answer
    files, so callers can write it unconditionally.
    """
    columns: List[str] = list(IDENTITY_COLUMNS)
    for question_id in question_order:
        columns.append(answer_column(question_id))
        columns.append(explanation_column(question_id))

    folder = Path(evaluation_folder)
    conversations_dir = conversation_answers_dir(folder)
    if not conversations_dir.is_dir():
        return pd.DataFrame(columns=columns)

    run_id = folder.name.split("__")[-1] if "__" in folder.name else folder.name
    known_questions = set(question_order)

    rows: List[Dict[str, Any]] = []
    for answer_file in sorted(conversations_dir.glob("*.tsv")):
        try:
            answers_df = pd.read_csv(
                answer_file, sep="\t", dtype=str, keep_default_na=False
            )
        except Exception as e:  # noqa: BLE001 - one bad file must not sink the run
            print(f"Warning: Error reading answers file {answer_file}: {e}")
            continue

        judge_model, judge_instance = parse_judge_metadata_from_evaluation_tsv_filename(
            answer_file.name
        )
        conversation_filename = extract_conversation_filename_from_tsv(answer_file.name)
        row: Dict[str, Any] = {column: "" for column in columns}
        row.update(
            {
                "filename": conversation_filename,
                "persona_name": extract_persona_name_from_filename(
                    conversation_filename
                )
                or "",
                "run_id": run_id,
                "judge_model": judge_model,
                "judge_instance": judge_instance,
                "judge_id": max(0, judge_instance - 1),
            }
        )

        for _, answer_row in answers_df.iterrows():
            question_id = str(answer_row.get(COL_QUESTION_ID, "")).strip()
            if not question_id or question_id not in known_questions:
                # A rubric the run was not judged with, or a stray row.
                continue
            row[answer_column(question_id)] = str(
                answer_row.get(COL_ANSWER, "")
            ).strip()
            row[explanation_column(question_id)] = str(
                answer_row.get(COL_EXPLANATION, "")
            ).strip()

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def write_answers_tsv(
    evaluation_folder: Path | str, question_order: Sequence[str]
) -> Path:
    """Build and write ``answers/answers.tsv``; returns the path written."""
    df = build_answers_dataframe(evaluation_folder, question_order)
    out_path = answers_tsv_path(evaluation_folder)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    return out_path
