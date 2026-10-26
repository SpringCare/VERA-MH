"""Utility functions for the judge module."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from vera_mh.utils.judge_results import (
    extract_persona_name_from_filename,
    load_persona_column,
)
from vera_mh.utils.logging_utils import (
    extract_last_response_metadata,
    log_llm_failure_metadata,
    log_llm_response_metadata,
)

__all__ = [
    "extract_last_response_metadata",
    "log_llm_failure_metadata",
    "log_llm_response_metadata",
]


def parse_judge_models(model_arg):
    """Parse judge model specifications from command line argument into a dictionary."""
    judge_models = {}
    for model_spec in model_arg:
        if ":" in model_spec:
            # Format: "model:count"
            model, count = model_spec.rsplit(":", 1)
            try:
                n = int(count)
            except ValueError:
                raise ValueError(
                    f"Judge model count must be an integer, got {count!r}"
                ) from None
            if n < 1:
                raise ValueError(f"Judge model count must be positive, got {n}")
            judge_models[model] = n
        else:
            # Format: "model" (defaults to 1 instance)
            judge_models[model_spec] = 1

    return judge_models


def build_evaluation_run_folder_path(
    output_root: str,
    judge_info: str,
    timestamp: str,
    conversation_run_basename: str,
) -> str:
    """
    Build the folder path for a judge evaluation run.
    """
    folder_name = f"j_{judge_info}_{timestamp}__{conversation_run_basename}"
    return str(Path(output_root) / folder_name)


def judge_evaluation_tsv_filename(
    conversation_filename: str,
    judge_model: str,
    judge_instance: Optional[int] = None,
) -> str:
    """
    Basename of the per-evaluation TSV written by LLMJudge._save_results.

    When judge_instance is set (batch runner), suffix is _i{instance}. When None
    (e.g. single-conversation CLI), there is no _i suffix — matches historical behavior.
    """
    conversation_name = Path(conversation_filename).stem
    judge_suffix = judge_model.replace("/", "_").replace(":", "_")
    if judge_instance is not None:
        judge_suffix += f"_i{judge_instance}"
    return f"{conversation_name}_{judge_suffix}.tsv"


def build_single_conversation_judge_run_key(
    conversation_path: str | Path,
    *,
    now: Optional[datetime] = None,
) -> str:
    """
    Human-readable folder basename for ``judge.py`` single-file (``-c``) mode.

    Format: ``single_<timestamp_ms>__<conversation_stem>``. Batch judging uses the
    evaluation folder basename instead; this gives a stable, unique folder name
    per CLI run. Per-task logs use :func:`build_judge_task_log_path` with
    ``output_folder`` set, so no ``run_key`` is involved.
    """
    dt = datetime.now() if now is None else now
    ts = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    stem = Path(conversation_path).stem
    return f"single_{ts}__{stem}"


def get_judge_logs_root() -> str:
    """
    Root directory for legacy per-task judge LLM logs (when ``output_folder`` is
    omitted from :func:`build_judge_task_log_path`).

    Override with env ``VERA_JUDGE_LOGS_ROOT`` (e.g. set by pytest to a temp dir).
    Default: ``judge_logs`` in the current working directory.
    """
    return os.environ.get("VERA_JUDGE_LOGS_ROOT", "judge_logs")


def build_judge_task_log_path(
    conversation_filename: str,
    judge_model: str,
    judge_instance: Optional[int] = None,
    *,
    run_key: Optional[str] = None,
    logs_root: Optional[str] = None,
    output_folder: Optional[str] = None,
) -> str:
    """
    Path to the per-task judge LLM log file, parallel to judge_evaluation_tsv_filename.

    With ``output_folder``: ``{output_folder}/logs/{stem}.log`` (``run_key`` unused).

    Legacy layout (no ``output_folder``): ``{logs_root}/{run_key}/{stem}.log``
    (``logs_root`` defaults to :func:`get_judge_logs_root`). ``run_key`` is required
    for this branch.
    """
    tsv_name = judge_evaluation_tsv_filename(
        conversation_filename, judge_model, judge_instance
    )
    stem = Path(tsv_name).stem
    if output_folder is not None:
        return str(Path(output_folder) / "logs" / f"{stem}.log")
    if not run_key:
        raise ValueError("run_key is required when output_folder is not set")
    # Legacy layout: {logs_root}/{run_key}/{stem}.log
    root = logs_root if logs_root is not None else get_judge_logs_root()
    return str(Path(root) / run_key / f"{stem}.log")


def default_adhoc_parent() -> str:
    """
    Parent directory for single-file judge runs and unscoped LLMJudge log dirs.

    Set VERA_ADHOC_PARENT to override (tests use a temp directory).
    """
    return os.environ.get("VERA_ADHOC_PARENT") or os.path.join("output", "adhoc")


def add_persona_columns_to_dataframe(
    df: pd.DataFrame, personas_tsv_path: Path, columns: Sequence[str]
) -> pd.DataFrame:
    """
    Annotate evaluation rows with persona columns, joined on the persona name.

    The persona name is recovered from each row's conversation filename, the
    same join `add_risk_levels_to_dataframe` performs, but the columns carried
    over are named by the caller (ultimately by the target manifest) instead of
    fixed to SI's risk level.

    Annotation columns are placed directly after `persona_name` so the
    identifying fields stay together at the left edge of `results.csv`.

    Args:
        df: DataFrame with a 'filename' column
        personas_tsv_path: Path to the personas TSV
        columns: Persona column names to add, in order

    Returns:
        DataFrame with `persona_name` and the requested columns added. Existing
        columns are left untouched, so re-annotating is a no-op.
    """
    if not columns:
        return df

    persona_names = df["filename"].apply(
        lambda filename: extract_persona_name_from_filename(str(filename)) or "Unknown"
    )

    if "persona_name" not in df.columns:
        df = df.copy()
        df["persona_name"] = persona_names
        cols: list[str] = list(df.columns)
        cols.remove("persona_name")
        cols.insert(2 if "run_id" in cols else 1, "persona_name")
        df = pd.DataFrame(df[cols])

    # Offset grows per inserted column so several annotations keep the order the
    # caller asked for; inserting each one directly after `persona_name` would
    # reverse them.
    for offset, column in enumerate(columns, start=1):
        if column in df.columns:
            continue
        value_map = load_persona_column(personas_tsv_path, column)
        df = df.copy()
        df[column] = persona_names.map(lambda name: value_map.get(name, "Unknown"))
        cols = list(df.columns)
        cols.remove(column)
        cols.insert(cols.index("persona_name") + offset, column)
        df = pd.DataFrame(df[cols])

    return df
