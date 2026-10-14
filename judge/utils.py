"""Utility functions for the judge module."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from utils.logging_utils import (
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


def extract_model_names_from_path(path_input: str) -> Dict[str, str]:
    """
    Extract all three model names from the evaluation directory path.

    Directory format:
        j_{judge}__p_{persona}__a_{agent}__t{max_turns}__r{runs}__{timestamp}

    Args:
        path_input: Path to results.csv file or evaluation directory

    Returns:
        Dictionary with 'judge', 'persona', and 'agent' keys, or 'Unknown' if not found
    """
    path = Path(path_input)

    # If it's a file (e.g., results.csv), get the parent directory name
    # If it's a directory, use the directory name directly
    if path.is_file():
        dir_name = path.parent.name
    else:
        dir_name = path.name

    result = {"judge": "Unknown", "persona": "Unknown", "agent": "Unknown"}

    # Extract judge model: j_{judge}__p_...
    # Note: judge name may include a timestamp like _20251112_171754_380
    if dir_name.startswith("j_"):
        parts = dir_name.split("__p_", 1)
        if len(parts) > 0:
            judge_part = parts[0][2:]  # Remove 'j_' prefix
            # Remove timestamp pattern (YYYYMMDD_HHMMSS_milliseconds or similar)
            # Look for pattern starting with underscore followed by 8 digits (date)
            # Remove timestamp: _YYYYMMDD_HHMMSS_... or _YYYYMMDD_...
            judge_part = re.sub(r"_\d{8}.*$", "", judge_part)
            result["judge"] = judge_part.replace("_", " ").strip()

    # Extract persona model: ...__p_{persona}__a_...
    if "__p_" in dir_name:
        parts = dir_name.split("__p_")
        if len(parts) > 1:
            persona_part = parts[1].split("__a_")[0]
            result["persona"] = persona_part.replace("_", " ").strip()

    # Extract agent model: ...__a_{agent}__t...
    if "__a_" in dir_name:
        parts = dir_name.split("__a_")
        if len(parts) > 1:
            agent_part = parts[1].split("__t")[0]
            result["agent"] = agent_part.replace("_", " ").strip()

    return result


def default_adhoc_parent() -> str:
    """
    Parent directory for single-file judge runs and unscoped LLMJudge log dirs.

    Set VERA_ADHOC_PARENT to override (tests use a temp directory).
    """
    return os.environ.get("VERA_ADHOC_PARENT") or os.path.join("output", "adhoc")
