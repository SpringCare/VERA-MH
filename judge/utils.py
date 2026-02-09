"""Utility functions for the judge module."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


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


def load_rubric_structure(
    rubric_path: str, sep: str = "\t"
) -> Tuple[List[str], List[str]]:
    """
    Load DIMENSIONS and OPTIONS from the rubric file.

    Args:
        rubric_path: Path to the rubric file
        sep: Separator for the file (default: tab)

    Returns:
        Tuple of (dimensions, options):
        - dimensions: List of unique dimension names from the Dimension column
        - options: List of scoring option column names (empty for question-flow rubrics)
    """
    rubric_df = pd.read_csv(rubric_path, sep=sep)

    # Get unique dimensions from the Dimension column
    dimensions = [
        d.strip()
        for d in rubric_df["Dimension"].dropna().unique()
        if d and str(d).strip() != "nan"
    ]

    # Get options from columns (exclude metadata columns)
    columns = [col.strip() for col in rubric_df.columns]
    # Question-flow rubric columns: Question ID, Dimension, Risk Type,
    # Question, Examples, Severity, Answer, GOTO
    metadata_columns = {
        "Question ID",
        "Dimension",
        "Risk Type",
        "Question",
        "Examples",
        "Severity",
        "Answer",
        "GOTO",
    }
    options = [col for col in columns if col not in metadata_columns]

    return dimensions, options


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


def extract_persona_name_from_filename(filename: str) -> Optional[str]:
    """
    Extract persona name from TSV filename.

    Filename format: {hash}_{persona_name}_{model}_run{number}.tsv
    Example: 1a84d1_Brian_g4o_run3.tsv -> "Brian"

    Works with any file extension (.tsv, .txt, etc.) since it splits on "_".

    Args:
        filename: TSV filename (with or without extension)

    Returns:
        Persona name or None if not found
    """
    # Format: {hash}_{persona}_{model}_run{number} or
    # {hash}_{persona}_{model}_run{number}.tsv
    # Extension doesn't matter since we split on "_"
    try:
        parts = filename.split("_")
        if len(parts) >= 2:
            # Return just the persona name (index 1), not the list
            return parts[1]
        return None
    except Exception as e:
        print(f"Error extracting persona name from filename {filename}: {e}")
        return None
