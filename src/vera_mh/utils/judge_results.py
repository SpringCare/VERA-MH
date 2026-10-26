"""Reading what a judge run wrote to disk.

A judge run leaves one TSV per evaluation. These helpers parse those
filenames, rebuild the run's ``results.csv`` from the TSVs, and read the
persona columns that get joined onto it. Both `judge/` (which writes
``results.csv``) and `score/` (which rebuilds it when missing) use them, so
they live here rather than in either package.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


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


# Load dimensions from rubric file (repo-root relative so it works from any cwd)
_REPO_ROOT = Path(__file__).resolve().parents[3]
DIMENSIONS, _ = load_rubric_structure(str(_REPO_ROOT / "data" / "SI" / "rubric.tsv"))


def parse_evaluation_filename(filename: str) -> Dict[str, Union[str, int]]:
    """
    Parse an evaluation filename to extract components.

    Basenames come from `judge.utils.judge_evaluation_tsv_filename` (see LLMJudge).

    Format: {id}_{persona}_{user_model}_{run}_{judge_model}_{iteration}.tsv
    Example: 000682_Alix_gemini-3-pro-preview_run22_gpt-4o_i1.tsv

    Args:
        filename: The evaluation filename

    Returns:
        Dict with parsed components: id, persona, user_model, run,
        judge_model, judge_iteration.
        Returns empty dict if filename doesn't match expected pattern.
    """
    # Remove .tsv extension
    name = filename.replace(".tsv", "")

    # Pattern: id_persona_usermodel_runN_judgemodel_iN
    # The user model can contain hyphens, so we need to be careful
    match = re.match(
        r"^([a-f0-9]+)_([A-Za-z]+)_(.+)_run(\d+)_([^_]+)_i(\d+)$",
        name,
    )

    if not match:
        return {}

    return {
        "id": match.group(1),
        "persona": match.group(2),
        "user_model": match.group(3),
        "run": int(match.group(4)),
        "judge_model": match.group(5),
        "judge_iteration": int(match.group(6)),
    }


def parse_judge_metadata_from_evaluation_tsv_filename(filename: str) -> Tuple[str, int]:
    """
    Extract ``judge_model`` and ``judge_instance`` (1-based) from a TSV basename.

    Uses :func:`parse_evaluation_filename` for structured production names
    (``{hex}_{persona}_{user}_run{N}_{judge}_i{instance}``).

    If that fails, applies the same batch layout as
    :func:`judge.utils.judge_evaluation_tsv_filename` with ``judge_instance`` set:
    ``{conversation_stem}_{normalized_judge}_i{instance}``, where ``normalized_judge``
    has no ``_`` (only ``/`` and ``:`` are replaced in the model id). Matches the
    first fallback in :func:`extract_conversation_filename_from_tsv`.

    Returns:
        ``("", 0)`` when the filename cannot be interpreted.

    Note:
        Judge model strings that contain underscores after normalization cannot be
        recovered from the basename alone; use structured filenames for those runs.
    """
    parsed = parse_evaluation_filename(filename)
    if parsed:
        return str(parsed["judge_model"]), int(parsed["judge_iteration"])

    name = filename.replace(".tsv", "")
    match = re.search(r"(.+?)_([^_]+)_i(\d+)$", name)
    if match:
        return match.group(2), int(match.group(3))
    return "", 0


def extract_conversation_filename_from_tsv(tsv_filename: str) -> str:
    """
    Extract original conversation filename from TSV evaluation filename.

    TSV format: {conversation_name}_{judge_model}_i{instance}.tsv
    Target: {conversation_name}.txt

    Args:
        tsv_filename: TSV filename (e.g., "3ea338_Lena_g5_run4_uuid_gpt-4o_i1.tsv")

    Returns:
        Original conversation filename (e.g., "3ea338_Lena_g5_run4_uuid.txt")
    """
    # Remove .tsv extension
    name = tsv_filename.replace(".tsv", "")

    # Try to parse using the structured format first
    parsed = parse_evaluation_filename(tsv_filename)
    if parsed:
        # Reconstruct: {id}_{persona}_{user_model}_run{number}
        return (
            f"{parsed['id']}_{parsed['persona']}_{parsed['user_model']}_"
            f"run{parsed['run']}.txt"
        )

    # Fallback: Remove judge suffix pattern _{judge_model}_i{instance}
    match = re.search(r"(.+?)_[^_]+_i\d+$", name)
    if match:
        # Found pattern, return the part before the judge suffix
        return match.group(1) + ".txt"

    # Alternative: match trailing _i{digits} (judge model may lack underscores)
    # This handles cases like "name_gpt-4o_i1"
    match = re.search(r"(.+)_i\d+$", name)
    if match:
        # Require '_' before _i{digits} to avoid false matches
        before_i = match.group(1)
        if "_" in before_i:
            match2 = re.search(r"(.+)_[^-_]+_i\d+$", name)
            if match2:
                return match2.group(1) + ".txt"

    # If no match, just convert extension (preserve original behavior)
    return name + ".txt"


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


def build_results_csv_from_tsv_files(evaluations_dir) -> pd.DataFrame:
    """
    Build a results DataFrame from TSV evaluation files in a directory.

    This function reads all .tsv files in the given directory and combines
    them into a single DataFrame suitable for VERA score calculation.
    Useful when results.csv is missing or has empty dimension columns.

    Args:
        evaluations_dir: Path to directory containing TSV evaluation files
            (can be str or Path)

    Returns:
        DataFrame with columns: filename, run_id, judge_model, judge_instance,
        judge_id, each dimension, and for each dimension: {dimension}_yes_question_id
        and {dimension}_yes_reasoning

    Raises:
        FileNotFoundError: If no TSV files are found in the directory
    """
    from pathlib import Path

    evaluations_dir = Path(evaluations_dir)
    results = []

    # Get run_id from directory name (format: j_...__run_id)
    run_id = (
        evaluations_dir.name.split("__")[-1]
        if "__" in evaluations_dir.name
        else evaluations_dir.name
    )

    # Find all TSV files in the directory
    tsv_files = list(evaluations_dir.glob("*.tsv"))

    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in: {evaluations_dir}")

    for tsv_file in tsv_files:
        filename = tsv_file.name
        # Read TSV file
        try:
            tsv_df = pd.read_csv(tsv_file, sep="\t")

            # Parse TSV filename to extract judge_model, judge_instance, and judge_id
            judge_model, judge_instance = (
                parse_judge_metadata_from_evaluation_tsv_filename(filename)
            )
            judge_id = max(0, judge_instance - 1)  # judge_id is 0-based

            # Build row dictionary
            row = {
                "filename": filename,
                "run_id": run_id,
                "judge_model": judge_model,
                "judge_instance": judge_instance,
                "judge_id": judge_id,
            }

            # Dimension scores; parse yes_question_id / yes_reasoning from Reasoning
            for _, tsv_row in tsv_df.iterrows():
                dimension = str(tsv_row.get("Dimension", "")).strip()
                score = str(tsv_row.get("Score", "")).strip()
                reasoning = str(tsv_row.get("Reasoning", "")).strip()

                # Parse yes_question_id and yes_reasoning from reasoning column
                # Format: "...Q{question_id}: {reasoning}..."
                # See _add_severity_reason in judge/llm_judge.py for more details
                # Extract the first occurrence of Q{id}: {reasoning} pattern
                yes_question_id = ""
                yes_reasoning = ""

                # Question IDs are opaque strings (for example, "1a"), not
                # necessarily integers. They cannot contain whitespace, a
                # semicolon, parentheses, or the colon that terminates the ID
                # in the serialized reasoning format.
                question_id_pattern = r"[^\s:;()]+"
                match = re.search(
                    rf"Q({question_id_pattern})(?: \(ASSIGN_END\))?:\s*"
                    rf"(.+?)(?=;\s*Q{question_id_pattern}"
                    rf"(?: \(ASSIGN_END\))?:|$)",
                    reasoning,
                    re.DOTALL,
                )
                if match:
                    yes_question_id = match.group(1)
                    yes_reasoning = match.group(
                        2
                    ).strip()  # Everything after ": " until next "Q{id}:" or end

                if dimension in DIMENSIONS:
                    row[dimension] = score
                    # Always add yes_* columns (even if empty)
                    row[f"{dimension}_yes_question_id"] = yes_question_id
                    row[f"{dimension}_yes_reasoning"] = yes_reasoning

            # Fill missing dimensions; ensure yes_* columns exist per dimension
            for dimension in DIMENSIONS:
                if dimension not in row:
                    row[dimension] = ""
                # Always add yes_* columns (even if empty)
                if f"{dimension}_yes_question_id" not in row:
                    row[f"{dimension}_yes_question_id"] = ""
                if f"{dimension}_yes_reasoning" not in row:
                    row[f"{dimension}_yes_reasoning"] = ""

            results.append(row)

        except Exception as e:
            print(f"Warning: Error reading TSV file {tsv_file}: {e}")
            continue

    # Build dataframe with correct column order
    # Include judge columns, dimension scores, and yes_question_id/yes_reasoning columns
    columns = [
        "filename",
        "run_id",
        "judge_model",
        "judge_instance",
        "judge_id",
    ]
    for dimension in DIMENSIONS:
        columns.append(dimension)
        columns.append(f"{dimension}_yes_question_id")
        columns.append(f"{dimension}_yes_reasoning")

    df = pd.DataFrame(results, columns=columns)

    return df


def build_dataframe_from_tsv_files(evaluations_dir: Path) -> pd.DataFrame:
    """
    Build a dataframe from TSV evaluation files.

    Args:
        evaluations_dir: Directory containing TSV evaluation files

    Returns:
        DataFrame with columns: filename, run_id, judge_model, judge_instance,
        judge_id, each dimension, and for each dimension: {dimension}_yes_question_id
        and {dimension}_yes_reasoning
    """
    # Use build_results_csv_from_tsv_files to build the dataframe
    df = build_results_csv_from_tsv_files(evaluations_dir)

    # Filename: .tsv -> .txt; strip judge suffix _{judge_model}_i{instance}
    df["filename"] = df["filename"].apply(extract_conversation_filename_from_tsv)

    return df


def load_persona_column(personas_tsv_path: Path, column: str) -> Dict[str, str]:
    """
    Load a mapping of persona name to one annotation column's value.

    The generic form behind `load_personas_risk_levels`. Targets do not share a
    persona schema: SI carries `Short Current Suicide Risk Level` while HFO
    carries `Active Life Threat`, so the column to join on is a parameter rather
    than a constant.

    Args:
        personas_tsv_path: Path to the personas TSV
        column: Persona column to read alongside `Name`

    Returns:
        Dictionary mapping persona name to that column's value

    Raises:
        KeyError: If the personas file has no such column. Returning an empty
            map instead would annotate every row "Unknown", which reads like a
            result rather than a misconfiguration.
    """
    # keep_default_na=False so a literal "None" stays the string "None" rather
    # than becoming NaN -- a meaningful value for risk-level style columns.
    df = pd.read_csv(personas_tsv_path, sep="\t", keep_default_na=False)
    if "Name" not in df.columns:
        raise KeyError(f"personas file {personas_tsv_path} has no 'Name' column")
    if column not in df.columns:
        raise KeyError(
            f"personas file {personas_tsv_path} has no {column!r} column; "
            f"available columns: {', '.join(map(str, df.columns))}"
        )
    return (
        df[["Name", column]].set_index("Name")[column].astype(str).str.strip().to_dict()
    )
