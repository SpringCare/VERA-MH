#!/usr/bin/env python3
"""Concatenate results.csv files and extract metadata from file paths.

This script reads all results.csv files from specified evaluation directories,
concatenates them, and extracts additional columns from the file/directory names:
- provider LLM (from a_{agent_model} in path)
- user LLM (from p_{persona_model} in path)
- persona name (from filename column in CSV)
- clean judge LLM (from j_{judge_model} in path, removing timestamp)
- max_turns (from t{turns} in path)

Usage:
    python3 scripts/concatenate_results_with_metadata.py [output_file]
    python3 scripts/concatenate_results_with_metadata.py -e evaluations/A -e evaluations/B \\
        -c conversations/C -o out.csv
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd


def extract_metadata_from_path(csv_path: Path) -> dict[str, Optional[str | int]]:
    """Extract metadata from the results.csv file path.

    Path format:
        evaluations/{base_dir}/p_{persona_model}__a_{agent_model}__t{turns}__r{runs}__{timestamp}/j_{judge_model}_{timestamp}__p_{persona_model}__a_{agent_model}__t{turns}__r{runs}__{timestamp}/results.csv

    Args:
        csv_path: Path to results.csv file

    Returns:
        Dictionary with extracted metadata:
        - provider_llm: Agent model name (e.g., "gemini_3_1")
        - user_llm: Persona model name (e.g., "opus_4_5")
        - max_turns: Maximum number of turns parameter (e.g., 100)
    """
    result: dict[str, Optional[str | int]] = {
        "provider_llm": None,
        "user_llm": None,
        "max_turns": None,
    }

    # Get the parent directory name (the j_* directory)
    dir_name = csv_path.parent.name

    # Extract persona model: ...__p_{persona}__a_...
    if "__p_" in dir_name:
        parts = dir_name.split("__p_")
        if len(parts) > 1:
            persona_part = parts[1].split("__a_")[0]
            result["user_llm"] = persona_part.replace("_", " ").strip()

    # Extract agent model: ...__a_{agent}__t...
    if "__a_" in dir_name:
        parts = dir_name.split("__a_")
        if len(parts) > 1:
            agent_part = parts[1].split("__t")[0]
            result["provider_llm"] = agent_part.replace("_", " ").strip()

    # Extract turns: ...__t{turns}__...
    if "__t" in dir_name:
        # Find pattern __t{number}__
        match = re.search(r"__t(\d+)__", dir_name)
        if match:
            try:
                result["max_turns"] = int(match.group(1))
            except ValueError:
                pass

    return result


def extract_persona_name_from_filename(filename: str) -> Optional[str]:
    """Extract persona name from conversation filename.

    Filename format: {hash}_{persona_name}_{model}_run{number}.txt
    Example: c4eb8e_Zoe_claude-opus-4-5-20251101_run1.txt -> "Zoe"

    Args:
        filename: Conversation filename (with or without extension)

    Returns:
        Persona name or None if not found
    """
    try:
        parts = filename.split("_")
        if len(parts) >= 2:
            return parts[1]
        return None
    except Exception:
        return None


def count_conversation_turns(conversation_path: Path) -> Optional[int]:
    """Count the number of turns in a conversation file.

    Conversation files have lines starting with "user:" or "chatbot:".
    Each such line represents one turn.

    Args:
        conversation_path: Path to conversation .txt file

    Returns:
        Number of turns, or None if file cannot be read
    """
    try:
        if not conversation_path.exists():
            return None
        content = conversation_path.read_text(encoding="utf-8", errors="replace")
        # Count lines that start with "user:" or "chatbot:"
        turn_count = 0
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("user:") or line.startswith("chatbot:"):
                turn_count += 1
        return turn_count if turn_count > 0 else None
    except Exception:
        return None


def find_conversation_file(
    filename: str, run_id: str, base_dirs: list[Path]
) -> Optional[Path]:
    """Find the conversation file path from filename and run_id.

    Args:
        filename: Conversation filename
            (e.g., "c4eb8e_Zoe_claude-opus-4-5-20251101_run1.txt")
        run_id: Run ID directory name
            (e.g., "p_opus_4_5__a_gemini_3_1__t100__r1__20260223_213637")
        base_dirs: List of base directories to search
            (e.g., ["conversations/HEOR_FIXED"])

    Returns:
        Path to conversation file, or None if not found
    """
    # Try to find the conversation file in the conversations directory
    # The run_id should match a directory name
    for base_dir in base_dirs:
        # Try direct path: conversations/{base}/p_*/{filename}
        # Also try: conversations/{base}/{run_id}/{filename}
        possible_paths = [
            base_dir / run_id / filename,
            base_dir / filename,
        ]
        # Also search recursively for the filename
        for possible_path in possible_paths:
            if possible_path.exists() and possible_path.is_file():
                return possible_path

        # Search recursively in base_dir for the filename
        for found_path in base_dir.rglob(filename):
            if found_path.is_file():
                return found_path

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate results.csv files and extract metadata from paths"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="concatenated_results.csv",
        help="Output CSV file path (default: concatenated_results.csv)",
    )
    parser.add_argument(
        "-e",
        "--eval-dir",
        action="append",
        dest="eval_dirs",
        metavar="DIR",
        help=(
            "Base directory to search for results.csv (repeatable). "
            "Default: evaluations/HEOR_GPT4o_FIXED and evaluations/HEOR_Sonnet45_FIXED"
        ),
    )
    parser.add_argument(
        "-c",
        "--conv-dir",
        action="append",
        dest="conv_dirs",
        metavar="DIR",
        help=(
            "Base directory to search for conversation .txt files (repeatable). "
            "Default: conversations/HEOR_FIXED"
        ),
    )
    args = parser.parse_args()

    eval_base_dirs = (
        [Path(p) for p in args.eval_dirs]
        if args.eval_dirs
        else [
            Path("evaluations/HEOR_GPT4o_FIXED"),
            Path("evaluations/HEOR_Sonnet45_FIXED"),
        ]
    )
    conv_base_dirs = (
        [Path(p) for p in args.conv_dirs]
        if args.conv_dirs
        else [Path("conversations/HEOR_FIXED")]
    )

    # Find all results.csv files
    all_csv_files = []
    for base_dir in eval_base_dirs:
        if not base_dir.exists():
            print(
                f"Warning: Directory {base_dir} does not exist, skipping",
                file=sys.stderr,
            )
            continue
        csv_files = list(base_dir.rglob("results.csv"))
        all_csv_files.extend(csv_files)
        print(f"Found {len(csv_files)} results.csv files in {base_dir}")

    if not all_csv_files:
        print("Error: No results.csv files found", file=sys.stderr)
        sys.exit(1)

    print(f"Total: {len(all_csv_files)} results.csv files found")

    # Process each CSV file
    all_dataframes = []
    file_row_counts = []  # Track (file_path, row_count) for verification
    total_expected_rows = 0

    for csv_path in all_csv_files:
        try:
            # Read the CSV
            df = pd.read_csv(csv_path)
            initial_row_count = len(df)
            total_expected_rows += initial_row_count

            if df.empty:
                print(f"Warning: {csv_path} is empty, skipping", file=sys.stderr)
                file_row_counts.append((str(csv_path), 0))
                continue

            # Extract metadata from path
            metadata = extract_metadata_from_path(csv_path)

            # Add metadata columns
            df["provider_llm"] = metadata["provider_llm"]
            df["user_llm"] = metadata["user_llm"]
            df["max_turns"] = metadata["max_turns"]

            # Extract persona name from filename column if it exists
            if "filename" in df.columns:
                df["persona_name"] = df["filename"].apply(
                    extract_persona_name_from_filename
                )
            else:
                df["persona_name"] = None

            # Count actual conversation turns for each row
            if "filename" in df.columns and "run_id" in df.columns:

                def count_turns_for_row(row):
                    filename = row.get("filename", "")
                    run_id = row.get("run_id", "")
                    if not filename or not run_id:
                        return None
                    conv_path = find_conversation_file(filename, run_id, conv_base_dirs)
                    if conv_path:
                        return count_conversation_turns(conv_path)
                    return None

                df["actual_conversation_turns"] = df.apply(count_turns_for_row, axis=1)
            else:
                df["actual_conversation_turns"] = None

            final_row_count = len(df)
            file_row_counts.append((str(csv_path), final_row_count))

            if initial_row_count != final_row_count:
                print(
                    f"Warning: Row count changed for {csv_path}: "
                    f"{initial_row_count} -> {final_row_count}",
                    file=sys.stderr,
                )

            all_dataframes.append(df)

        except Exception as e:
            print(f"Error processing {csv_path}: {e}", file=sys.stderr)
            file_row_counts.append((str(csv_path), 0))
            continue

    if not all_dataframes:
        print("Error: No valid data to concatenate", file=sys.stderr)
        sys.exit(1)

    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Reorder columns to put metadata columns first (after existing key columns)
    # Keep existing columns in their original order, then add new ones
    metadata_cols = [
        "provider_llm",
        "user_llm",
        "persona_name",
        "max_turns",
        "actual_conversation_turns",
    ]
    existing_cols = [col for col in combined_df.columns if col not in metadata_cols]
    new_cols = metadata_cols

    # Put new columns after filename/run_id/judge columns if they exist
    key_cols = ["filename", "run_id", "judge_model", "judge_instance", "judge_id"]
    ordered_cols = []
    for col in key_cols:
        if col in existing_cols:
            ordered_cols.append(col)
            existing_cols.remove(col)

    # Add new metadata columns
    ordered_cols.extend(new_cols)

    # Add remaining existing columns
    ordered_cols.extend([col for col in existing_cols if col not in ordered_cols])

    combined_df = combined_df[ordered_cols]

    # Write output
    output_path = Path(args.output_file)
    combined_df.to_csv(output_path, index=False)

    # Verification summary
    actual_total_rows = len(combined_df)
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Files processed: {len(all_dataframes)}")
    print(f"  Expected total rows: {total_expected_rows}")
    print(f"  Actual total rows: {actual_total_rows}")
    print(f"  Difference: {total_expected_rows - actual_total_rows}")
    print("=" * 60)

    if total_expected_rows != actual_total_rows:
        print("\n⚠️  WARNING: Row count mismatch!")
        print(f"   Missing {total_expected_rows - actual_total_rows} rows")
        print("\nDetailed row counts per file:")
        for file_path, row_count in file_row_counts:
            print(f"  {row_count:5d} rows: {file_path}")
    else:
        print("\n✅ Row counts match!")
        # Show summary of row counts per file (grouped by directory)
        print("\nRow counts by directory:")
        dir_counts = defaultdict(int)
        for file_path, row_count in file_row_counts:
            # Extract directory name (parent of results.csv)
            path_parts = Path(file_path).parts
            if len(path_parts) >= 2:
                # Get the evaluation directory (e.g., j_*__p_*__a_*)
                dir_name = path_parts[-2]
                dir_counts[dir_name] += row_count
        for dir_name, count in sorted(dir_counts.items()):
            print(f"  {count:5d} rows: {dir_name}")

    print(f"\n✅ Concatenated {len(all_dataframes)} CSV files")
    print(f"✅ Total rows: {actual_total_rows}")
    print(f"✅ Output written to: {output_path}")


if __name__ == "__main__":
    main()
