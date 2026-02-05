"""Integration test comparing judge.py output to expected clinician ratings.

This test:
1. Runs judge.py on conversations in tests/fixtures/conversations
2. Extracts ratings from the judge output
3. Compares them to expected values from transcript_agreement_scores.csv
"""

import ast
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from judge.constants import (
    BEST_PRACTICE,
    DAMAGING,
    NEUTRAL,
    NOT_RELEVANT,
)

# Map clinician dimension names to judge dimension names
DIMENSION_MAPPING = {
    "clinician_detects_risk": "Detects Risk",
    "clinician_probes_risk": "Clarifies Risk",
    "clinician_resources": "Guides to Human Support",
    "clinician_collaborates": "Collaborates and Validates Appropriately",
    "clinician_boundaries": "Maintains Safe Boundaries",
}


def extract_transcript_id(filename: str) -> str:
    """Extract transcript_id from conversation filename.

    Example: "afaec2_Omar_g5_run1.txt" -> "afaec2"
    """
    return filename.split("_")[0]


def extract_expected_ratings(csv_path: Path) -> pd.DataFrame:
    """Extract expected ratings from transcript_agreement_scores.csv.

    Args:
        csv_path: Path to the transcript_agreement_scores.csv file

    Returns:
        DataFrame with columns: transcript_id and all judge dimension columns.
        Each row represents expected ratings for one transcript.

    Raises:
        pytest.skip: If ratings cannot be parsed for a transcript
    """
    expected_df = pd.read_csv(csv_path)
    expected_rows = []

    for _, row in expected_df.iterrows():
        transcript_id = row["transcript_id"]
        # Parse the unique_values_per_dim string to get the expected ratings
        unique_vals_str = str(row["unique_values_per_dim"])
        try:
            unique_vals_dict = ast.literal_eval(unique_vals_str)
            # Map to judge dimension names and get the most common value
            # (when there are multiple unique values, take the first one)
            ratings = {"transcript_id": transcript_id}
            for clinician_dim, values in unique_vals_dict.items():
                if clinician_dim in DIMENSION_MAPPING:
                    judge_dim = DIMENSION_MAPPING[clinician_dim]
                    # Take the first value (most common when there's agreement,
                    # or first when there are multiple unique values)
                    if values:
                        ratings[judge_dim] = values[0]
            expected_rows.append(ratings)
        except (ValueError, SyntaxError) as e:
            pytest.skip(f"Could not parse expected ratings for {transcript_id}: {e}")

    # Create DataFrame with transcript_id and all dimension columns
    expected_ratings_df = pd.DataFrame(expected_rows)

    # Validate that all required dimensions are present
    required_dimensions = list(DIMENSION_MAPPING.values())
    missing_dimensions = [
        dim for dim in required_dimensions if dim not in expected_ratings_df.columns
    ]
    if missing_dimensions:
        raise ValueError(
            f"Missing required dimensions in expected ratings: {missing_dimensions}. "
            f"Found columns: {list(expected_ratings_df.columns)}"
        )

    return expected_ratings_df


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def conversations_dir(fixtures_dir: Path) -> Path:
    """Path to conversations fixtures directory."""
    return fixtures_dir / "conversations"


@pytest.fixture
def expected_ratings_csv(conversations_dir: Path) -> Path:
    """Path to expected ratings CSV file."""
    csv_path = conversations_dir / "transcript_agreement_scores.csv"
    if not csv_path.exists():
        pytest.skip(f"Expected ratings CSV not found: {csv_path}")
    return csv_path


@pytest.mark.integration
class TestJudgeAgainstClinicianRatings:
    """Test judge.py output against expected clinician ratings."""

    def test_judge_scores_match_expected_ratings(
        self,
        conversations_dir: Path,
        expected_ratings_csv: Path,
        tmp_path: Path,
    ):
        """Test that judge ratings match expected clinician ratings.

        This test:
        1. Runs judge.py on conversations in fixtures
        2. Reads TSV output files
        3. Compares ratings to expected ratings from transcript_agreement_scores.csv

        Note: Warns for transcripts without 100% clinician agreement
        (exact_match_pct != 100%) but tests all available transcripts.
        """
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping real LLM test")

        # Extract expected ratings from CSV
        print(f"\nLoading expected ratings from {expected_ratings_csv.name}...")
        expected_ratings = extract_expected_ratings(expected_ratings_csv)
        print(f"Loaded {len(expected_ratings)} expected transcript ratings")

        # Run judge.py as subprocess
        print(f"\nRunning judge.py on conversations in {conversations_dir}...")
        project_root = Path(__file__).parent.parent.parent
        # For debugging: output to fixtures/conversations folder
        output_dir = conversations_dir / "evaluations"
        # For production: use temp folder (uncomment when done debugging)
        # output_dir = tmp_path / "evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "judge.py",
            "-f",
            str(conversations_dir),
            "-j",
            # "gpt-4o",
            "claude-sonnet-4-5-20250929",
            "-o",
            str(output_dir),
        ]

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        print("judge.py completed")

        if result.returncode != 0:
            pytest.fail(
                f"judge.py failed with return code {result.returncode}.\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Find the output folder (judge.py creates a timestamped subdirectory)
        # Look for directories matching the pattern j_*__conversations
        # Use the most recently created folder
        output_folders = [
            d
            for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("j_") and "__conversations" in d.name
        ]

        if not output_folders:
            pytest.fail(f"No output folder found in {output_dir}")

        # Use the most recently created folder
        output_folder = max(output_folders, key=lambda p: p.stat().st_mtime)

        try:
            # Read from results.csv (created by judge.py)
            # This tests that judge.py correctly creates the CSV with dimension data
            results_csv = output_folder / "results.csv"
            if not results_csv.exists():
                pytest.fail(
                    f"results.csv not found in {output_folder}. "
                    f"judge.py should create this file with evaluation results."
                )

            print(f"\nReading results from {results_csv}...")
            results_df = pd.read_csv(results_csv)
            print(f"Found {len(results_df)} conversation results")
            print(f"Columns in results.csv: {list(results_df.columns)}")

            # Verify number of TSV files matches number of rows in results.csv
            tsv_files = list(output_folder.glob("*.tsv"))
            num_tsv_files = len(tsv_files)
            num_csv_rows = len(results_df)
            print(
                f"Found {num_tsv_files} TSV files, {num_csv_rows} rows in results.csv"
            )
            if num_tsv_files != num_csv_rows:
                pytest.fail(
                    f"Mismatch: {num_tsv_files} TSV files found but "
                    f"{num_csv_rows} rows in results.csv. Expected them to match."
                )

            # Add transcript_id column
            results_df["transcript_id"] = results_df["filename"].apply(
                lambda f: extract_transcript_id(str(f))
            )

            # Verify results.csv has the required columns
            required_columns = ["filename"] + list(DIMENSION_MAPPING.values())
            missing_columns = [
                col for col in required_columns if col not in results_df.columns
            ]
            if missing_columns:
                pytest.fail(
                    f"results.csv is missing required columns: {missing_columns}.\n"
                    f"Found columns: {list(results_df.columns)}\n"
                    f"Expected dimensions: {list(DIMENSION_MAPPING.values())}"
                )

            # Verify dimension columns have data (not all empty)
            empty_dimensions = []
            for dim in DIMENSION_MAPPING.values():
                if dim in results_df.columns:
                    col_values = results_df[dim].fillna("").astype(str).str.strip()
                    if not (col_values != "").any():
                        empty_dimensions.append(dim)

            if empty_dimensions:
                pytest.fail(
                    f"results.csv has empty dimension columns: {empty_dimensions}. "
                    f"judge.py should populate these with evaluation scores."
                )

            # Validate that all dimension values are in the expected set
            valid_values = {BEST_PRACTICE, NEUTRAL, DAMAGING, NOT_RELEVANT}
            invalid_values = []
            for dim in DIMENSION_MAPPING.values():
                if dim in results_df.columns:
                    dim_values = results_df[dim].dropna().astype(str).str.strip()
                    invalid = dim_values[
                        ~dim_values.isin(valid_values) & (dim_values != "")
                    ]
                    if not invalid.empty:
                        for idx, val in invalid.items():
                            filename = results_df.loc[idx, "filename"]
                            invalid_values.append(
                                f"{filename} - {dim}: '{val}' (not in {valid_values})"
                            )

            if invalid_values:
                pytest.fail(
                    "results.csv contains invalid dimension values:\n"
                    + "\n".join(f"  - {v}" for v in invalid_values)
                )

            # Merge expected and actual ratings on transcript_id
            merged_df = results_df.merge(
                expected_ratings,
                on="transcript_id",
                how="outer",
                suffixes=("_actual", "_expected"),
            )

            # Check for missing transcripts in results
            missing_transcripts = merged_df[merged_df["filename"].isna()][
                "transcript_id"
            ]
            if len(missing_transcripts) > 0:
                print(
                    f"⚠ Warning: {len(missing_transcripts)} transcript(s) "
                    f"missing from results"
                )
            mismatches = [
                f"{tid}: No judge ratings found (expected ratings exist)"
                for tid in missing_transcripts
            ]

            # Compare dimension columns for transcripts present in both
            comparison_df = merged_df[
                merged_df["transcript_id"].notna() & merged_df["filename"].notna()
            ]
            print(f"\nComparing ratings for {len(comparison_df)} transcripts...")

            def is_empty(series: pd.Series) -> pd.Series:
                """Check if values are empty or invalid."""
                return (series == "") | (series.str.lower().isin(["nan", "none"]))

            for dimension in DIMENSION_MAPPING.values():
                print(f"  Checking {dimension}...", end="", flush=True)
                expected_col = f"{dimension}_expected"
                actual_col = f"{dimension}_actual"

                # Convert to strings and strip
                expected = comparison_df.loc[:, expected_col].astype(str).str.strip()
                actual = comparison_df.loc[:, actual_col].astype(str).str.strip()

                # Validate values are not missing or empty
                if is_empty(expected).any():
                    empty_ids = comparison_df[is_empty(expected)][
                        "transcript_id"
                    ].tolist()
                    raise ValueError(
                        f"Missing or empty expected ratings for {dimension} "
                        f"in transcripts: {empty_ids}"
                    )
                if is_empty(actual).any():
                    empty_ids = comparison_df[is_empty(actual)][
                        "transcript_id"
                    ].tolist()
                    raise ValueError(
                        f"Missing or empty actual ratings for {dimension} "
                        f"in transcripts: {empty_ids}"
                    )

                # Find mismatches
                mismatched_mask = expected != actual
                if mismatched_mask.any():
                    num_mismatches = mismatched_mask.sum()
                    mismatched_df = comparison_df.loc[mismatched_mask]
                    mismatches.extend(
                        f"{row['transcript_id']} - {dimension}: "
                        f"Expected '{expected.loc[idx]}', got '{actual.loc[idx]}'"
                        for idx, row in mismatched_df.iterrows()
                    )
                    print(f" {num_mismatches} mismatch(es)")
                else:
                    print(" ✓")

            print()  # Blank line before summary
            if mismatches:
                error_msg = "Judge ratings don't match expected clinician ratings:\n"
                error_msg += "\n".join(f"  - {m}" for m in mismatches)
                pytest.fail(error_msg)

            # If we get here, all ratings matched
            assert len(results_df) > 0, "No judge ratings were generated"
            print(
                f"✓ All ratings matched for {len(comparison_df)} transcripts "
                f"across {len(DIMENSION_MAPPING)} dimensions"
            )
        finally:
            # Clean up: delete the output folder
            # if output_folder.exists():
            #     shutil.rmtree(output_folder)
            pass
