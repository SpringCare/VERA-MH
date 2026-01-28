"""Integration test comparing judge.py output to expected clinician ratings.

This test:
1. Runs judge.py on conversations in tests/fixtures/conversations
2. Extracts scores from the judge output
3. Compares them to expected values from transcript_agreement_scores.csv
"""

import ast
import os
import subprocess
import warnings
from pathlib import Path

import pandas as pd
import pytest

# Map clinician dimension names to judge dimension names
DIMENSION_MAPPING = {
    "clinician_detects_risk": "Detects Risk",
    "clinician_probes_risk": "Probes Risk",
    "clinician_resources": "Guides to Human Support",
    "clinician_collaborates": "Collaborates and Validates Appropriately",
    "clinician_boundaries": "Maintains Safe Boundaries",
}


def extract_transcript_id(filename: str) -> str:
    """Extract transcript_id from conversation filename.

    Example: "afaec2_Omar_g5_run1.txt" -> "afaec2"
    """
    return filename.split("_")[0]


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
        """Test that judge scores match expected clinician ratings.

        This test:
        1. Runs judge.py on conversations in fixtures
        2. Reads TSV output files
        3. Compares scores to expected ratings from transcript_agreement_scores.csv

        Note: Warns for transcripts without 100% clinician agreement
        (exact_match_pct != 100%) but tests all available transcripts.
        """
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping real LLM test")

        # Load expected ratings
        expected_df = pd.read_csv(expected_ratings_csv)

        # Extract expected ratings from the dataframe
        expected_ratings = {}
        for _, row in expected_df.iterrows():
            transcript_id = row["transcript_id"]
            # Warn if transcript doesn't have 100% agreement
            if row["exact_match_pct"] != 100.0:
                agreement_pct = row["exact_match_pct"]
                warnings.warn(
                    f"Transcript {transcript_id} has {agreement_pct}% agreement "
                    f"(not 100%), may have ambiguous expected values",
                    UserWarning,
                )
            # Parse the unique_values_per_dim string to get the expected ratings
            unique_vals_str = str(row["unique_values_per_dim"])
            try:
                unique_vals_dict = ast.literal_eval(unique_vals_str)
                # Map to judge dimension names and get the first (only) value
                ratings = {}
                for clinician_dim, values in unique_vals_dict.items():
                    if clinician_dim in DIMENSION_MAPPING:
                        judge_dim = DIMENSION_MAPPING[clinician_dim]
                        # If all clinicians agreed (100% agreement),
                        # there's only one value
                        if len(values) == 1:
                            ratings[judge_dim] = values[0]
                expected_ratings[transcript_id] = ratings
            except (ValueError, SyntaxError) as e:
                pytest.skip(
                    f"Could not parse expected ratings for {transcript_id}: {e}"
                )

        # Run judge.py as subprocess
        project_root = Path(__file__).parent.parent.parent
        output_dir = tmp_path / "evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "judge.py",
            "-f",
            str(conversations_dir),
            "-j",
            "gpt-4o",
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

        if result.returncode != 0:
            pytest.fail(
                f"judge.py failed with return code {result.returncode}.\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Find the output folder (judge.py creates a timestamped subdirectory)
        # Look for directories matching the pattern j_*__conversations
        output_folders = [
            d
            for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("j_") and "__conversations" in d.name
        ]

        if not output_folders:
            pytest.fail(f"No output folder found in {output_dir}")

        # Use the most recently created folder
        output_folder = max(output_folders, key=lambda p: p.stat().st_mtime)

        # Read from results.csv (created by judge.py)
        # This tests that judge.py correctly creates the CSV with dimension data
        results_csv = output_folder / "results.csv"
        if not results_csv.exists():
            pytest.fail(
                f"results.csv not found in {output_folder}. "
                f"judge.py should create this file with evaluation results."
            )

        results_df = pd.read_csv(results_csv)

        # Verify results.csv has the required columns
        required_columns = ["filename"] + list(DIMENSION_MAPPING.values())
        missing_columns = [
            col for col in required_columns if col not in results_df.columns
        ]
        if missing_columns:
            pytest.fail(
                f"results.csv is missing required columns: {missing_columns}. "
                f"Found columns: {list(results_df.columns)}"
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

        # Extract scores from dataframe
        judge_scores = {}
        for _, row in results_df.iterrows():
            filename = str(row["filename"])
            transcript_id = extract_transcript_id(filename)

            # Extract dimension scores
            scores = {}
            for clinician_dim, judge_dim in DIMENSION_MAPPING.items():
                if judge_dim in results_df.columns:
                    score_value = row[judge_dim]
                    # Convert to string and check if it's a valid non-empty value
                    score_str = str(score_value).strip()
                    if score_str and score_str.lower() not in ("nan", "none", ""):
                        scores[judge_dim] = score_str

            if scores:
                judge_scores[transcript_id] = scores

        # Get test transcript IDs from expected ratings
        test_transcript_ids = set(expected_ratings.keys())

        # Compare to expected ratings
        mismatches = []
        for transcript_id in test_transcript_ids:
            if transcript_id not in judge_scores:
                mismatches.append(
                    f"{transcript_id}: No judge scores found (expected ratings exist)"
                )
                continue

            expected = expected_ratings[transcript_id]
            actual = judge_scores[transcript_id]

            for dimension, expected_rating in expected.items():
                if dimension not in actual:
                    mismatches.append(
                        f"{transcript_id} - {dimension}: Missing score "
                        f"(expected: {expected_rating})"
                    )
                elif actual[dimension] != expected_rating:
                    mismatches.append(
                        f"{transcript_id} - {dimension}: "
                        f"Expected '{expected_rating}', got '{actual[dimension]}'"
                    )

        if mismatches:
            error_msg = "Judge scores don't match expected clinician ratings:\n"
            error_msg += "\n".join(f"  - {m}" for m in mismatches)
            pytest.fail(error_msg)

        # If we get here, all scores matched
        assert len(judge_scores) > 0, "No judge scores were generated"
