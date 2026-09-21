"""Unit tests for judge rubric configuration."""

from pathlib import Path

import pandas as pd
import pytest

from judge.rubric_config import (
    COL_ANSWER,
    COL_DIMENSION,
    COL_EXAMPLES,
    COL_GOTO,
    COL_QUESTION,
    COL_QUESTION_ID,
    COL_RISK_TYPE,
    COL_SEVERITY,
    EXPECTED_DIMENSION_NAMES,
    IGNORE_COLUMNS,
)


@pytest.mark.unit
class TestRubricConfigConstants:
    """Tests for rubric configuration constants."""

    def test_rubric_columns_match_actual_tsv(self):
        """Test that rubric column constants match the actual rubric.tsv file.

        This test ensures that if the rubric.tsv column names change,
        the constants in rubric_config.py are updated accordingly.
        """
        # Load the actual rubric file
        rubric_path = Path("data/rubric.tsv")
        assert rubric_path.exists(), f"Rubric file not found: {rubric_path}"

        df = pd.read_csv(rubric_path, sep="\t")
        actual_columns = set(c for c in df.columns if not str(c).startswith("Unnamed"))

        # Define expected columns from our constants
        expected_columns = {
            COL_QUESTION_ID,
            COL_DIMENSION,
            COL_SEVERITY,
            COL_RISK_TYPE,
            COL_QUESTION,
            COL_EXAMPLES,
            COL_ANSWER,
            COL_GOTO,
        }

        # Check that all expected columns exist in the actual file
        missing_columns = expected_columns - actual_columns
        assert not missing_columns, (
            f"Constants defined in rubric_config.py but missing from rubric.tsv: "
            f"{missing_columns}. Please update the rubric "
            "or add the missing columns to the constants in rubric_config."
        )

        # Check for extra columns in rubric.tsv that aren't in our constants
        # Only allowed_extra columns are allowed as extra columns
        allowed_extra = IGNORE_COLUMNS
        extra_columns = actual_columns - expected_columns
        disallowed_extra = extra_columns - allowed_extra
        assert not disallowed_extra, (
            f"rubric.tsv has extra columns {disallowed_extra} not defined. "
            "Please add the missing columns to the constants in rubric_config.py "
            "or remove the columns from the rubric."
        )

    def test_dimension_values_match_rubric(self):
        """Test that EXPECTED_DIMENSION_NAMES matches actual dimensions in rubric.tsv.

        This ensures that if dimensions are added/removed/renamed in the rubric,
        EXPECTED_DIMENSION_NAMES in rubric_config.py is updated.
        """
        # Load the actual rubric file
        rubric_path = Path("data/rubric.tsv")
        assert rubric_path.exists(), f"Rubric file not found: {rubric_path}"

        df = pd.read_csv(rubric_path, sep="\t")

        # Get actual dimensions from the file
        actual_dimensions = set(df[COL_DIMENSION].dropna().unique())

        # Get expected dimensions from our constants
        expected_dimensions = EXPECTED_DIMENSION_NAMES

        # Check that all dimensions in the rubric are in EXPECTED_DIMENSION_NAMES
        missing = actual_dimensions - expected_dimensions
        assert not missing, (
            f"Dimensions in rubric.tsv not in EXPECTED_DIMENSION_NAMES: {missing}."
            "Please add them to EXPECTED_DIMENSION_NAMES in rubric_config.py."
        )

        # Check for expected dimensions that no longer exist in rubric
        extra = expected_dimensions - actual_dimensions
        assert not extra, (
            f"EXPECTED_DIMENSION_NAMES contains dimensions not in rubric.tsv: {extra}."
            "Please remove them from EXPECTED_DIMENSION_NAMES in rubric_config.py."
        )

    def test_expected_dimension_names_structure(self):
        """Test that EXPECTED_DIMENSION_NAMES has valid structure."""
        assert isinstance(
            EXPECTED_DIMENSION_NAMES, set
        ), "EXPECTED_DIMENSION_NAMES should be a set"

        for name in EXPECTED_DIMENSION_NAMES:
            assert isinstance(
                name, str
            ), f"Dimension name should be a string, got {type(name)}"
            assert name, "Dimension name should not be empty"

    def test_rubric_file_can_be_parsed_with_constants(self):
        """Test that the prod rubric file can be successfully parsed using constants."""
        rubric_path = Path("data/rubric.tsv")
        df = pd.read_csv(rubric_path, sep="\t")

        # Verify we can access all columns using our constants
        assert COL_QUESTION_ID in df.columns
        assert COL_DIMENSION in df.columns
        assert COL_SEVERITY in df.columns
        assert COL_RISK_TYPE in df.columns
        assert COL_QUESTION in df.columns
        assert COL_EXAMPLES in df.columns
        assert COL_ANSWER in df.columns
        assert COL_GOTO in df.columns

        # Verify we can read data from each column
        question_ids = df[COL_QUESTION_ID].dropna()
        dimensions = df[COL_DIMENSION].dropna()
        questions = df[COL_QUESTION].dropna()

        assert len(question_ids) > 0, "Should have at least one question ID"
        assert len(dimensions) > 0, "Should have at least one dimension"
        assert len(questions) > 0, "Should have at least one question"

    def test_no_duplicate_dimensions(self):
        """Test that there are no duplicate dimension names in the rubric."""
        rubric_path = Path("data/rubric.tsv")
        df = pd.read_csv(rubric_path, sep="\t")

        dimensions = df[COL_DIMENSION].dropna().tolist()
        unique_dimensions = set(dimensions)

        # It's okay to have dimensions repeated across rows (for different questions),
        # but the unique set should match EXPECTED_DIMENSION_NAMES
        assert len(unique_dimensions) == len(EXPECTED_DIMENSION_NAMES), (
            f"Number of unique dimensions in rubric ({len(unique_dimensions)}) "
            f"doesn't match EXPECTED_DIMENSION_NAMES ({len(EXPECTED_DIMENSION_NAMES)})"
        )
