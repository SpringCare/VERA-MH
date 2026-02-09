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
    DIMENSION_SHORT_NAMES,
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
        """Test that DIMENSION_SHORT_NAMES keys match actual dimensions in rubric.tsv.

        This ensures that if dimensions are added/removed/renamed in the rubric,
        the DIMENSION_SHORT_NAMES dict is updated.
        """
        # Load the actual rubric file
        rubric_path = Path("data/rubric.tsv")
        assert rubric_path.exists(), f"Rubric file not found: {rubric_path}"

        df = pd.read_csv(rubric_path, sep="\t")

        # Get actual dimensions from the file
        actual_dimensions = set(df[COL_DIMENSION].dropna().unique())

        # Get dimensions from our constants
        expected_dimensions = set(DIMENSION_SHORT_NAMES.keys())

        # Check that all dimensions in the rubric have short names defined
        missing_short_names = actual_dimensions - expected_dimensions
        assert not missing_short_names, (
            f"Dimensions in rubric.tsv without short names defined: "
            f"{missing_short_names}. Please add them to DIMENSION_SHORT_NAMES "
            f"in rubric_config.py."
        )

        # Check for dimensions with short names that no longer exist in rubric
        extra_short_names = expected_dimensions - actual_dimensions
        assert not extra_short_names, (
            f"Dimensions with short names defined but not in rubric.tsv: "
            f"{extra_short_names}. Please remove them from DIMENSION_SHORT_NAMES "
            f"in rubric_config.py."
        )

    def test_dimension_short_names_structure(self):
        """Test that DIMENSION_SHORT_NAMES has valid structure."""
        assert isinstance(
            DIMENSION_SHORT_NAMES, dict
        ), "DIMENSION_SHORT_NAMES should be a dictionary"

        for full_name, short_name in DIMENSION_SHORT_NAMES.items():
            assert isinstance(
                full_name, str
            ), f"Dimension full name should be a string, got {type(full_name)}"
            assert isinstance(
                short_name, str
            ), f"Dimension short name should be a string, got {type(short_name)}"
            assert full_name, "Dimension full name should not be empty"
            assert short_name, "Dimension short name should not be empty"

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
        # but the unique set should match DIMENSION_SHORT_NAMES
        assert len(unique_dimensions) == len(DIMENSION_SHORT_NAMES), (
            f"Number of unique dimensions in rubric ({len(unique_dimensions)}) "
            f"doesn't match DIMENSION_SHORT_NAMES ({len(DIMENSION_SHORT_NAMES)})"
        )
