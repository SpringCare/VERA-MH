"""SI-specific assertions for the shipped rubric structure."""

from pathlib import Path

import pandas as pd
import pytest

from judge.rubric_config import (
    COL_ANSWER,
    COL_AUTO_ANSWER,
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

pytestmark = pytest.mark.integration
RUBRIC_PATH = Path("data/SI/rubric.tsv")


def load_si_rubric() -> pd.DataFrame:
    assert RUBRIC_PATH.exists(), f"Rubric file not found: {RUBRIC_PATH}"
    return pd.read_csv(RUBRIC_PATH, sep="\t", dtype=str)


class TestSIRubricStructure:
    def test_rubric_columns_match_constants(self):
        rubric_df = load_si_rubric()
        actual_columns = {
            column
            for column in rubric_df.columns
            if not str(column).startswith("Unnamed")
        }
        expected_columns = {
            COL_QUESTION_ID,
            COL_DIMENSION,
            COL_SEVERITY,
            COL_RISK_TYPE,
            COL_QUESTION,
            COL_EXAMPLES,
            COL_ANSWER,
            COL_GOTO,
            COL_AUTO_ANSWER,
        }

        assert expected_columns <= actual_columns
        assert actual_columns - expected_columns <= IGNORE_COLUMNS

    def test_dimension_values_match_expected_si_dimensions(self):
        actual_dimensions = set(load_si_rubric()[COL_DIMENSION].dropna().unique())
        assert actual_dimensions == EXPECTED_DIMENSION_NAMES

    def test_expected_dimension_names_are_nonempty_strings(self):
        assert isinstance(EXPECTED_DIMENSION_NAMES, set)
        assert all(isinstance(name, str) and name for name in EXPECTED_DIMENSION_NAMES)

    def test_rubric_fields_are_populated(self):
        rubric_df = load_si_rubric()
        assert not rubric_df[COL_QUESTION_ID].dropna().empty
        assert not rubric_df[COL_DIMENSION].dropna().empty
        assert not rubric_df[COL_QUESTION].dropna().empty

    def test_unique_dimension_count_matches_expected_si_dimensions(self):
        dimensions = set(load_si_rubric()[COL_DIMENSION].dropna())
        assert len(dimensions) == len(EXPECTED_DIMENSION_NAMES)
