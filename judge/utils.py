"""Utility functions for the judge module."""

import pandas as pd
from pathlib import Path
from typing import List, Tuple


def load_rubric_structure(rubric_path: str, sep: str = "\t") -> Tuple[List[str], List[str]]:
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
    dimensions = [d.strip() for d in rubric_df['Dimension'].dropna().unique() if d and str(d).strip() != 'nan']

    # Get options from columns (exclude metadata columns)
    columns = [col.strip() for col in rubric_df.columns]
    # Question-flow rubric columns: Question ID, Dimension, Risk Type, Question, Examples, Severity, Answer, GOTO
    metadata_columns = {'Question ID', 'Dimension', 'Risk Type', 'Question', 'Examples', 'Severity', 'Answer', 'GOTO'}
    options = [col for col in columns if col not in metadata_columns]

    return dimensions, options
