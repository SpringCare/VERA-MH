"""
Committed evaluation TSVs + results.csv stay in sync with build_dataframe_from_tsv_files
"""

from pathlib import Path

import pandas as pd

from judge.score_utils import build_dataframe_from_tsv_files, read_results_csv

# Folder name must contain ``__<run_id>`` so :func:`build_results_csv_from_tsv_files`
# assigns the same ``run_id`` as in the saved ``results.csv``.
_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "eval_tsv_rebuild__20260415_140037"
)


def test_rebuilt_dataframe_matches_committed_results_csv() -> None:
    df_csv = read_results_csv(_FIXTURE_DIR / "results.csv")
    df_tsv = build_dataframe_from_tsv_files(_FIXTURE_DIR)

    sort_cols = ["filename", "judge_model", "judge_instance"]
    df_csv = df_csv.sort_values(sort_cols).reset_index(drop=True)
    df_tsv = df_tsv.sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        df_csv[df_tsv.columns],
        df_tsv,
        check_dtype=False,
    )
