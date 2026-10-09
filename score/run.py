"""
Score one judge run's ``results.csv``: the scoring application entry point.

Run with `vera score -r <results.csv>`; the legacy CLI is `legacy/score.py`.

Reads results.csv, re-calculates the dataframe from the tsv files in the same
folder if the results.csv is empty, calculates dimension-level and aggregate scores,
and outputs to console, JSON files under ``scores/``, and generates visualizations:
- scores/scores_visualization.png: Overall scores with pie chart and dimension breakdown
- scores/scores_by_risk_visualization.png: Scores broken down by persona risk level
"""

import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from judge.score_utils import (
    build_dataframe_from_tsv_files,
    has_dimension_data,
    read_judge_results_csv,
)
from score.aggregate import (
    print_scores,
    score_results,
    score_results_by_risk,
    scores_output_dir,
)
from score.viz import create_risk_level_visualizations, create_visualizations


def _rebuild_dataframe_if_needed(results_csv_path: Path) -> bool:
    """Rebuild dataframe from TSV files if dimension columns are empty."""
    df_existing = read_judge_results_csv(results_csv_path)
    if has_dimension_data(df_existing):
        return False

    print(f"⚠️  Dimension columns are empty in {results_csv_path}")
    print(f"📊 Rebuilding dataframe from TSV files in {results_csv_path.parent}...")

    try:
        df_new = build_dataframe_from_tsv_files(results_csv_path.parent)

        # Preserve existing columns (like question_id and reasoning)
        merge_cols = ["filename"]
        if "run_id" in df_existing.columns and "run_id" in df_new.columns:
            merge_cols.append("run_id")

        # Get columns to preserve (exclude merge columns and columns already in df_new)
        cols_to_preserve = [
            col
            for col in df_existing.columns
            if col not in df_new.columns and col not in merge_cols
        ]

        if cols_to_preserve:
            # Merge to add preserved columns
            df_existing_subset = df_existing[merge_cols + cols_to_preserve]
            df = df_new.merge(df_existing_subset, on=merge_cols, how="left")
            print(
                f"✅ Preserved {len(cols_to_preserve)} additional columns "
                "from existing CSV"
            )
        else:
            df = df_new

        df.to_csv(results_csv_path, index=False)
        print(
            f"✅ Rebuilt dataframe with {len(df)} rows and saved to {results_csv_path}"
        )
        return True
    except Exception as e:
        print(f"❌ Error rebuilding dataframe from TSV files: {e}")
        return False


def run_scoring(
    *,
    results_csv: str,
    output_json: Optional[str],
    personas_tsv: Optional[str],
    skip_risk_analysis: bool,
) -> Tuple[Dict[str, Any], Path]:
    """Score one ``results.csv`` from fully resolved inputs.

    This is the scoring domain's application function, the counterpart of
    `judge.run.run_judging`: it receives final values, does the work, and
    returns. It does not parse arguments, apply defaults, or decide where
    anything lives -- every one of those is the caller's. That boundary is what
    lets one function serve both `vera score` and the legacy `legacy/score.py`.

    Visualization failures are warnings rather than errors: the scores are the
    result, and a chart that could not be drawn does not invalidate them.

    Args:
        results_csv: Path to the results CSV produced by judging
        output_json: Full path for the primary scores JSON, or None to write
            ``scores/scores.json`` beside the CSV
        personas_tsv: Personas file backing the risk-level grouping, or None to
            skip that step
        skip_risk_analysis: Skip risk-level analysis outright, regardless of
            whether a personas file was given

    Returns:
        Tuple of (scores, scores_dir) where scores_dir holds the derived
        artifacts written beside the CSV.

    Raises:
        FileNotFoundError: If the results CSV does not exist.
        ValueError: If the CSV has no dimension data and cannot be rebuilt.
    """
    results_csv_path = Path(results_csv)
    if not results_csv_path.exists():
        raise FileNotFoundError(f"Results CSV file not found: {results_csv}")

    if not _rebuild_dataframe_if_needed(results_csv_path):
        # Rebuild declined or failed; only an error if there was nothing to score.
        if not has_dimension_data(read_judge_results_csv(results_csv_path)):
            raise ValueError(
                f"{results_csv} has no dimension data and could not be rebuilt "
                "from the TSV files beside it"
            )

    results = score_results(str(results_csv_path), output_path=output_json)
    print_scores(results)

    scores_dir = scores_output_dir(str(results_csv_path))
    json_path = Path(output_json) if output_json else scores_dir / "scores.json"
    print(f"\n✅ Scores saved to: {json_path}")

    viz_path = scores_dir / "scores_visualization.png"
    try:
        create_visualizations(results, viz_path)
    except Exception as e:
        print(f"⚠️  Warning: Could not create standard visualizations: {e}")

    if skip_risk_analysis:
        return results, scores_dir

    if personas_tsv is None:
        # `vera score` reaches here whenever --personas was omitted. Grouping by
        # risk reads a persona column, so with no personas file there is nothing
        # to group by -- say so rather than emitting empty buckets.
        print("⚠️  No personas file given; skipping risk-level analysis.")
        print("   Pass a personas TSV to group scores by risk level.")
        return results, scores_dir

    personas_tsv_path = Path(personas_tsv)
    if not personas_tsv_path.exists():
        print(f"⚠️  Warning: Personas TSV file not found: {personas_tsv}")
        print(
            "   Skipping risk-level analysis. Use --skip-risk-analysis "
            "to suppress this warning."
        )
        return results, scores_dir

    try:
        risk_results = score_results_by_risk(
            str(results_csv_path), str(personas_tsv_path)
        )
        risk_viz_path = scores_dir / "scores_by_risk_visualization.png"
        create_risk_level_visualizations(risk_results, risk_viz_path)
    except Exception as e:
        print(f"⚠️  Warning: Could not create risk-level analysis: {e}")
        traceback.print_exc()

    return results, scores_dir
