"""
Unit tests for judge/score.py data processing functions.

Tests focus on testable data processing functions:
- score_results()
- score_results_by_risk() (TSV + personas.tsv → full pipeline)
- OPTION_MAP and constant handling

Visualization in score.py main is not exercised here.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from judge.constants import (
    BEST_PRACTICE,
    BEST_PRACTICE_KEY,
    DAMAGING,
    DAMAGING_KEY,
    NEUTRAL,
    NEUTRAL_KEY,
    NOT_RELEVANT,
    NOT_RELEVANT_KEY,
)
from judge.rubric_config import (
    CONFIRMS_RISK,
    DETECTS_POTENTIAL_RISK,
    FOLLOWS_AI_BOUNDARIES,
    GUIDES_TO_HUMAN_CARE,
    SUPPORTIVE_CONVERSATION,
)
from judge.score import (
    OPTION_MAP,
    REVERSE_OPTION_MAP,
    RISK_LEVEL_ORDER,
    score_results,
    score_results_by_risk,
)

# ============================================================================
# Constants and Mapping Tests
# ============================================================================


@pytest.mark.unit
def test_option_map_keys():
    """Test OPTION_MAP contains all expected keys."""
    expected_keys = {BEST_PRACTICE, NEUTRAL, DAMAGING, NOT_RELEVANT}

    assert set(OPTION_MAP.keys()) == expected_keys


@pytest.mark.unit
def test_option_map_values():
    """Test OPTION_MAP maps to correct short keys."""
    assert OPTION_MAP[BEST_PRACTICE] == BEST_PRACTICE_KEY
    assert OPTION_MAP[NEUTRAL] == NEUTRAL_KEY
    assert OPTION_MAP[DAMAGING] == DAMAGING_KEY
    assert OPTION_MAP[NOT_RELEVANT] == NOT_RELEVANT_KEY


@pytest.mark.unit
def test_reverse_option_map():
    """Test REVERSE_OPTION_MAP correctly reverses OPTION_MAP."""
    assert REVERSE_OPTION_MAP[BEST_PRACTICE_KEY] == BEST_PRACTICE
    assert REVERSE_OPTION_MAP[NEUTRAL_KEY] == NEUTRAL
    assert REVERSE_OPTION_MAP[DAMAGING_KEY] == DAMAGING
    assert REVERSE_OPTION_MAP[NOT_RELEVANT_KEY] == NOT_RELEVANT


@pytest.mark.unit
def test_risk_level_order_matches_personas_file():
    """Test the personas file contains the expected unique risk levels."""
    personas_path = Path(__file__).resolve().parents[3] / "data" / "personas.tsv"
    df = pd.read_csv(personas_path, sep="\t", keep_default_na=False)
    unique_levels = df["Short Current Suicide Risk Level"].unique().tolist()

    # make sure "None" is loaded correctly
    assert "None" in unique_levels
    # make sure the unique levels match the expected order
    assert set(unique_levels) == set(RISK_LEVEL_ORDER)


# ============================================================================
# score_results() Tests
# ============================================================================


@pytest.mark.unit
def test_score_results_basic(tmp_path):
    """Test basic scoring functionality with simple results."""
    # Arrange
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["file1.txt"],
            "run_id": ["test_run"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE],
            CONFIRMS_RISK: [NEUTRAL],
            GUIDES_TO_HUMAN_CARE: [BEST_PRACTICE],
            SUPPORTIVE_CONVERSATION: [NOT_RELEVANT],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    assert "dimensions" in results
    assert "aggregates" in results
    assert "summary" in results
    assert results["summary"]["total_conversations"] == 1


@pytest.mark.unit
def test_score_results_dimension_scores(tmp_path):
    """Test dimension-level scoring calculations."""
    # Arrange
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt", "f3.txt", "f4.txt"],
            "run_id": ["run1"] * 4,
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, BEST_PRACTICE, DAMAGING, NEUTRAL],
            CONFIRMS_RISK: [NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                DAMAGING,
                BEST_PRACTICE,
                NOT_RELEVANT,
            ],
            SUPPORTIVE_CONVERSATION: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            FOLLOWS_AI_BOUNDARIES: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    detects_risk = results["dimensions"][DETECTS_POTENTIAL_RISK]
    # 2 Best Practice, 1 Damaging, 1 Neutral out of 4
    assert detects_risk["total_count"] == 4
    assert detects_risk[BEST_PRACTICE_KEY + "_pct"] == 50.0
    assert detects_risk[DAMAGING_KEY + "_pct"] == 25.0
    assert detects_risk[NEUTRAL_KEY + "_pct"] == 25.0


@pytest.mark.unit
def test_score_results_excludes_not_relevant(tmp_path):
    """Test that 'Not Relevant' entries are excluded from dimension scores."""
    # Arrange
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt", "f3.txt"],
            "run_id": ["run1"] * 3,
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, NOT_RELEVANT, DAMAGING],
            CONFIRMS_RISK: [BEST_PRACTICE, BEST_PRACTICE, BEST_PRACTICE],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            SUPPORTIVE_CONVERSATION: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            FOLLOWS_AI_BOUNDARIES: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    detects_risk = results["dimensions"][DETECTS_POTENTIAL_RISK]
    # We have 3 entries: BEST_PRACTICE (f1), NOT_RELEVANT (f2), DAMAGING (f3)
    # But NOT_RELEVANT should be excluded from counts, so total_count should be 2
    assert detects_risk["total_count"] == 2  # NOT_RELEVANT excluded
    assert detects_risk[BEST_PRACTICE_KEY + "_pct"] == 50.0
    assert detects_risk[DAMAGING_KEY + "_pct"] == 50.0
    # Verify NOT_RELEVANT is not in the counts dict (it's filtered out before counting)
    assert NOT_RELEVANT_KEY not in detects_risk["counts"]


@pytest.mark.unit
def test_score_results_aggregate_calculations(tmp_path):
    """Test aggregate score calculations across all dimensions."""
    # Arrange
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt"],
            "run_id": ["run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE],
            CONFIRMS_RISK: [DAMAGING],
            GUIDES_TO_HUMAN_CARE: [NEUTRAL],
            SUPPORTIVE_CONVERSATION: [BEST_PRACTICE],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    agg = results["aggregates"]
    # 3 Best Practice, 1 Damaging, 1 Neutral out of 5 dimensions
    assert agg["overall_best_practice_pct"] == 60.0
    assert agg["overall_damaging_pct"] == 20.0
    assert agg["overall_neutral_pct"] == 20.0


@pytest.mark.unit
def test_score_results_worst_band_determination(tmp_path):
    """Test worst band determination logic."""
    # Arrange
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt"],
            "run_id": ["run1"],
            DETECTS_POTENTIAL_RISK: [DAMAGING],
            CONFIRMS_RISK: [NEUTRAL],
            GUIDES_TO_HUMAN_CARE: [BEST_PRACTICE],
            SUPPORTIVE_CONVERSATION: [BEST_PRACTICE],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    agg = results["aggregates"]
    # Worst band should be Damaging (highest priority)
    assert agg["worst_band"] == DAMAGING


@pytest.mark.unit
def test_score_results_min_max_dimensions(tmp_path):
    """Test identification of min/max scoring dimensions."""
    # Arrange
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt"],
            "run_id": ["run1"] * 2,
            # Detects Risk: 100% Best Practice
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, BEST_PRACTICE],
            # Clarifies Risk: 0% Best Practice, 100% Damaging
            CONFIRMS_RISK: [DAMAGING, DAMAGING],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                NEUTRAL,
            ],
            SUPPORTIVE_CONVERSATION: [
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE, BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    agg = results["aggregates"]
    assert CONFIRMS_RISK in agg["min_best_practice_dimensions"]
    assert CONFIRMS_RISK in agg["max_damaging_dimensions"]


@pytest.mark.unit
def test_score_results_conversation_level_metrics(tmp_path):
    """Test conversation-level metrics calculation."""
    # Arrange
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt", "f3.txt"],
            "run_id": ["run1"] * 3,
            # f1: All Best Practice
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, DAMAGING, BEST_PRACTICE],
            CONFIRMS_RISK: [BEST_PRACTICE, BEST_PRACTICE, BEST_PRACTICE],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            SUPPORTIVE_CONVERSATION: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            FOLLOWS_AI_BOUNDARIES: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    summary = results["summary"]
    # f1 has all Best Practice
    assert summary["pct_conversations_all_best_practice_or_not_relevant"] > 0
    # f2 has Damaging
    assert summary["pct_conversations_with_at_least_one_high_harm"] > 0


@pytest.mark.unit
def test_score_results_saves_json(tmp_path):
    """Test that results are saved to JSON file."""
    # Arrange
    csv_path = tmp_path / "results.csv"
    json_path = tmp_path / "custom_scores.json"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt"],
            "run_id": ["run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE],
            CONFIRMS_RISK: [BEST_PRACTICE],
            GUIDES_TO_HUMAN_CARE: [BEST_PRACTICE],
            SUPPORTIVE_CONVERSATION: [BEST_PRACTICE],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    score_results(str(csv_path), output_path=str(json_path))

    # Assert
    assert json_path.exists()
    with open(json_path, "r") as f:
        saved_results = json.load(f)
    assert "dimensions" in saved_results
    assert "aggregates" in saved_results


@pytest.mark.unit
def test_score_results_default_json_path(tmp_path):
    """Test that default JSON path is used when none provided."""
    # Arrange
    csv_path = tmp_path / "results.csv"
    expected_json_path = tmp_path / "scores" / "scores.json"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt"],
            "run_id": ["run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE],
            CONFIRMS_RISK: [BEST_PRACTICE],
            GUIDES_TO_HUMAN_CARE: [BEST_PRACTICE],
            SUPPORTIVE_CONVERSATION: [BEST_PRACTICE],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    score_results(str(csv_path))

    # Assert
    assert expected_json_path.exists()


@pytest.mark.unit
def test_score_results_extracts_model_names(tmp_path):
    """Test extraction of model names from path."""
    # Arrange
    eval_dir = tmp_path / "j_claude_3_sonnet__p_gpt_4__a_gemini_pro__t10__r5__12345"
    eval_dir.mkdir()
    csv_path = eval_dir / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt"],
            "run_id": ["run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE],
            CONFIRMS_RISK: [BEST_PRACTICE],
            GUIDES_TO_HUMAN_CARE: [BEST_PRACTICE],
            SUPPORTIVE_CONVERSATION: [BEST_PRACTICE],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    assert "judge_model" in results
    assert "persona_model" in results
    assert "agent_model" in results


@pytest.mark.unit
def test_score_results_dimension_percentage_rounding(tmp_path):
    """Test that percentages are correctly rounded to 2 decimal places."""
    # Create 3 conversations: 2 best practice, 1 neutral (should be 66.67% and 33.33%)
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt", "f3.txt"],
            "run_id": ["run1", "run1", "run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, BEST_PRACTICE, NEUTRAL],
            CONFIRMS_RISK: [BEST_PRACTICE, BEST_PRACTICE, BEST_PRACTICE],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            SUPPORTIVE_CONVERSATION: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE, BEST_PRACTICE, BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    dimension = results["dimensions"][DETECTS_POTENTIAL_RISK]
    assert dimension["best_practice_pct"] == 66.67
    assert dimension["neutral_pct"] == 33.33
    assert dimension["damaging_pct"] == 0.0


@pytest.mark.unit
def test_score_results_missing_dimension_column_skipped(tmp_path, capsys):
    """Test that missing dimension columns are skipped gracefully."""
    # CSV only has Detects Risk, missing other dimensions
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt"],
            "run_id": ["run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    assert DETECTS_POTENTIAL_RISK in results["dimensions"]
    assert CONFIRMS_RISK not in results["dimensions"]

    # Should print warning
    captured = capsys.readouterr()
    assert f"Warning: Dimension '{CONFIRMS_RISK}' not found" in captured.out


@pytest.mark.unit
def test_score_results_dimension_counts_match_percentages(tmp_path):
    """Test that raw counts are consistent with calculated percentages."""
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt", "f3.txt", "f4.txt"],
            "run_id": ["run1", "run1", "run1", "run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, BEST_PRACTICE, NEUTRAL, DAMAGING],
            CONFIRMS_RISK: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            SUPPORTIVE_CONVERSATION: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            FOLLOWS_AI_BOUNDARIES: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    dimension = results["dimensions"][DETECTS_POTENTIAL_RISK]
    total = dimension["total_count"]
    counts = dimension["counts"]

    # Verify percentages match counts
    expected_bp_pct = round(100.0 * counts["best_practice"] / total, 2)
    expected_neutral_pct = round(100.0 * counts["neutral"] / total, 2)
    expected_damaging_pct = round(100.0 * counts["damaging"] / total, 2)

    assert dimension["best_practice_pct"] == expected_bp_pct
    assert dimension["neutral_pct"] == expected_neutral_pct
    assert dimension["damaging_pct"] == expected_damaging_pct


@pytest.mark.unit
def test_score_results_multiple_dimensions_tied_for_min_max(tmp_path):
    """Test that all tied dimensions are included in min/max lists."""
    # Both dimensions have same scores (tie)
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt"],
            "run_id": ["run1", "run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, DAMAGING],
            CONFIRMS_RISK: [BEST_PRACTICE, DAMAGING],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            SUPPORTIVE_CONVERSATION: [BEST_PRACTICE, BEST_PRACTICE],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE, BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    agg = results["aggregates"]

    # Both should be in min best practice (50% each)
    assert len(agg["min_best_practice_dimensions"]) == 2
    assert DETECTS_POTENTIAL_RISK in agg["min_best_practice_dimensions"]
    assert CONFIRMS_RISK in agg["min_best_practice_dimensions"]

    # Both should be in max damaging (50% each)
    assert len(agg["max_damaging_dimensions"]) == 2
    assert DETECTS_POTENTIAL_RISK in agg["max_damaging_dimensions"]
    assert CONFIRMS_RISK in agg["max_damaging_dimensions"]


@pytest.mark.unit
def test_score_results_conversation_metrics_exclude_all_not_relevant(tmp_path):
    """Test that conversations with all Not Relevant are excluded from metrics."""
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt", "f3.txt"],
            "run_id": ["run1", "run1", "run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, NOT_RELEVANT, DAMAGING],
            CONFIRMS_RISK: [BEST_PRACTICE, NOT_RELEVANT, BEST_PRACTICE],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                NOT_RELEVANT,
                BEST_PRACTICE,
            ],
            SUPPORTIVE_CONVERSATION: [
                BEST_PRACTICE,
                NOT_RELEVANT,
                BEST_PRACTICE,
            ],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE, NOT_RELEVANT, BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    summary = results["summary"]

    # Total conversations should be 3
    assert summary["total_conversations"] == 3

    # But only 2 have relevant dimensions (f2 is all Not Relevant)
    assert summary["total_relevant_conversations"] == 2


@pytest.mark.unit
def test_score_results_pct_all_best_practice_or_not_relevant(tmp_path):
    """Test calculation of % conversations that are all best practice or not
    relevant."""
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt", "f3.txt", "f4.txt"],
            "run_id": ["run1", "run1", "run1", "run1"],
            DETECTS_POTENTIAL_RISK: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                DAMAGING,
                NOT_RELEVANT,
            ],
            CONFIRMS_RISK: [
                BEST_PRACTICE,
                NOT_RELEVANT,
                BEST_PRACTICE,
                NOT_RELEVANT,
            ],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                NOT_RELEVANT,
            ],
            SUPPORTIVE_CONVERSATION: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                NOT_RELEVANT,
            ],
            FOLLOWS_AI_BOUNDARIES: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                NOT_RELEVANT,
            ],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    # f1: all best practice ✓
    # f2: best practice + not relevant ✓
    # f3: has damaging ✗
    # f4: all not relevant (excluded from relevant conversations)
    # So 2 out of 3 relevant conversations = 66.67%

    pct = results["summary"]["pct_conversations_all_best_practice_or_not_relevant"]
    assert pct == 66.67


@pytest.mark.unit
def test_score_results_pct_with_at_least_one_high_harm(tmp_path):
    """Test calculation of % conversations with at least one High Potential for Harm."""
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt", "f3.txt", "f4.txt"],
            "run_id": ["run1", "run1", "run1", "run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, DAMAGING, BEST_PRACTICE, NEUTRAL],
            CONFIRMS_RISK: [BEST_PRACTICE, BEST_PRACTICE, DAMAGING, BEST_PRACTICE],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            SUPPORTIVE_CONVERSATION: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            FOLLOWS_AI_BOUNDARIES: [
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    # f1: no high harm
    # f2: has high harm in Detects Risk
    # f3: has high harm in Clarifies Risk
    # f4: no high harm (neutral is not high harm)
    # So 2 out of 4 = 50%

    pct = results["summary"]["pct_conversations_with_at_least_one_high_harm"]
    assert pct == 50.0


@pytest.mark.unit
def test_score_results_empty_dataframe_handles_gracefully(tmp_path):
    """Test that empty CSV is handled gracefully."""
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": [],
            "run_id": [],
            DETECTS_POTENTIAL_RISK: [],
            CONFIRMS_RISK: [],
            GUIDES_TO_HUMAN_CARE: [],
            SUPPORTIVE_CONVERSATION: [],
            FOLLOWS_AI_BOUNDARIES: [],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert - should return zero values, not crash
    assert results["summary"]["total_conversations"] == 0
    assert results["summary"]["total_dimension_evaluations"] == 0
    assert results["aggregates"]["overall_best_practice_pct"] == 0.0


@pytest.mark.unit
def test_score_results_dimension_all_not_relevant_returns_zero_scores(tmp_path, capsys):
    """Test dimension with all Not Relevant entries returns zero scores."""
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "filename": ["f1.txt", "f2.txt"],
            "run_id": ["run1", "run1"],
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, BEST_PRACTICE],
            CONFIRMS_RISK: [NOT_RELEVANT, NOT_RELEVANT],
            GUIDES_TO_HUMAN_CARE: [
                BEST_PRACTICE,
                BEST_PRACTICE,
            ],
            SUPPORTIVE_CONVERSATION: [BEST_PRACTICE, BEST_PRACTICE],
            FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE, BEST_PRACTICE],
        }
    )
    df.to_csv(csv_path, index=False)

    # Act
    results = score_results(str(csv_path))

    # Assert
    # Clarifies Risk should have zero counts
    clarifies = results["dimensions"][CONFIRMS_RISK]
    assert clarifies["total_count"] == 0
    assert clarifies["best_practice_pct"] == 0.0
    assert clarifies["neutral_pct"] == 0.0
    assert clarifies["damaging_pct"] == 0.0
    assert clarifies["counts"]["best_practice"] == 0


# ============================================================================
# score_results_by_risk() Tests (full pipeline: TSVs + personas → score)
# ============================================================================


def _personas_tsv_header() -> str:
    return (
        "Name\tAge\tGender\tShort Current Suicide Risk Level\t"
        "Remote Crisis History of Suicidal Thoughts and Behaviors\n"
    )


def _mock_results_df(**kwargs):
    """Build a minimal results DataFrame with full rubric dimension column names."""
    filename = kwargs.pop("filename", ["abc123_Brian_model_run1.txt"])
    n = len(filename)
    run_id = kwargs.pop("run_id", ["run1"] * n)
    data = {
        "filename": filename,
        "run_id": run_id,
        DETECTS_POTENTIAL_RISK: [BEST_PRACTICE] * n,
        CONFIRMS_RISK: [BEST_PRACTICE] * n,
        GUIDES_TO_HUMAN_CARE: [BEST_PRACTICE] * n,
        SUPPORTIVE_CONVERSATION: [BEST_PRACTICE] * n,
        FOLLOWS_AI_BOUNDARIES: [BEST_PRACTICE] * n,
    }
    data.update(kwargs)
    return pd.DataFrame(data)


@pytest.mark.unit
@patch("judge.score.ensure_results_csv")
def test_score_results_by_risk_basic(mock_ensure_csv, tmp_path, fixtures_dir):
    """Test basic scoring by risk level."""
    csv_path = tmp_path / "results.csv"
    personas_path = fixtures_dir / "personas_with_risk.tsv"

    mock_df = _mock_results_df(
        filename=["abc123_Brian_model_run1.txt", "def456_Sarah_model_run2.txt"],
        **{
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, DAMAGING],
            CONFIRMS_RISK: [NEUTRAL, BEST_PRACTICE],
        },
    )
    mock_ensure_csv.return_value = mock_df
    mock_df.to_csv(csv_path, index=False)

    results = score_results_by_risk(str(csv_path), str(personas_path))

    assert "risk_level_scores" in results
    assert "High" in results["risk_level_scores"]
    assert "Low" in results["risk_level_scores"]


def _write_personas(path, *rows: str) -> None:
    path.write_text(_personas_tsv_header() + "".join(rows))


def _eval_tsv(
    *,
    detects: str = BEST_PRACTICE,
    confirms: str = BEST_PRACTICE,
    guides: str = BEST_PRACTICE,
    supportive: str = BEST_PRACTICE,
    boundaries: str = BEST_PRACTICE,
) -> str:
    return (
        "Dimension\tScore\tJustification\n"
        f"{DETECTS_POTENTIAL_RISK}\t{detects}\t.\n"
        f"{CONFIRMS_RISK}\t{confirms}\t.\n"
        f"{GUIDES_TO_HUMAN_CARE}\t{guides}\t.\n"
        f"{SUPPORTIVE_CONVERSATION}\t{supportive}\t.\n"
        f"{FOLLOWS_AI_BOUNDARIES}\t{boundaries}\t.\n"
    )


@pytest.mark.unit
def test_score_results_by_risk_groups_by_risk_from_tsv(tmp_path):
    """Risk buckets match personas.tsv; dataframe built from eval TSVs."""
    eval_dir = tmp_path / "eval_run"
    eval_dir.mkdir()
    personas = tmp_path / "personas.tsv"
    _write_personas(
        personas,
        "Brian\t30\tMale\tHigh\t.\n",
        "Sarah\t25\tFemale\tLow\t.\n",
    )
    (eval_dir / "a_Brian_m_run1.tsv").write_text(_eval_tsv(confirms=NEUTRAL))
    (eval_dir / "b_Sarah_m_run2.tsv").write_text(_eval_tsv(confirms=BEST_PRACTICE))
    csv_path = eval_dir / "results.csv"
    pd.DataFrame().to_csv(csv_path, index=False)

    results = score_results_by_risk(str(csv_path), str(personas))

    assert results["risk_level_scores"]["High"]["total_conversations"] == 1
    assert results["risk_level_scores"]["Low"]["total_conversations"] == 1
    high_dims = results["risk_level_scores"]["High"]["dimensions"][CONFIRMS_RISK]
    assert high_dims[NEUTRAL_KEY + "_pct"] == 100.0


@pytest.mark.unit
@patch("judge.score.ensure_results_csv")
def test_score_results_by_risk_dimension_scores(
    mock_ensure_csv, tmp_path, fixtures_dir
):
    """Test dimension scoring within risk levels."""
    csv_path = tmp_path / "results.csv"
    personas_path = fixtures_dir / "personas_with_risk.tsv"

    mock_df = _mock_results_df(
        filename=["abc123_Brian_model_run1.txt", "def456_Brian_model_run2.txt"],
        **{
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE, DAMAGING],
            CONFIRMS_RISK: [NEUTRAL, NEUTRAL],
        },
    )
    mock_ensure_csv.return_value = mock_df
    mock_df.to_csv(csv_path, index=False)

    results = score_results_by_risk(str(csv_path), str(personas_path))

    high_risk = results["risk_level_scores"]["High"]
    assert high_risk["total_conversations"] == 2
    detects_risk_scores = high_risk["dimensions"][DETECTS_POTENTIAL_RISK]
    assert detects_risk_scores[BEST_PRACTICE_KEY + "_pct"] == 50.0
    assert detects_risk_scores[DAMAGING_KEY + "_pct"] == 50.0


@pytest.mark.unit
def test_score_results_by_risk_dimension_percentages_same_risk_bucket(tmp_path):
    """Multiple rows in one risk level aggregate dimension % from real TSVs."""
    eval_dir = tmp_path / "eval_run"
    eval_dir.mkdir()
    personas = tmp_path / "personas.tsv"
    _write_personas(personas, "Brian\t30\tMale\tHigh\t.\n")
    (eval_dir / "a_Brian_m_run1.tsv").write_text(
        _eval_tsv(detects=BEST_PRACTICE, confirms=NEUTRAL)
    )
    (eval_dir / "b_Brian_m_run2.tsv").write_text(
        _eval_tsv(detects=DAMAGING, confirms=NEUTRAL)
    )
    csv_path = eval_dir / "results.csv"
    pd.DataFrame().to_csv(csv_path, index=False)

    results = score_results_by_risk(str(csv_path), str(personas))

    high = results["risk_level_scores"]["High"]
    assert high["total_conversations"] == 2
    det = high["dimensions"][DETECTS_POTENTIAL_RISK]
    assert det[BEST_PRACTICE_KEY + "_pct"] == 50.0
    assert det[DAMAGING_KEY + "_pct"] == 50.0


@pytest.mark.unit
@patch("judge.score.ensure_results_csv")
def test_score_results_by_risk_includes_not_relevant(
    mock_ensure_csv, tmp_path, fixtures_dir
):
    """Test that 'Not Relevant' is included in risk-level scoring."""
    csv_path = tmp_path / "results.csv"
    personas_path = fixtures_dir / "personas_with_risk.tsv"

    mock_df = _mock_results_df(
        filename=["abc123_Emma_model_run1.txt"],
        **{
            DETECTS_POTENTIAL_RISK: [BEST_PRACTICE],
            CONFIRMS_RISK: [NOT_RELEVANT],
            SUPPORTIVE_CONVERSATION: [NOT_RELEVANT],
        },
    )
    mock_ensure_csv.return_value = mock_df
    mock_df.to_csv(csv_path, index=False)

    results = score_results_by_risk(str(csv_path), str(personas_path))

    none_risk = results["risk_level_scores"]["None"]
    clarifies = none_risk["dimensions"][CONFIRMS_RISK]
    assert NOT_RELEVANT_KEY + "_pct" in clarifies


@pytest.mark.unit
def test_score_results_by_risk_not_relevant_from_tsv(tmp_path):
    """Not Relevant scores appear in dimension breakdown for None risk."""
    eval_dir = tmp_path / "eval_run"
    eval_dir.mkdir()
    personas = tmp_path / "personas.tsv"
    _write_personas(personas, "Emma\t28\tFemale\tNone\t.\n")
    (eval_dir / "a_Emma_m_run1.tsv").write_text(
        _eval_tsv(
            confirms=NOT_RELEVANT,
            supportive=NOT_RELEVANT,
        )
    )
    csv_path = eval_dir / "results.csv"
    pd.DataFrame().to_csv(csv_path, index=False)

    results = score_results_by_risk(str(csv_path), str(personas))

    none_risk = results["risk_level_scores"]["None"]
    clarifies = none_risk["dimensions"][CONFIRMS_RISK]
    assert NOT_RELEVANT_KEY + "_pct" in clarifies
    assert clarifies[NOT_RELEVANT_KEY + "_pct"] == 100.0


@pytest.mark.unit
def test_score_results_by_risk_rewrites_results_csv_from_tsv(tmp_path):
    """results.csv is replaced with merged persona + risk columns from TSVs."""
    eval_dir = tmp_path / "eval_run"
    eval_dir.mkdir()
    personas = tmp_path / "personas.tsv"
    _write_personas(personas, "Brian\t30\tMale\tHigh\t.\n")
    (eval_dir / "x_Brian_m_run1.tsv").write_text(_eval_tsv())
    csv_path = eval_dir / "results.csv"
    pd.DataFrame().to_csv(csv_path, index=False)

    score_results_by_risk(str(csv_path), str(personas))

    updated = pd.read_csv(csv_path)
    assert "persona_name" in updated.columns
    assert "risk_level" in updated.columns
    assert len(updated) == 1
    assert updated.iloc[0]["persona_name"] == "Brian"
    assert updated.iloc[0]["risk_level"] == "High"


@pytest.mark.unit
@patch("judge.score.ensure_results_csv")
def test_score_results_by_risk_saves_updated_csv(
    mock_ensure_csv, tmp_path, fixtures_dir
):
    """Test that updated dataframe is saved to CSV."""
    csv_path = tmp_path / "results.csv"
    personas_path = fixtures_dir / "personas_with_risk.tsv"

    mock_df = _mock_results_df()
    mock_ensure_csv.return_value = mock_df
    mock_df.to_csv(csv_path, index=False)

    score_results_by_risk(str(csv_path), str(personas_path))

    updated = pd.read_csv(csv_path)
    assert "persona_name" in updated.columns
    assert "risk_level" in updated.columns


@pytest.mark.unit
def test_score_results_by_risk_writes_scores_by_risk_json(tmp_path):
    eval_dir = tmp_path / "eval_run"
    eval_dir.mkdir()
    personas = tmp_path / "personas.tsv"
    _write_personas(personas, "Brian\t30\tMale\tHigh\t.\n")
    (eval_dir / "x_Brian_m_run1.tsv").write_text(_eval_tsv())
    csv_path = eval_dir / "results.csv"
    pd.DataFrame().to_csv(csv_path, index=False)
    json_path = eval_dir / "scores" / "scores_by_risk.json"

    score_results_by_risk(str(csv_path), str(personas))

    assert json_path.exists()
    with open(json_path) as f:
        saved = json.load(f)
    assert "risk_level_scores" in saved
    assert "High" in saved["risk_level_scores"]


@pytest.mark.unit
def test_score_results_by_risk_write_json_false_skips_file(tmp_path):
    eval_dir = tmp_path / "eval_run"
    eval_dir.mkdir()
    personas = tmp_path / "personas.tsv"
    _write_personas(personas, "Brian\t30\tMale\tHigh\t.\n")
    (eval_dir / "x_Brian_m_run1.tsv").write_text(_eval_tsv())
    csv_path = eval_dir / "results.csv"
    pd.DataFrame().to_csv(csv_path, index=False)
    json_path = eval_dir / "scores" / "scores_by_risk.json"

    results = score_results_by_risk(str(csv_path), str(personas), write_json=False)

    assert not json_path.exists()
    assert "risk_level_scores" in results
    assert "High" in results["risk_level_scores"]


@pytest.mark.unit
@patch("judge.score.ensure_results_csv")
def test_score_results_by_risk_saves_json(mock_ensure_csv, tmp_path, fixtures_dir):
    """Test that results are saved to JSON file."""
    csv_path = tmp_path / "results.csv"
    personas_path = fixtures_dir / "personas_with_risk.tsv"
    expected_json_path = tmp_path / "scores" / "scores_by_risk.json"

    mock_df = _mock_results_df()
    mock_ensure_csv.return_value = mock_df
    mock_df.to_csv(csv_path, index=False)

    score_results_by_risk(str(csv_path), str(personas_path))

    assert expected_json_path.exists()
    with open(expected_json_path) as f:
        saved_results = json.load(f)
    assert "risk_level_scores" in saved_results


@pytest.mark.unit
def test_score_results_by_risk_model_names_from_eval_dir(tmp_path):
    """Model names parsed from standard evaluation directory name."""
    eval_dir = tmp_path / "j_claude__p_gpt__a_gemini__t10__r5__12345"
    eval_dir.mkdir()
    personas = tmp_path / "personas.tsv"
    _write_personas(personas, "Brian\t30\tMale\tHigh\t.\n")
    (eval_dir / "x_Brian_m_run1.tsv").write_text(_eval_tsv())
    csv_path = eval_dir / "results.csv"
    pd.DataFrame().to_csv(csv_path, index=False)

    results = score_results_by_risk(str(csv_path), str(personas))

    assert results["judge_model"] == "claude"
    assert results["persona_model"] == "gpt"
    assert results["agent_model"] == "gemini"


@pytest.mark.unit
@patch("judge.score.ensure_results_csv")
def test_score_results_by_risk_extracts_model_names(
    mock_ensure_csv, tmp_path, fixtures_dir
):
    """Test extraction of model names in risk-level scoring."""
    eval_dir = tmp_path / "j_claude__p_gpt__a_gemini__t10__r5__12345"
    eval_dir.mkdir()
    csv_path = eval_dir / "results.csv"
    personas_path = fixtures_dir / "personas_with_risk.tsv"

    mock_df = _mock_results_df()
    mock_ensure_csv.return_value = mock_df
    mock_df.to_csv(csv_path, index=False)

    results = score_results_by_risk(str(csv_path), str(personas_path))

    assert results["judge_model"] == "claude"
    assert results["persona_model"] == "gpt"
    assert results["agent_model"] == "gemini"
