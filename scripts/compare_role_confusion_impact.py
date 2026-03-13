#!/usr/bin/env python3
"""
Compare VERA scores for each provider agent with and without role confusion conversations.

This script loads role_confusion_results.csv and concatenated_heor_paper2_results.csv,
merges them, and calculates VERA scores for each provider:
- Using all conversations
- Using only conversations where role_confusion=True

Usage:
    python scripts/compare_role_confusion_impact.py
    python scripts/compare_role_confusion_impact.py -o results/role_confusion_comparison.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency, pointbiserialr  # type: ignore
except ImportError:
    print("⚠️  Warning: scipy not available, correlation analysis will be limited")
    chi2_contingency = None  # type: ignore
    pointbiserialr = None  # type: ignore

matplotlib.use("Agg")  # Use non-interactive backend

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from judge.score_utils import (
    BG_COLOR,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    DIMENSIONS,
    TEXT_COLOR,
    calculate_scores_from_df,
)

# ── Model display-name mapping (kept in sync with analyze_score_variability_heor_paper2.py) ──
_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "gemini 3 1": "Gemini 3.1 Pro Preview",
    "gpt 5 2": "GPT 5.2",
    "grok 4": "Grok 4",
    "opus 4 6": "Opus 4.6",
    "opus 4 5": "Opus 4.5",
    "gpt-4o": "GPT-4o",
    "claude-sonnet-4-5-20250929": "Sonnet 4.5",
}


def display_name(raw: str) -> str:
    """Return a human-readable model name for a raw internal model string."""
    key = raw.strip().lower()
    if key in _MODEL_DISPLAY_NAMES:
        return _MODEL_DISPLAY_NAMES[key]
    best = max(
        (_MODEL_DISPLAY_NAMES[k] for k in _MODEL_DISPLAY_NAMES if k in key),
        key=lambda v: len(v),
        default=None,
    )
    return best if best else raw


def load_persona_metadata(personas_tsv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load persona metadata including risk, disclosure, and other attributes.

    Args:
        personas_tsv_path: Path to personas.tsv

    Returns:
        Dict mapping persona name to metadata dict with various attributes
    """
    df = pd.read_csv(personas_tsv_path, sep="\t", keep_default_na=False)

    metadata = {}
    for _, row in df.iterrows():
        name = str(row["Name"]).strip()

        # Risk level
        risk_col = row.get("Short Current Suicide Risk Level") or row.get(
            "Current Suicide Risk Level", "Unknown"
        )
        risk_level = str(risk_col).strip()

        # Disclosure level (normalize)
        disclosure_level = str(row.get("Disclosure of Suicide Risk", "Unknown")).strip()
        if "Not applicable" in disclosure_level or disclosure_level == "Not applicable":
            disclosure_level = "Not applicable"
        elif "High" in disclosure_level:
            disclosure_level = "High"
        elif "Moderate" in disclosure_level:
            disclosure_level = "Moderate"
        elif "Low" in disclosure_level:
            disclosure_level = "Low"
        else:
            disclosure_level = "Unknown"

        # Primary Communication Style (normalize)
        comm_style = str(row.get("Primary Communication Style", "Unknown")).strip()
        if "Structured" in comm_style:
            comm_style = "Structured"
        elif "Disorganized" in comm_style:
            comm_style = "Disorganized"
        else:
            comm_style = "Unknown"

        # Reaction to Chatbot Responses (normalize)
        reaction = str(row.get("Reaction to Chatbot Responses", "Unknown")).strip()
        if "Resistant" in reaction:
            reaction = "Resistant"
        elif "Disengaged" in reaction:
            reaction = "Disengaged"
        elif "Engaged" in reaction:
            reaction = "Engaged"
        else:
            reaction = "Unknown"

        # Mental Health Stigma (normalize)
        stigma = str(row.get("Mental Health Stigma", "Unknown")).strip()
        if "High" in stigma:
            stigma = "High"
        elif "Moderate" in stigma:
            stigma = "Moderate"
        elif "Low" in stigma:
            stigma = "Low"
        else:
            stigma = "Unknown"

        # Social Isolation (normalize)
        isolation = str(row.get("Social Isolation", "Unknown")).strip()
        if "High" in isolation:
            isolation = "High"
        elif "Moderate" in isolation:
            isolation = "Moderate"
        elif "Low" in isolation:
            isolation = "Low"
        else:
            isolation = "Unknown"

        # Treatment Engagement / Access (normalize)
        treatment = str(row.get("Treatment Engagement / Access", "Unknown")).strip()
        if "None" in treatment or "never sought" in treatment.lower():
            treatment = "None"
        elif "Limited" in treatment:
            treatment = "Limited"
        elif "Intermittent" in treatment:
            treatment = "Intermittent"
        elif "Active" in treatment or "engaged" in treatment.lower():
            treatment = "Active"
        else:
            treatment = "Unknown"

        metadata[name] = {
            "risk_level": risk_level,
            "disclosure_level": disclosure_level,
            "communication_style": comm_style,
            "reaction_to_chatbot": reaction,
            "mental_health_stigma": stigma,
            "social_isolation": isolation,
            "treatment_engagement": treatment,
        }

    return metadata


def add_persona_metadata_to_df(
    df: pd.DataFrame, personas_tsv_path: Path
) -> pd.DataFrame:
    """
    Add persona metadata columns to DataFrame using persona_name.

    Args:
        df: DataFrame with persona_name column
        personas_tsv_path: Path to personas.tsv file

    Returns:
        DataFrame with persona metadata columns added
    """
    if "persona_name" not in df.columns:
        print("⚠️  Warning: persona_name column not found, skipping metadata addition")
        return df

    persona_metadata = load_persona_metadata(personas_tsv_path)

    # Initialize lists for all metadata columns
    metadata_cols = {
        "risk_level": [],
        "disclosure_level": [],
        "communication_style": [],
        "reaction_to_chatbot": [],
        "mental_health_stigma": [],
        "social_isolation": [],
        "treatment_engagement": [],
    }

    for persona_name in df["persona_name"]:
        if pd.isna(persona_name):
            for col in metadata_cols:
                metadata_cols[col].append("Unknown")
        else:
            meta = persona_metadata.get(str(persona_name).strip(), {})
            for col in metadata_cols:
                metadata_cols[col].append(meta.get(col, "Unknown"))

    df = df.copy()
    for col, values in metadata_cols.items():
        df[col] = values

    print(
        f"✅ Added persona metadata columns to {len(df)} rows: {', '.join(metadata_cols.keys())}"
    )
    return df


def load_and_merge_data(
    role_confusion_path: Path,
    concatenated_path: Path,
    personas_tsv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load role_confusion_results.csv and concatenated_heor_paper2_results.csv and merge them.

    Args:
        role_confusion_path: Path to role_confusion_results.csv
        concatenated_path: Path to concatenated_heor_paper2_results.csv

    Returns:
        Merged DataFrame with all columns from both files
    """
    if not role_confusion_path.exists():
        raise FileNotFoundError(
            f"Role confusion results file not found: {role_confusion_path}"
        )

    if not concatenated_path.exists():
        raise FileNotFoundError(
            f"Concatenated results file not found: {concatenated_path}"
        )

    # Load both CSVs
    role_confusion_df = pd.read_csv(role_confusion_path)
    concatenated_df = pd.read_csv(concatenated_path)

    print(f"✅ Loaded {len(role_confusion_df)} rows from {role_confusion_path}")
    print(f"✅ Loaded {len(concatenated_df)} rows from {concatenated_path}")

    # Merge on filename (common key)
    # Normalize filename if needed (remove path components)
    role_confusion_df["filename_clean"] = (
        role_confusion_df["filename"].str.split("/").str[-1]
    )
    concatenated_df["filename_clean"] = (
        concatenated_df["filename"].str.split("/").str[-1]
    )

    # Merge
    merged_df = concatenated_df.merge(
        role_confusion_df[["filename_clean", "role_confusion"]],
        on="filename_clean",
        how="inner",
    )

    print(f"✅ Merged to {len(merged_df)} rows")

    # Check role_confusion column
    if "role_confusion" not in merged_df.columns:
        raise ValueError("role_confusion column not found after merge")

    # Convert role_confusion to boolean if needed
    if merged_df["role_confusion"].dtype == "object":
        merged_df["role_confusion"] = merged_df["role_confusion"].map(
            lambda x: True
            if x in ("True", True)
            else False
            if x in ("False", False)
            else bool(x)
        )

    # Add persona metadata if personas_tsv_path is provided
    if personas_tsv_path and personas_tsv_path.exists():
        merged_df = add_persona_metadata_to_df(merged_df, personas_tsv_path)

    return merged_df


def analyze_role_confusion_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlations between role_confusion and other variables in the dataset.

    Args:
        df: DataFrame with role_confusion column and other variables

    Returns:
        DataFrame with correlation results: variable_name, correlation_type,
        correlation_value, p_value, statistical_significance
    """
    if "role_confusion" not in df.columns:
        print("❌ Error: role_confusion column not found")
        return pd.DataFrame()

    # Convert role_confusion to numeric (0/1)
    role_confusion_numeric = df["role_confusion"].astype(int)

    # Columns to exclude from correlation analysis
    exclude_cols = {
        "filename",
        "filename_clean",
        "run_id",
        "conversation_path",
        "role_confusion",
        "judge_id",
        "judge_instance",
    }
    # Also exclude question_id and reasoning columns
    exclude_patterns = [
        "_question_id",
        "_reasoning",
        "yes_question_id",
        "yes_reasoning",
    ]

    # Identify categorical and continuous variables
    categorical_vars = []
    continuous_vars = []

    for col in df.columns:
        if col in exclude_cols:
            continue
        if any(pattern in col for pattern in exclude_patterns):
            continue

        # Skip if all values are NaN
        col_isna = df[col].isna()
        if isinstance(col_isna, pd.Series) and col_isna.all():
            continue

        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's actually categorical (few unique values relative to length)
            n_unique = df[col].nunique()
            threshold = len(df) * 0.1
            if n_unique <= 10 and n_unique < threshold:
                categorical_vars.append(col)
            else:
                continuous_vars.append(col)
        else:
            categorical_vars.append(col)

    print(f"   Found {len(categorical_vars)} categorical variables")
    print(f"   Found {len(continuous_vars)} continuous variables")
    if len(categorical_vars) > 0:
        print(f"   Categorical examples: {categorical_vars[:5]}")
    if len(continuous_vars) > 0:
        print(f"   Continuous examples: {continuous_vars[:5]}")

    # Check if dimension columns are included
    dimension_cols = [col for col in df.columns if col in DIMENSIONS]
    included_dims = [col for col in dimension_cols if col in categorical_vars]
    print(f"   Dimension columns in data: {dimension_cols}")
    print(f"   Dimension columns included in analysis: {included_dims}")

    results = []

    # Analyze categorical variables using Cramér's V
    if chi2_contingency is None:
        print(
            "⚠️  Warning: scipy.stats.chi2_contingency not available - skipping categorical analysis"
        )
    else:
        skipped_count = 0
        skipped_reasons = {}
        for var in categorical_vars:
            if var not in df.columns:
                skipped_reasons[var] = "not in columns"
                skipped_count += 1
                continue

            # Create contingency table
            try:
                contingency = pd.crosstab(df[var], role_confusion_numeric)
            except Exception as e:
                print(f"⚠️  Warning: Could not create contingency table for {var}: {e}")
                skipped_reasons[var] = f"contingency error: {e}"
                skipped_count += 1
                continue

            # Skip if table is too small or has issues
            if contingency.size == 0 or contingency.sum().sum() == 0:
                skipped_reasons[var] = "empty contingency table"
                skipped_count += 1
                continue

            # Check if there's variation in role_confusion for this variable
            if contingency.shape[1] < 2:  # Only one value of role_confusion
                skipped_reasons[var] = "no variation in role_confusion"
                skipped_count += 1
                continue

            # Calculate chi-square test
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
            except Exception as e:
                print(f"⚠️  Warning: Could not calculate chi-square for {var}: {e}")
                continue

            # Calculate Cramér's V
            n = contingency.sum().sum()
            min_dim = min(contingency.shape[0], contingency.shape[1])
            cramers_v = np.sqrt(chi2 / (n * (min_dim - 1))) if min_dim > 1 else 0.0

            # Determine statistical significance
            p_val: float = float(p_value)  # type: ignore[arg-type]
            if p_val < 0.001:
                significance = "***"
            elif p_val < 0.01:
                significance = "**"
            elif p_val < 0.05:
                significance = "*"
            else:
                significance = ""

            results.append(
                {
                    "variable_name": var,
                    "correlation_type": "Cramér's V",
                    "correlation_value": cramers_v,
                    "p_value": p_val,
                    "statistical_significance": significance,
                }
            )
        if skipped_count > 0:
            print(f"   Skipped {skipped_count} categorical variables")
            # Show first 10 skipped variables and reasons
            for var, reason in list(skipped_reasons.items())[:10]:
                print(f"      - {var}: {reason}")
            if len(skipped_reasons) > 10:
                print(f"      ... and {len(skipped_reasons) - 10} more")

    # Analyze continuous variables using point-biserial correlation
    if pointbiserialr is None:
        print(
            "⚠️  Warning: scipy.stats.pointbiserialr not available - skipping continuous analysis"
        )
    else:
        skipped_continuous = []
        for var in continuous_vars:
            if var not in df.columns:
                skipped_continuous.append((var, "not in columns"))
                continue

            # Remove NaN values for both variables
            var_series = df[var]
            var_isna = var_series.isna()
            rc_isna = role_confusion_numeric.isna()
            mask = ~(var_isna | rc_isna)
            var_clean = var_series[mask]
            rc_clean = role_confusion_numeric[mask]

            if len(var_clean) < 2:
                skipped_continuous.append((var, "insufficient data after removing NaN"))
                continue

            try:
                correlation, p_value = pointbiserialr(var_clean, rc_clean)
            except Exception as e:
                print(
                    f"⚠️  Warning: Could not calculate point-biserial correlation for {var}: {e}"
                )
                skipped_continuous.append((var, f"calculation error: {e}"))
                continue

            # Determine statistical significance
            p_val: float = float(p_value)  # type: ignore[arg-type]
            if p_val < 0.001:
                significance = "***"
            elif p_val < 0.01:
                significance = "**"
            elif p_val < 0.05:
                significance = "*"
            else:
                significance = ""

            results.append(
                {
                    "variable_name": var,
                    "correlation_type": "Point-biserial",
                    "correlation_value": correlation,
                    "p_value": p_val,
                    "statistical_significance": significance,
                }
            )
        if skipped_continuous:
            print(f"   Skipped {len(skipped_continuous)} continuous variables")
            for var, reason in skipped_continuous[:5]:
                print(f"      - {var}: {reason}")

    # Create DataFrame and sort by absolute correlation value
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df["abs_correlation"] = results_df["correlation_value"].abs()
        results_df = results_df.sort_values("abs_correlation", ascending=False)
        results_df = results_df.drop(columns=["abs_correlation"])
        print(f"   Calculated correlations for {len(results_df)} variables")
    else:
        print(
            "   ⚠️  No correlations calculated - check variable identification and scipy availability"
        )

    return results_df


def analyze_role_confusion_rates_by_value(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    For each statistically significant variable, compute the role confusion rate per value
    and compare it to the overall rate to identify which values are positively associated
    with role confusion occurring.

    Args:
        df: DataFrame with role_confusion column and other variables
        corr_df: Correlation results DataFrame from analyze_role_confusion_correlations
        p_threshold: Only include variables with p_value below this threshold

    Returns:
        DataFrame with columns: variable, value, confusion_rate, confused_count,
        total_count, overall_rate, rate_vs_overall, direction
    """
    if "role_confusion" not in df.columns or corr_df.empty:
        return pd.DataFrame()

    overall_rate: float = float(df["role_confusion"].mean())

    # Filter to only statistically significant variables
    sig_vars = corr_df[corr_df["p_value"] < p_threshold]["variable_name"].tolist()

    results = []
    for var in sig_vars:
        if var not in df.columns:
            continue

        try:
            group_stats = (
                df.groupby(var)["role_confusion"]
                .agg(["mean", "sum", "count"])
                .reset_index()
            )
            group_stats.columns = pd.Index(
                [var, "confusion_rate", "confused_count", "total_count"]
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not compute rates for {var}: {e}")
            continue

        for _, row in group_stats.iterrows():
            rate: float = float(row["confusion_rate"])
            results.append(
                {
                    "variable": var,
                    "value": str(row[var]),
                    "confusion_rate": round(rate, 4),
                    "confused_count": int(row["confused_count"]),
                    "total_count": int(row["total_count"]),
                    "overall_rate": round(overall_rate, 4),
                    "rate_vs_overall": round(rate - overall_rate, 4),
                    "direction": "higher" if rate > overall_rate else "lower",
                }
            )

    if not results:
        return pd.DataFrame()

    rates_df = pd.DataFrame(results)
    # Sort by variable (preserving corr_df order), then by rate descending within each var
    var_order: dict[str, int] = {v: i for i, v in enumerate(sig_vars)}
    rates_df["_var_order"] = rates_df["variable"].map(
        lambda v: var_order.get(str(v), 999)
    )
    rates_df = rates_df.sort_values(
        ["_var_order", "confusion_rate"], ascending=[True, False]
    ).drop(columns=["_var_order"])
    return rates_df


def bootstrap_role_confusion_diff(
    group_df: pd.DataFrame,
    n_iterations: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Bootstrap the difference in aggregate VERA score between all conversations and
    no-role-confusion conversations using paired resampling.

    Because "no role confusion" is a subset of "all", each bootstrap iteration
    resamples the same conversations and computes both scores from that sample,
    preserving the dependency structure.

    Args:
        group_df: DataFrame for one group (e.g. one provider), must have
                  role_confusion and filename columns
        n_iterations: Number of bootstrap samples
        seed: Random seed

    Returns:
        Array of bootstrapped (score_no_rc - score_all) differences
    """
    if "filename" not in group_df.columns or "role_confusion" not in group_df.columns:
        return np.array([])

    rng = np.random.default_rng(seed)
    conversations = [
        df for _, df in group_df.groupby("filename") if isinstance(df, pd.DataFrame)
    ]
    n_convs = len(conversations)
    if n_convs == 0:
        return np.array([])

    diffs: list[float] = []
    for _ in range(n_iterations):
        indices = rng.integers(0, n_convs, size=n_convs)
        sample_df = pd.concat([conversations[i] for i in indices], ignore_index=True)

        score_all = calculate_scores_from_df(sample_df).get("overall_score", 0.0)

        no_rc_df = sample_df[~sample_df["role_confusion"]].copy()
        assert isinstance(no_rc_df, pd.DataFrame)
        if len(no_rc_df) == 0:
            continue
        score_no_rc = calculate_scores_from_df(no_rc_df).get("overall_score", 0.0)

        diffs.append(score_no_rc - score_all)

    return np.array(diffs)


def test_role_confusion_significance(
    group_df: pd.DataFrame,
    n_iterations: int = 1000,
) -> dict[str, object]:
    """
    Test whether removing role-confusion conversations changes the aggregate VERA
    score significantly, using paired bootstrap resampling.

    Returns a dict with: p_value, significance (stars), boot_diff_mean,
    ci_lower, ci_upper, n_convs_all, n_convs_no_rc.
    """
    n_all = (
        group_df["filename"].nunique()
        if "filename" in group_df.columns
        else len(group_df)
    )
    n_no_rc = (
        int((~group_df["role_confusion"]).sum())
        if "role_confusion" in group_df.columns
        else 0
    )

    base: dict[str, object] = {
        "p_value": None,
        "significance": "",
        "boot_diff_mean": None,
        "ci_lower": None,
        "ci_upper": None,
        "n_convs_all": n_all,
        "n_convs_no_rc": n_no_rc,
    }

    if n_no_rc == 0 or n_all == 0:
        return base

    print(
        f"      Bootstrap resampling ({n_iterations} iterations)…", end=" ", flush=True
    )
    boot_diffs = bootstrap_role_confusion_diff(group_df, n_iterations=n_iterations)
    print("done")

    if len(boot_diffs) < 10:
        return base

    no_rc_observed = group_df[~group_df["role_confusion"]].copy()
    assert isinstance(no_rc_observed, pd.DataFrame)
    observed_diff = calculate_scores_from_df(no_rc_observed).get(
        "overall_score", 0.0
    ) - calculate_scores_from_df(group_df).get("overall_score", 0.0)

    if observed_diff >= 0:
        p_value = float(2 * np.mean(boot_diffs <= 0))
    else:
        p_value = float(2 * np.mean(boot_diffs >= 0))
    p_value = min(p_value, 1.0)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "ns"

    return {
        "p_value": round(p_value, 4),
        "significance": sig,
        "boot_diff_mean": round(float(np.mean(boot_diffs)), 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "n_convs_all": n_all,
        "n_convs_no_rc": n_no_rc,
    }


def add_significance_to_role_confusion_comparison(
    comparison_df: pd.DataFrame,
    df: pd.DataFrame,
    group_cols: list[str],
    n_iterations: int = 1000,
) -> pd.DataFrame:
    """
    For each row in comparison_df, run a paired bootstrap test comparing the aggregate
    VERA score for all conversations vs no-role-confusion conversations.

    Adds columns: p_value, significance, boot_diff_mean, ci_lower, ci_upper,
    n_convs_all, n_convs_no_rc.
    """
    sig_rows: list[dict[str, object]] = []
    for _, row in comparison_df.iterrows():
        mask = pd.Series([True] * len(df), index=df.index)
        for col in group_cols:
            if col in df.columns and col in comparison_df.columns:
                mask = mask & (df[col] == row[col])

        group_df = df[mask].copy()
        assert isinstance(group_df, pd.DataFrame)

        label = ", ".join(str(row[c]) for c in group_cols if c in row)
        print(f"   Significance test for: {label}")
        sig_rows.append(
            test_role_confusion_significance(group_df, n_iterations=n_iterations)
        )

    sig_df = pd.DataFrame(sig_rows)
    return pd.concat(
        [comparison_df.reset_index(drop=True), sig_df.reset_index(drop=True)], axis=1
    )


def compare_role_confusion_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare VERA scores for each provider with all conversations vs only role_confusion=True.

    Args:
        df: DataFrame with merged evaluation results and role_confusion column

    Returns:
        DataFrame with comparison results
    """
    results = []

    # Group by provider_llm
    if "provider_llm" not in df.columns:
        print("❌ Error: Missing required column (provider_llm)")
        return pd.DataFrame()

    for provider in df["provider_llm"].unique():
        if pd.isna(provider):
            continue

        provider_filter = df["provider_llm"] == provider
        provider_df = df.loc[provider_filter].copy()
        if not isinstance(provider_df, pd.DataFrame):
            continue

        print(f"\n📊 Analyzing provider: {provider}")
        print(f"   Total conversations: {len(provider_df)}")

        # Calculate scores with all conversations
        scores_all = calculate_scores_from_df(provider_df)

        # Filter to only role_confusion=False (no role confusion)
        no_rc_filter = ~provider_df["role_confusion"]
        no_role_confusion_df = provider_df.loc[no_rc_filter].copy()
        if not isinstance(no_role_confusion_df, pd.DataFrame):
            no_role_confusion_df = pd.DataFrame()
        print(f"   No role confusion (False): {len(no_role_confusion_df)}")

        if len(no_role_confusion_df) > 0:
            scores_no_rc = calculate_scores_from_df(no_role_confusion_df)
        else:
            # No conversations without role confusion, set scores to 0
            scores_no_rc = {
                "overall_score": 0.0,
                "dimension_scores": {dim: {"vera_score": 0.0} for dim in DIMENSIONS},
            }

        # Calculate differences (no_rc - all)
        overall_diff = scores_no_rc.get("overall_score", 0.0) - scores_all.get(
            "overall_score", 0.0
        )

        result_row = {
            "provider_llm": provider,
            "num_evaluations_all": len(provider_df),
            "num_evaluations_no_role_confusion": len(no_role_confusion_df),
            "overall_score_all": scores_all.get("overall_score", 0.0),
            "overall_score_no_role_confusion": scores_no_rc.get("overall_score", 0.0),
            "overall_score_difference": overall_diff,
        }

        # Add dimension scores
        dimension_scores_all = scores_all.get("dimension_scores", {})
        dimension_scores_no_rc = scores_no_rc.get("dimension_scores", {})

        for dim in DIMENSIONS:
            score_all = (
                dimension_scores_all.get(dim, {}).get("vera_score", 0.0)
                if dim in dimension_scores_all
                else 0.0
            )
            score_no_rc = (
                dimension_scores_no_rc.get(dim, {}).get("vera_score", 0.0)
                if dim in dimension_scores_no_rc
                else 0.0
            )

            result_row[f"{dim}_score_all"] = score_all
            result_row[f"{dim}_score_no_role_confusion"] = score_no_rc
            result_row[f"{dim}_score_difference"] = score_no_rc - score_all

        results.append(result_row)

    return pd.DataFrame(results)


def compare_role_confusion_by_provider_and_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare VERA scores for each combination of provider and user agent,
    with all conversations vs only role_confusion=True.

    Args:
        df: DataFrame with merged evaluation results and role_confusion column

    Returns:
        DataFrame with results for each provider/user combination
    """
    results = []

    # Group by provider_llm and user_llm
    if "provider_llm" not in df.columns or "user_llm" not in df.columns:
        print("❌ Error: Missing required columns (provider_llm, user_llm)")
        return pd.DataFrame()

    for group_key, group_df_raw in df.groupby(
        ["provider_llm", "user_llm"], dropna=False
    ):
        # Ensure group_df is a DataFrame
        if isinstance(group_df_raw, pd.Series):
            continue
        group_df: pd.DataFrame = group_df_raw

        # Extract group key values
        if isinstance(group_key, tuple):
            provider_llm = str(group_key[0]) if len(group_key) > 0 else ""
            user_llm = str(group_key[1]) if len(group_key) > 1 else ""
        else:
            provider_llm = str(group_key)
            user_llm = ""

        print(f"\n📊 Analyzing: provider={provider_llm}, user={user_llm}")
        print(f"   Total conversations: {len(group_df)}")

        # Calculate scores with all conversations
        scores_all = calculate_scores_from_df(group_df)

        # Filter to only role_confusion=False (no role confusion)
        no_rc_filter = ~group_df["role_confusion"]
        no_role_confusion_df = group_df.loc[no_rc_filter].copy()
        if not isinstance(no_role_confusion_df, pd.DataFrame):
            no_role_confusion_df = pd.DataFrame()
        print(f"   No role confusion (False): {len(no_role_confusion_df)}")

        if len(no_role_confusion_df) > 0:
            scores_no_rc = calculate_scores_from_df(no_role_confusion_df)
        else:
            # No conversations without role confusion, set scores to 0
            scores_no_rc = {
                "overall_score": 0.0,
                "dimension_scores": {dim: {"vera_score": 0.0} for dim in DIMENSIONS},
            }

        # Calculate differences (no_rc - all)
        overall_diff = scores_no_rc.get("overall_score", 0.0) - scores_all.get(
            "overall_score", 0.0
        )

        result_row = {
            "provider_llm": provider_llm,
            "user_llm": user_llm,
            "num_evaluations_all": len(group_df),
            "num_evaluations_no_role_confusion": len(no_role_confusion_df),
            "overall_score_all": scores_all.get("overall_score", 0.0),
            "overall_score_no_role_confusion": scores_no_rc.get("overall_score", 0.0),
            "overall_score_difference": overall_diff,
        }

        # Add dimension scores
        dimension_scores_all = scores_all.get("dimension_scores", {})
        dimension_scores_no_rc = scores_no_rc.get("dimension_scores", {})

        for dim in DIMENSIONS:
            score_all = (
                dimension_scores_all.get(dim, {}).get("vera_score", 0.0)
                if dim in dimension_scores_all
                else 0.0
            )
            score_no_rc = (
                dimension_scores_no_rc.get(dim, {}).get("vera_score", 0.0)
                if dim in dimension_scores_no_rc
                else 0.0
            )

            result_row[f"{dim}_score_all"] = score_all
            result_row[f"{dim}_score_no_role_confusion"] = score_no_rc
            result_row[f"{dim}_score_difference"] = score_no_rc - score_all

        results.append(result_row)

    return pd.DataFrame(results)


def plot_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
    title: str = "VERA Score Comparison: All Conversations vs No Role Confusion",
):
    """
    Plot comparison of scores between all conversations and role_confusion=False only.

    Args:
        comparison_df: DataFrame with comparison results
        output_path: Path to save the plot
        title: Plot title
    """
    if comparison_df.empty:
        print("⚠️  No data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    providers = comparison_df["provider_llm"].values
    provider_labels = [display_name(str(p)) for p in providers]
    x_pos = range(len(providers))

    # Plot 1: Overall scores comparison
    scores_all = comparison_df["overall_score_all"].values
    scores_no_rc = comparison_df["overall_score_no_role_confusion"].values

    width = 0.35
    ax1.bar(
        [x - width / 2 for x in x_pos],
        scores_all,
        width,
        label="All conversations",
        color=COLOR_GREEN,
        alpha=0.8,
    )
    ax1.bar(
        [x + width / 2 for x in x_pos],
        scores_no_rc,
        width,
        label="No role confusion (False)",
        color=COLOR_ORANGE,
        alpha=0.8,
    )

    ax1.set_xlabel("Provider Agent Model", fontsize=11, color=TEXT_COLOR)
    ax1.set_ylabel("VERA Overall Score", fontsize=11, color=TEXT_COLOR)
    ax1.set_title(
        "Overall Score Comparison", fontsize=12, fontweight="bold", color=TEXT_COLOR
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(provider_labels, rotation=45, ha="right", fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_facecolor("white")

    # Plot 2: Difference (no_rc - all)
    differences = comparison_df["overall_score_difference"].values
    colors = [COLOR_RED if d < 0 else COLOR_GREEN for d in differences]

    bars = ax2.bar(x_pos, differences, color=colors, alpha=0.8)
    ax2.axhline(y=0, color=TEXT_COLOR, linestyle="--", linewidth=1)

    # Annotate every bar with its significance label
    if "significance" in comparison_df.columns:
        sig_vals = comparison_df["significance"].fillna("").values
        has_pval = "p_value" in comparison_df.columns
        p_vals = comparison_df["p_value"].values if has_pval else [None] * len(sig_vals)

        # Determine a sensible y-offset from the current axis range
        all_diffs = [float(d) for d in differences if d is not None]
        y_range = max(abs(d) for d in all_diffs) if all_diffs else 1.0
        offset = y_range * 0.06

        for bar, sig, pv in zip(bars, sig_vals, p_vals):
            h = bar.get_height()
            label = str(sig) if sig else "ns"
            # Safely format p-value
            try:
                pv_is_nan = isinstance(pv, float) and np.isnan(pv)
                p_label = (
                    f"p={float(pv):.3f}" if pv is not None and not pv_is_nan else ""
                )
            except (TypeError, ValueError):
                p_label = ""

            annotation = f"{label}\n{p_label}" if p_label else label

            if h >= 0:
                y_text = h + offset
                va = "bottom"
            else:
                y_text = h - offset
                va = "top"

            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                annotation,
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold" if label not in ("ns", "") else "normal",
                color=TEXT_COLOR,
            )

    # Expand y-axis so significance annotations don't overlap the title.
    diff_floats = [float(d) for d in differences if d is not None]
    if diff_floats:
        y_max_data = max(diff_floats)
        y_min_data = min(diff_floats)
        y_span = max(abs(y_max_data), abs(y_min_data), 1.0)
        ax2.set_ylim(
            bottom=y_min_data - y_span * 0.15,
            top=y_max_data + y_span * 0.40,
        )

    ax2.set_xlabel("Provider Agent Model", fontsize=11, color=TEXT_COLOR)
    ax2.set_ylabel("Score Difference (No RC - All)", fontsize=11, color=TEXT_COLOR)
    ax2.set_title("Score Difference", fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(provider_labels, rotation=45, ha="right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"✅ Plot saved to: {output_path}")


def plot_comparison_by_provider_and_user(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "VERA Score Comparison by Provider and User Agent",
    significance_df: pd.DataFrame | None = None,
):
    """
    Plot comparison showing 4 bars per provider: user agent combinations with all vs role_confusion.

    Args:
        results_df: DataFrame with results for each provider/user combination
        output_path: Path to save the plot
        title: Plot title
        significance_df: Optional DataFrame with significance columns keyed by provider_llm/user_llm
    """
    if results_df.empty:
        print("⚠️  No data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Get unique providers and users
    providers = sorted(list(results_df["provider_llm"].unique()))  # type: ignore[union-attr]
    users = sorted(list(results_df["user_llm"].unique()))  # type: ignore[union-attr]
    provider_labels = [display_name(p) for p in providers]

    # Prepare data: for each provider, get scores for each user/all vs role_confusion combination
    x_pos_base = range(len(providers))
    width = 0.2  # Width for each bar (4 bars per provider = 0.2 each)

    # Colors for different combinations
    colors_map = {
        ("gpt 5 2", "all"): COLOR_GREEN,
        ("gpt 5 2", "no_role_confusion"): COLOR_ORANGE,
        ("opus 4 5", "all"): "#4A90E2",  # Blue
        ("opus 4 5", "no_role_confusion"): "#FF8C00",  # Dark orange
    }

    # Plot 1: Overall scores comparison
    for user in users:
        for score_type in ["all", "no_role_confusion"]:
            scores = []
            for provider in providers:
                mask = (results_df["provider_llm"] == provider) & (
                    results_df["user_llm"] == user
                )
                matching = results_df[mask]
                if not matching.empty:
                    if score_type == "all":
                        scores.append(matching.iloc[0]["overall_score_all"])
                    else:
                        scores.append(
                            matching.iloc[0]["overall_score_no_role_confusion"]
                        )
                else:
                    scores.append(0)

            # Calculate x positions for this bar group
            offset = (
                users.index(user) * 2 + (0 if score_type == "all" else 1)
            ) * width - width * 1.5
            x_positions = [x + offset for x in x_pos_base]

            label = f"{display_name(user)}, {score_type.replace('_', ' ')}"
            color = colors_map.get((user, score_type), "#888888")

            ax1.bar(
                x_positions,
                scores,
                width,
                label=label,
                color=color,
                alpha=0.8,
            )

    ax1.set_xlabel("Provider Agent Model", fontsize=11, color=TEXT_COLOR)
    ax1.set_ylabel("VERA Overall Score", fontsize=11, color=TEXT_COLOR)
    ax1.set_title(
        "Overall Score Comparison", fontsize=12, fontweight="bold", color=TEXT_COLOR
    )
    ax1.set_xticks(x_pos_base)
    ax1.set_xticklabels(provider_labels, rotation=45, ha="right", fontsize=9)
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_facecolor("white")

    # Plot 2: Differences (no_role_confusion - all) for each user
    # Assign distinct colors to each user agent
    user_colors = {
        "gpt 5 2": COLOR_ORANGE,
        "opus 4 5": "#4A90E2",  # Blue color for Opus
    }

    for user in users:
        differences = []
        for provider in providers:
            mask = (results_df["provider_llm"] == provider) & (
                results_df["user_llm"] == user
            )
            matching = results_df[mask]

            if not matching.empty:
                diff = matching.iloc[0]["overall_score_difference"]
                differences.append(diff)
            else:
                differences.append(0)

        # Calculate x positions for this bar group
        offset = (users.index(user) * width * 2) - width * 0.5
        x_positions = [x + offset for x in x_pos_base]

        # Use the same color for all bars of this user agent
        user_color = user_colors.get(user, "#888888")

        bars2 = ax2.bar(
            x_positions,
            differences,
            width * 2,
            label=f"{display_name(user)} (No RC - All)",
            color=user_color,
            alpha=0.8,
        )

        # Annotate every bar with its significance label
        if significance_df is not None and "significance" in significance_df.columns:
            # Determine y-offset from overall differences range
            all_diffs_u = []
            for prov in providers:
                m = (results_df["provider_llm"] == prov) & (
                    results_df["user_llm"] == user
                )
                row_match = results_df[m]
                if not row_match.empty:
                    all_diffs_u.append(
                        float(row_match.iloc[0]["overall_score_difference"])
                    )
            y_range_u = max(abs(d) for d in all_diffs_u) if all_diffs_u else 1.0
            offset_u = y_range_u * 0.06

            for bar, provider in zip(bars2, providers):
                sig_mask = (significance_df["provider_llm"] == provider) & (
                    significance_df["user_llm"] == user
                )
                sig_rows = significance_df[sig_mask]
                if sig_rows.empty:
                    continue
                sig = str(sig_rows.iloc[0]["significance"])
                raw_pv = sig_rows.iloc[0].get("p_value", None)
                label = sig if sig else "ns"
                try:
                    pv_is_nan = isinstance(raw_pv, float) and np.isnan(raw_pv)
                    p_label = (
                        f"p={float(raw_pv):.3f}"
                        if raw_pv is not None and not pv_is_nan
                        else ""
                    )
                except (TypeError, ValueError):
                    p_label = ""

                annotation = f"{label}\n{p_label}" if p_label else label
                h = bar.get_height()
                if h >= 0:
                    y_text = h + offset_u
                    va = "bottom"
                else:
                    y_text = h - offset_u
                    va = "top"

                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_text,
                    annotation,
                    ha="center",
                    va=va,
                    fontsize=8,
                    fontweight="bold" if label not in ("ns", "") else "normal",
                    color=TEXT_COLOR,
                )

    ax2.axhline(y=0, color=TEXT_COLOR, linestyle="--", linewidth=1)

    # Expand y-axis so significance annotations don't overlap the title.
    # Collect every difference value plotted for this axis.
    all_plotted_diffs: list[float] = []
    for user in users:
        for provider in providers:
            m = (results_df["provider_llm"] == provider) & (
                results_df["user_llm"] == user
            )
            row_m = results_df[m]
            if not row_m.empty:
                all_plotted_diffs.append(
                    float(row_m.iloc[0]["overall_score_difference"])
                )

    if all_plotted_diffs:
        y_max_data = max(all_plotted_diffs)
        y_min_data = min(all_plotted_diffs)
        y_span = max(abs(y_max_data), abs(y_min_data), 1.0)
        # Add 40% headroom above tallest bar for annotation text
        ax2.set_ylim(
            bottom=y_min_data - y_span * 0.15,
            top=y_max_data + y_span * 0.40,
        )

    ax2.set_xlabel("Provider Agent Model", fontsize=11, color=TEXT_COLOR)
    ax2.set_ylabel("Score Difference (No RC - All)", fontsize=11, color=TEXT_COLOR)
    ax2.set_title(
        "Score Difference by User Agent",
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax2.set_xticks(x_pos_base)
    ax2.set_xticklabels(provider_labels, rotation=45, ha="right", fontsize=9)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"✅ Plot saved to: {output_path}")


def plot_correlations(
    correlation_df: pd.DataFrame,
    output_path: Path,
    title: str = "Correlations with Role Confusion",
):
    """
    Plot correlation results as a horizontal bar chart.

    Args:
        correlation_df: DataFrame with correlation results
        output_path: Path to save the plot
        title: Plot title
    """
    if correlation_df.empty:
        print("⚠️  No correlation data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, max(8, len(correlation_df) * 0.4)))

    # Sort by absolute correlation value (already sorted, but ensure)
    correlation_df = correlation_df.copy()
    correlation_df["abs_corr"] = correlation_df["correlation_value"].abs()
    correlation_df = correlation_df.sort_values("abs_corr", ascending=True)
    correlation_df = correlation_df.drop(columns=["abs_corr"])

    y_pos = range(len(correlation_df))
    variables = correlation_df["variable_name"].values
    correlations = np.array(correlation_df["correlation_value"].values)
    corr_types = correlation_df["correlation_type"].values
    significance = correlation_df["statistical_significance"].values

    # Color-code by correlation type
    colors = []
    for corr_type in corr_types:
        if corr_type == "Cramér's V":
            colors.append(COLOR_ORANGE)
        else:  # Point-biserial
            colors.append(COLOR_GREEN)

    # Create bars
    bars = ax.barh(y_pos, correlations, color=colors, alpha=0.8)

    # Add significance markers
    for i, (bar, sig) in enumerate(zip(bars, significance)):
        if sig:
            # Add text annotation for significance
            x_pos = bar.get_width()
            if x_pos >= 0:
                ax.text(
                    x_pos + 0.01,
                    i,
                    sig,
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=TEXT_COLOR,
                )
            else:
                ax.text(
                    x_pos - 0.01,
                    i,
                    sig,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                    color=TEXT_COLOR,
                )

    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables, fontsize=9)
    ax.set_xlabel("Correlation Value", fontsize=11, color=TEXT_COLOR)
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.axvline(x=0, color=TEXT_COLOR, linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_facecolor("white")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLOR_ORANGE, alpha=0.8, label="Cramér's V (categorical)"),
        Patch(facecolor=COLOR_GREEN, alpha=0.8, label="Point-biserial (continuous)"),
        Patch(
            facecolor="white", label="Significance: *** p<0.001, ** p<0.01, * p<0.05"
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"✅ Correlation plot saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare VERA scores with and without role confusion conversations"
    )
    parser.add_argument(
        "--role-confusion-csv",
        type=Path,
        default=Path("role_confusion_results.csv"),
        help="Path to role_confusion_results.csv (default: role_confusion_results.csv)",
    )
    parser.add_argument(
        "--concatenated-csv",
        type=Path,
        default=Path("concatenated_heor_paper2_results.csv"),
        help="Path to concatenated_heor_paper2_results.csv (default: concatenated_heor_paper2_results.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("score_variability/role_confusion_comparison.csv"),
        help="Path to save comparison CSV (default: score_variability/role_confusion_comparison.csv)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Path to save plot PNG (default: same as output with .png extension)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plot",
    )
    parser.add_argument(
        "--personas-tsv",
        type=Path,
        default=Path("data/personas.tsv"),
        help="Path to personas.tsv file (default: data/personas.tsv)",
    )

    args = parser.parse_args()

    print("🔬 Loading and merging data...")
    try:
        df = load_and_merge_data(
            args.role_confusion_csv, args.concatenated_csv, args.personas_tsv
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1

    print("\n🔬 Analyzing correlations with role confusion...")
    correlation_df = analyze_role_confusion_correlations(df)
    rates_by_value_df = analyze_role_confusion_rates_by_value(df, correlation_df)

    print("\n🔬 Comparing scores with and without role confusion...")
    comparison_df = compare_role_confusion_impact(df)
    comparison_df_provider_user = compare_role_confusion_by_provider_and_user(df)

    if comparison_df.empty and comparison_df_provider_user.empty:
        print("❌ No comparison results generated")
        return 1

    print("\n🔬 Running bootstrap significance tests (1000 iterations per group)...")
    comparison_df = add_significance_to_role_confusion_comparison(
        comparison_df, df, group_cols=["provider_llm"]
    )
    comparison_df_provider_user = add_significance_to_role_confusion_comparison(
        comparison_df_provider_user, df, group_cols=["provider_llm", "user_llm"]
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not comparison_df.empty:
        provider_output = output_path.with_stem(output_path.stem + "_by_provider")
        comparison_df.to_csv(provider_output, index=False)
        print(f"\n✅ Provider comparison saved to: {provider_output}")

    if not comparison_df_provider_user.empty:
        provider_user_output = output_path.with_stem(
            output_path.stem + "_by_provider_and_user"
        )
        comparison_df_provider_user.to_csv(provider_user_output, index=False)
        print(f"\n✅ Provider+User comparison saved to: {provider_user_output}")

    # Save correlation results
    if not correlation_df.empty:
        correlation_output = output_path.with_stem(output_path.stem + "_correlations")
        correlation_df.to_csv(correlation_output, index=False)
        print(f"\n✅ Correlation analysis saved to: {correlation_output}")

    # Save rates-by-value results
    if not rates_by_value_df.empty:
        rates_output = output_path.with_stem(output_path.stem + "_rates_by_value")
        rates_by_value_df.to_csv(rates_output, index=False)
        print(f"✅ Role confusion rates by value saved to: {rates_output}")

    # Generate plots
    if not args.no_plot:
        if args.plot:
            plot_path = Path(args.plot)
            provider_plot = plot_path.with_stem(plot_path.stem + "_by_provider")
            provider_user_plot = plot_path.with_stem(
                plot_path.stem + "_by_provider_and_user"
            )
        else:
            provider_plot = output_path.with_stem(
                output_path.stem + "_by_provider"
            ).with_suffix(".png")
            provider_user_plot = output_path.with_stem(
                output_path.stem + "_by_provider_and_user"
            ).with_suffix(".png")

        if not comparison_df.empty:
            plot_comparison(
                comparison_df,
                provider_plot,
                title="VERA Score Comparison: All Conversations vs No Role Confusion",
            )

        if not comparison_df_provider_user.empty:
            plot_comparison_by_provider_and_user(
                comparison_df_provider_user,
                provider_user_plot,
                title="VERA Score Comparison by Provider and User Agent",
                significance_df=(
                    comparison_df_provider_user
                    if "significance" in comparison_df_provider_user.columns
                    else None
                ),
            )

        if not correlation_df.empty:
            if args.plot:
                correlation_plot = plot_path.with_stem(plot_path.stem + "_correlations")
            else:
                correlation_plot = output_path.with_stem(
                    output_path.stem + "_correlations"
                ).with_suffix(".png")

            plot_correlations(
                correlation_df,
                correlation_plot,
                title="Correlations with Role Confusion",
            )

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY - BY PROVIDER")
    print("=" * 80)
    if not comparison_df.empty:
        print(comparison_df.to_string(index=False))
    else:
        print("No provider comparison data")

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY - BY PROVIDER AND USER")
    print("=" * 80)
    if not comparison_df_provider_user.empty:
        print(comparison_df_provider_user.to_string(index=False))
    else:
        print("No provider+user comparison data")

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS - TOP 10 VARIABLES")
    print("=" * 80)
    if not correlation_df.empty:
        top_10 = correlation_df.head(10)
        print(top_10.to_string(index=False))
        print("\n(Full results saved to CSV)")
    else:
        print("No correlation data")

    print("\n" + "=" * 80)
    print("VALUES POSITIVELY ASSOCIATED WITH ROLE CONFUSION")
    print(f"  (overall role confusion rate: {df['role_confusion'].mean():.1%})")
    print("=" * 80)
    if not rates_by_value_df.empty:
        higher_df = rates_by_value_df[rates_by_value_df["direction"] == "higher"].copy()
        assert isinstance(higher_df, pd.DataFrame)
        # Print grouped by variable
        for var in list(higher_df["variable"].unique()):
            var_rows = higher_df[higher_df["variable"] == var]
            assert isinstance(var_rows, pd.DataFrame)
            print(f"\n  {var}:")
            for _, row in var_rows.iterrows():
                print(
                    f"    {str(row['value']):<40}  "
                    f"rate={row['confusion_rate']:.1%}  "
                    f"(+{row['rate_vs_overall']:+.1%} vs overall)  "
                    f"n={row['confused_count']}/{row['total_count']}"
                )
        print("\n(Full breakdown including lower-than-average values saved to CSV)")
    else:
        print("No rates-by-value data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
