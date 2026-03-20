#!/usr/bin/env python3
"""
Compare VERA scores between max_turns=20 and max_turns=100 for each provider agent model.

This script loads concatenated_heor_paper2_results.csv and calculates VERA scores
for each provider agent model, comparing scores between conversations with max_turns=20
and max_turns=100.

Usage:
    python scripts/compare_max_turns_by_provider.py
    python scripts/compare_max_turns_by_provider.py -o results/max_turns_comparison.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend

try:
    from scipy.stats import mannwhitneyu as _mannwhitneyu

    def mannwhitneyu(
        a: Any, b: Any, alternative: str = "two-sided"
    ) -> tuple[float, float]:
        result = _mannwhitneyu(a, b, alternative=alternative)
        return float(result.statistic), float(result.pvalue)

except ImportError:
    mannwhitneyu = None  # type: ignore[assignment]

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from judge.score_utils import (
    BG_COLOR,
    DIMENSIONS,
    TEXT_COLOR,
    calculate_scores_from_df,
)

# ── Simulated turn thresholds & colours ───────────────────────────────────────
# For each threshold T we include conversations where actual_conversation_turns <= T.
TURN_THRESHOLDS: list[int] = [20, 30, 40, 50, 60, 70, 80, 90, 100]
# Sequential blue palette — lightest → darkest
_THRESHOLD_COLORS: list[str] = ["#9ecae1", "#4292c6", "#2166ac", "#084594"]

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


def load_concatenated_results(
    csv_path: Path = Path("concatenated_heor_paper2_results.csv"),
) -> pd.DataFrame:
    """
    Load the concatenated results CSV file.

    Args:
        csv_path: Path to concatenated_heor_paper2_results.csv

    Returns:
        DataFrame with all evaluation results
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Concatenated results file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from {csv_path}")

    return df


def bootstrap_vera_scores(
    group_df: pd.DataFrame,
    n_iterations: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Bootstrap the aggregate VERA score for a group of conversations by resampling
    whole conversations (identified by filename) with replacement.

    The VERA score is designed to work on aggregated data, so we must resample at
    the conversation level (preserving within-conversation correlation structure)
    and recompute the aggregate score each time.

    Args:
        group_df: DataFrame of rows belonging to one group (e.g. one provider at t=20)
        n_iterations: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        Array of bootstrapped aggregate VERA scores (length = n_iterations)
    """
    if "filename" not in group_df.columns:
        return np.array([])

    rng = np.random.default_rng(seed)
    conversations = [
        df for _, df in group_df.groupby("filename") if isinstance(df, pd.DataFrame)
    ]
    n_convs = len(conversations)
    if n_convs == 0:
        return np.array([])

    boot_scores: list[float] = []
    for _ in range(n_iterations):
        indices = rng.integers(0, n_convs, size=n_convs)
        sample_df = pd.concat([conversations[i] for i in indices], ignore_index=True)
        score = calculate_scores_from_df(sample_df).get("overall_score", 0.0)
        boot_scores.append(score)

    return np.array(boot_scores)


def test_significance(
    group_t20: pd.DataFrame,
    group_t100: pd.DataFrame,
    n_iterations: int = 1000,
) -> dict[str, Any]:
    """
    Test whether the aggregate VERA score difference between max_turns=20 and
    max_turns=100 is statistically significant using bootstrap resampling.

    Resamples whole conversations with replacement, computes aggregate VERA score
    each time, then estimates a p-value from the proportion of bootstrap differences
    that cross zero (two-sided).

    Returns a dict with: p_value, significance (stars), boot_diff_mean,
    ci_lower, ci_upper, n_convs_t20, n_convs_t100.
    """
    n_t20 = (
        group_t20["filename"].nunique()
        if "filename" in group_t20.columns
        else len(group_t20)
    )
    n_t100 = (
        group_t100["filename"].nunique()
        if "filename" in group_t100.columns
        else len(group_t100)
    )

    base: dict[str, Any] = {
        "p_value": None,
        "significance": "",
        "boot_diff_mean": None,
        "ci_lower": None,
        "ci_upper": None,
        "n_convs_t20": n_t20,
        "n_convs_t100": n_t100,
    }

    if group_t20.empty or group_t100.empty:
        return base

    print(
        f"      Bootstrap resampling ({n_iterations} iterations)…", end=" ", flush=True
    )
    boot_t20 = bootstrap_vera_scores(group_t20, n_iterations=n_iterations)
    boot_t100 = bootstrap_vera_scores(group_t100, n_iterations=n_iterations)
    print("done")

    if len(boot_t20) == 0 or len(boot_t100) == 0:
        return base

    boot_diffs = boot_t100 - boot_t20

    # Two-sided p-value: proportion of differences that are on the wrong side of 0
    observed_diff = calculate_scores_from_df(group_t100).get(
        "overall_score", 0.0
    ) - calculate_scores_from_df(group_t20).get("overall_score", 0.0)
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
        "n_convs_t20": n_t20,
        "n_convs_t100": n_t100,
    }


def add_significance_to_comparison(
    comparison_df: pd.DataFrame,
    df: pd.DataFrame,
    group_cols: list[str],
    n_iterations: int = 1000,
) -> pd.DataFrame:
    """
    For each row in comparison_df, run a bootstrap significance test comparing
    the aggregate VERA score at max_turns=20 vs max_turns=100.

    Adds columns: p_value, significance, boot_diff_mean, ci_lower, ci_upper,
    n_convs_t20, n_convs_t100.
    """
    sig_rows: list[dict[str, Any]] = []
    for _, row in comparison_df.iterrows():
        mask = pd.Series([True] * len(df), index=df.index)
        for col in group_cols:
            if col in df.columns and col in comparison_df.columns:
                mask = mask & (df[col] == row[col])

        group_t20 = df[mask & (df["max_turns"] == 20)]
        group_t100 = df[mask & (df["max_turns"] == 100)]
        assert isinstance(group_t20, pd.DataFrame)
        assert isinstance(group_t100, pd.DataFrame)

        label = ", ".join(str(row[c]) for c in group_cols if c in row)
        print(f"   Significance test for: {label}")
        sig_rows.append(
            test_significance(group_t20, group_t100, n_iterations=n_iterations)
        )

    sig_df = pd.DataFrame(sig_rows)
    return pd.concat(
        [comparison_df.reset_index(drop=True), sig_df.reset_index(drop=True)], axis=1
    )


def compare_max_turns_by_provider(
    df: pd.DataFrame,
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """
    Simulate the effect of different max_turns settings by filtering to
    conversations where actual_conversation_turns <= T for each threshold T.

    Args:
        df: DataFrame with all evaluation results (must have actual_conversation_turns)
        thresholds: Turn-count thresholds to evaluate (default: TURN_THRESHOLDS)

    Returns:
        DataFrame with one row per provider and columns score_tT for each threshold T.
    """
    if thresholds is None:
        thresholds = TURN_THRESHOLDS

    if (
        "provider_llm" not in df.columns
        or "actual_conversation_turns" not in df.columns
    ):
        print(
            "❌ Error: Missing required columns (provider_llm, actual_conversation_turns)"
        )
        return pd.DataFrame()

    providers = sorted(df["provider_llm"].dropna().unique())
    results: list[dict[str, Any]] = []

    for provider in providers:
        prov_df_raw = df[df["provider_llm"] == provider]
        assert isinstance(prov_df_raw, pd.DataFrame)
        prov_df: pd.DataFrame = prov_df_raw

        row: dict[str, Any] = {"provider_llm": provider}

        for t in thresholds:
            subset_raw = prov_df[prov_df["actual_conversation_turns"] <= t]
            assert isinstance(subset_raw, pd.DataFrame)
            subset: pd.DataFrame = subset_raw

            n_convs = (
                subset["filename"].nunique()
                if "filename" in subset.columns
                else len(subset)
            )
            print(
                f"\n📊 {provider}, actual_turns <= {t}: {len(subset)} rows ({n_convs} conversations)"
            )

            row[f"num_convs_t{t}"] = n_convs
            if not subset.empty:
                scores = calculate_scores_from_df(subset)
                row[f"score_t{t}"] = scores.get("overall_score", 0.0)
                dim_scores = scores.get("dimension_scores", {})
                for dim in DIMENSIONS:
                    row[f"{dim}_t{t}"] = (
                        dim_scores[dim].get("vera_score", 0.0)
                        if dim in dim_scores
                        else 0.0
                    )
            else:
                row[f"score_t{t}"] = None
                for dim in DIMENSIONS:
                    row[f"{dim}_t{t}"] = None

        results.append(row)

    return pd.DataFrame(results)


def compare_max_turns_by_provider_and_user(
    df: pd.DataFrame,
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """
    Simulate the effect of different max_turns settings, split by provider and user agent.

    Args:
        df: DataFrame with all evaluation results (must have actual_conversation_turns)
        thresholds: Turn-count thresholds to evaluate (default: TURN_THRESHOLDS)

    Returns:
        DataFrame with one row per (provider, user) and columns score_tT for each T.
    """
    if thresholds is None:
        thresholds = TURN_THRESHOLDS

    if (
        "provider_llm" not in df.columns
        or "user_llm" not in df.columns
        or "actual_conversation_turns" not in df.columns
    ):
        print(
            "❌ Error: Missing required columns (provider_llm, user_llm, actual_conversation_turns)"
        )
        return pd.DataFrame()

    results: list[dict[str, Any]] = []

    for group_key, group_df_raw in df.groupby(
        ["provider_llm", "user_llm"], dropna=False
    ):
        if isinstance(group_df_raw, pd.Series):
            continue
        group_df: pd.DataFrame = group_df_raw

        provider_llm = (
            str(group_key[0]) if isinstance(group_key, tuple) else str(group_key)
        )
        user_llm = (
            str(group_key[1])
            if isinstance(group_key, tuple) and len(group_key) > 1
            else ""
        )

        row: dict[str, Any] = {"provider_llm": provider_llm, "user_llm": user_llm}

        for t in thresholds:
            subset_raw = group_df[group_df["actual_conversation_turns"] <= t]
            assert isinstance(subset_raw, pd.DataFrame)
            subset: pd.DataFrame = subset_raw

            n_convs = (
                subset["filename"].nunique()
                if "filename" in subset.columns
                else len(subset)
            )
            print(
                f"\n📊 provider={provider_llm}, user={user_llm}, actual_turns <= {t}: "
                f"{len(subset)} rows ({n_convs} conversations)"
            )

            row[f"num_convs_t{t}"] = n_convs
            if not subset.empty:
                scores = calculate_scores_from_df(subset)
                row[f"score_t{t}"] = scores.get("overall_score", 0.0)
            else:
                row[f"score_t{t}"] = None

        results.append(row)

    return pd.DataFrame(results)


def plot_actual_turns_histogram(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Distribution of Actual Conversation Turns — max_turns=100 Conversations",
) -> None:
    """
    Histogram of actual_conversation_turns for unique conversations,
    with one subplot per provider.

    Deduplicates by 'filename' before plotting so each conversation is counted once.
    Vertical lines are drawn at each TURN_THRESHOLD to show how many conversations
    fall within each simulated max-turns window.
    """
    if "actual_conversation_turns" not in df.columns:
        print("⚠️  No actual_conversation_turns column — skipping histogram")
        return

    # Restrict to max_turns=100 conversations only
    if "max_turns" in df.columns:
        df_raw = df[df["max_turns"] == 100]
        assert isinstance(df_raw, pd.DataFrame)
        df = df_raw

    # One row per unique conversation
    id_col = "filename" if "filename" in df.columns else None
    if id_col:
        unique_df_raw = df.drop_duplicates(subset=[id_col])
        assert isinstance(unique_df_raw, pd.DataFrame)
        unique_df: pd.DataFrame = unique_df_raw
    else:
        unique_df = df.copy()

    unique_df = unique_df.dropna(subset=["actual_conversation_turns"]).copy()
    unique_df["actual_conversation_turns"] = unique_df[
        "actual_conversation_turns"
    ].astype(int)

    providers = (
        sorted(unique_df["provider_llm"].dropna().unique())
        if "provider_llm" in unique_df.columns
        else []
    )
    n_prov = len(providers)
    if n_prov == 0:
        # Fall back to a single histogram of all conversations
        fig, ax = plt.subplots(figsize=(10, 5))
        turns = unique_df["actual_conversation_turns"]
        ax.hist(
            turns,
            bins=range(int(turns.min()), int(turns.max()) + 2),
            color="#5B9BD5",
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOR)
        ax.set_xlabel("Actual conversation turns", fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel("Number of conversations", fontsize=10, color=TEXT_COLOR)
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor("white")
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        print(f"📈 Histogram saved to: {output_path}")
        return

    prov_palette = ["#5B9BD5", "#ED7D31", "#A9D18E", "#FFC000", "#7030A0", "#00B0F0"]
    prov_colors = {
        p: prov_palette[i % len(prov_palette)] for i, p in enumerate(providers)
    }

    ncols = 2
    nrows = (n_prov + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    fig.patch.set_facecolor(BG_COLOR)

    # Shared x range across all subplots
    all_turns = unique_df["actual_conversation_turns"]
    x_min = int(all_turns.min())
    x_max = int(all_turns.max())
    bins = range(x_min, x_max + 2)

    for i, provider in enumerate(providers):
        ax = axes_flat[i]
        ax.set_facecolor("white")

        prov_turns_raw = unique_df[unique_df["provider_llm"] == provider][
            "actual_conversation_turns"
        ]
        assert isinstance(prov_turns_raw, pd.Series)
        prov_turns: pd.Series = prov_turns_raw

        ax.hist(
            prov_turns,
            bins=list(bins),
            color=prov_colors[provider],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
        )

        ax.set_title(
            display_name(provider), fontsize=11, fontweight="bold", color=TEXT_COLOR
        )
        ax.set_xlabel("Actual conversation turns", fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel("Unique conversations", fontsize=10, color=TEXT_COLOR)
        ax.set_xlim(x_min, x_max + 1)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.25, axis="y")

    # Hide unused axes
    for j in range(n_prov, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01, color=TEXT_COLOR)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Histogram saved to: {output_path}")


def plot_actual_turns_histogram_combined(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Distribution of Actual Conversation Turns — All Providers (max_turns=100)",
) -> None:
    """
    Single histogram of actual_conversation_turns across all providers combined,
    for max_turns=100 conversations only. Each conversation counted once.
    """
    if "actual_conversation_turns" not in df.columns:
        print("⚠️  No actual_conversation_turns column — skipping combined histogram")
        return

    # Restrict to max_turns=100 conversations only
    if "max_turns" in df.columns:
        df_raw = df[df["max_turns"] == 100]
        assert isinstance(df_raw, pd.DataFrame)
        df = df_raw

    # One row per unique conversation
    id_col = "filename" if "filename" in df.columns else None
    if id_col:
        unique_df_raw = df.drop_duplicates(subset=[id_col])
        assert isinstance(unique_df_raw, pd.DataFrame)
        unique_df: pd.DataFrame = unique_df_raw
    else:
        unique_df = df.copy()

    unique_df = unique_df.dropna(subset=["actual_conversation_turns"]).copy()
    unique_df["actual_conversation_turns"] = unique_df[
        "actual_conversation_turns"
    ].astype(int)

    turns = unique_df["actual_conversation_turns"]
    if turns.empty:
        print("⚠️  No turn data after filtering — skipping combined histogram")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor("white")

    bins = range(int(turns.min()), int(turns.max()) + 2)
    ax.hist(
        turns,
        bins=list(bins),
        color="#5B9BD5",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Actual conversation turns", fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel("Unique conversations", fontsize=10, color=TEXT_COLOR)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Combined histogram saved to: {output_path}")


def plot_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
    title: str = "VERA Score Comparison: Simulated Max Turns",
    by_provider: bool = True,
    thresholds: list[int] | None = None,
):
    """
    Plot VERA scores across simulated max-turns thresholds.

    Left panel: grouped bars — one group per entity (provider/user), one bar per threshold.
    Right panel: line chart — score vs threshold, one line per entity.

    Args:
        comparison_df: DataFrame from compare_max_turns_by_provider (columns score_tT).
        output_path: Path to save the plot.
        by_provider: If True label by provider_llm; else by user_llm.
        thresholds: Thresholds to plot (default: TURN_THRESHOLDS).
    """
    if thresholds is None:
        thresholds = TURN_THRESHOLDS

    if comparison_df.empty:
        print("⚠️  No data to plot")
        return

    label_col = "provider_llm" if by_provider else "user_llm"
    if label_col not in comparison_df.columns:
        print(f"⚠️  Missing column: {label_col}")
        return

    # Detect which thresholds are actually present
    available = [t for t in thresholds if f"score_t{t}" in comparison_df.columns]
    if not available:
        print("⚠️  No score_tT columns found in comparison_df")
        return

    baseline_t = max(available)
    xlabel = "Simulated max turns  (actual turns <= T)"
    line_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: line chart — absolute score vs threshold per entity ──────────────
    for row_i, row in comparison_df.iterrows():
        ys = [row.get(f"score_t{t}") for t in available]
        valid = [(t, y) for t, y in zip(available, ys) if y is not None]
        if not valid:
            continue
        vx, vy = zip(*valid)
        lbl = display_name(str(row[label_col]))
        ax1.plot(
            vx,
            vy,
            marker="o",
            linewidth=2,
            markersize=7,
            color=line_colors[int(row_i) % len(line_colors)],  # type: ignore[arg-type]
            label=lbl,
        )

    ax1.set_xlabel(xlabel, fontsize=10, color=TEXT_COLOR)
    ax1.set_ylabel("VERA Overall Score", fontsize=11, color=TEXT_COLOR)
    ax1.set_title(
        "Score vs Max-Turns Threshold", fontsize=12, fontweight="bold", color=TEXT_COLOR
    )
    ax1.set_xticks(available)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("white")

    # ── Right: line chart — score difference from T=max per entity ─────────────
    baseline_scores_series = comparison_df[f"score_t{baseline_t}"]
    for row_i, row in comparison_df.iterrows():
        baseline_val = baseline_scores_series.iloc[int(row_i)]  # type: ignore[arg-type]
        diffs: list[tuple[int, float]] = []
        for t in available:
            val = row.get(f"score_t{t}")
            if val is not None and baseline_val is not None:
                diffs.append((t, float(val) - float(baseline_val)))
        if not diffs:
            continue
        vx2, vy2 = zip(*diffs)
        lbl = display_name(str(row[label_col]))
        ax2.plot(
            vx2,
            vy2,
            marker="o",
            linewidth=2,
            markersize=7,
            color=line_colors[int(row_i) % len(line_colors)],  # type: ignore[arg-type]
            label=lbl,
        )

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel(xlabel, fontsize=10, color=TEXT_COLOR)
    ax2.set_ylabel(
        f"Score Difference vs. T={baseline_t}", fontsize=11, color=TEXT_COLOR
    )
    ax2.set_title(
        f"Score Difference from T={baseline_t}",
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax2.set_xticks(available)
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("white")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02, color=TEXT_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Plot saved to: {output_path}")


def plot_comparison_by_provider_and_user(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "VERA Score vs Simulated Max Turns by Provider and User Agent",
    significance_df: pd.DataFrame | None = None,
    thresholds: list[int] | None = None,
):
    """
    Line chart: score vs simulated-max-turns threshold, one line per (provider, user) combo.

    Left panel: GPT user agent lines.
    Right panel: Opus user agent lines.

    Args:
        results_df: DataFrame from compare_max_turns_by_provider_and_user.
        output_path: Path to save the plot.
        thresholds: Thresholds to plot (default: TURN_THRESHOLDS).
    """
    if thresholds is None:
        thresholds = TURN_THRESHOLDS

    if results_df.empty:
        print("⚠️  No data to plot")
        return

    available = [t for t in thresholds if f"score_t{t}" in results_df.columns]
    if not available:
        print("⚠️  No score_tT columns found")
        return

    providers = sorted(results_df["provider_llm"].unique())
    users = sorted(results_df["user_llm"].unique())

    # One colour per provider, consistent across panels
    prov_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    prov_colors = {
        p: prov_palette[i % len(prov_palette)] for i, p in enumerate(providers)
    }

    # One line style per user (solid / dashed)
    user_ls = {u: ls for u, ls in zip(users, ["-", "--", "-.", ":"])}

    baseline_t = max(available)

    # Compute diffs from baseline for all rows, then find shared y limits
    all_diffs: list[float] = []
    for _, row in results_df.iterrows():
        baseline_val = row.get(f"score_t{baseline_t}")
        if baseline_val is None:
            continue
        for t in available:
            val = row.get(f"score_t{t}")
            if val is not None:
                all_diffs.append(float(val) - float(baseline_val))
    if all_diffs:
        y_pad = max((max(all_diffs) - min(all_diffs)) * 0.08, 0.5)
        shared_ymin = min(all_diffs) - y_pad
        shared_ymax = max(all_diffs) + y_pad
    else:
        shared_ymin, shared_ymax = None, None

    n_users = len(users)
    fig, axes = plt.subplots(1, n_users, figsize=(7 * n_users, 6), squeeze=False)
    fig.patch.set_facecolor(BG_COLOR)

    for u_idx, user in enumerate(users):
        ax = axes[0][u_idx]
        user_df_raw = results_df[results_df["user_llm"] == user]
        assert isinstance(user_df_raw, pd.DataFrame)
        user_df: pd.DataFrame = user_df_raw

        for _, row in user_df.iterrows():
            provider = str(row["provider_llm"])
            baseline_val = row.get(f"score_t{baseline_t}")
            if baseline_val is None:
                continue
            diffs = [
                (t, float(row.get(f"score_t{t}")) - float(baseline_val))  # type: ignore[arg-type]
                for t in available
                if row.get(f"score_t{t}") is not None
            ]
            if not diffs:
                continue
            vx, vy = zip(*diffs)
            ax.plot(
                vx,
                vy,
                marker="o",
                linewidth=2,
                markersize=7,
                color=prov_colors[provider],
                linestyle=user_ls.get(user, "-"),
                label=display_name(provider),
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(
            "Simulated max turns  (actual turns <= T)", fontsize=10, color=TEXT_COLOR
        )
        ax.set_ylabel(
            f"Score Difference vs. T={baseline_t}", fontsize=11, color=TEXT_COLOR
        )
        ax.set_title(
            f"User: {display_name(user)}",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        ax.set_xticks(available)
        if shared_ymin is not None:
            ax.set_ylim(shared_ymin, shared_ymax)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("white")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02, color=TEXT_COLOR)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"📈 Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare VERA scores between max_turns=20 and max_turns=100 by provider"
    )

    parser.add_argument(
        "--input-csv",
        "-i",
        default="concatenated_heor_paper2_results.csv",
        help="Input concatenated results CSV file (default: concatenated_heor_paper2_results.csv)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="score_variability/max_turns_comparison.csv",
        help="Output CSV file path",
    )

    parser.add_argument(
        "--plot",
        default=None,
        help="Output plot file path (default: same as output with .png extension)",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the plot",
    )

    args = parser.parse_args()

    input_csv_path = Path(args.input_csv)
    if not input_csv_path.exists():
        print(f"❌ Error: Input CSV file not found: {input_csv_path}")
        return 1

    print(f"📂 Loading concatenated results from: {input_csv_path}")
    df = load_concatenated_results(input_csv_path)

    print(
        f"\n🔬 Comparing scores across simulated turn thresholds {TURN_THRESHOLDS}..."
    )
    comparison_df_provider = compare_max_turns_by_provider(
        df, thresholds=TURN_THRESHOLDS
    )
    results_df_provider_user = compare_max_turns_by_provider_and_user(
        df, thresholds=TURN_THRESHOLDS
    )

    # Bootstrap significance tests — commented out for speed; re-enable when needed
    # print("\n🔬 Running bootstrap significance tests (1000 iterations per group)...")
    # comparison_df_provider = add_significance_to_comparison(
    #     comparison_df_provider, df, group_cols=["provider_llm"]
    # )
    # results_df_provider_user = add_significance_to_comparison(
    #     results_df_provider_user, df, group_cols=["provider_llm", "user_llm"]
    # )

    if comparison_df_provider.empty and results_df_provider_user.empty:
        print("❌ No comparison results generated")
        return 1

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not comparison_df_provider.empty:
        provider_output = output_path.with_stem(output_path.stem + "_by_provider")
        comparison_df_provider.to_csv(provider_output, index=False)
        print(f"\n✅ Provider comparison saved to: {provider_output}")

    if not results_df_provider_user.empty:
        provider_user_output = output_path.with_stem(
            output_path.stem + "_by_provider_and_user"
        )
        results_df_provider_user.to_csv(provider_user_output, index=False)
        print(f"\n✅ Provider+User results saved to: {provider_user_output}")

    # Generate plots
    if not args.no_plot:
        hist_plot = output_path.with_stem("actual_turns_histogram").with_suffix(".png")
        plot_actual_turns_histogram(df, hist_plot)
        hist_combined_plot = output_path.with_stem(
            "actual_turns_histogram_combined"
        ).with_suffix(".png")
        plot_actual_turns_histogram_combined(df, hist_combined_plot)

        base_stem = output_path.stem  # e.g. "max_turns_comparison"
        plot_dir = output_path.parent

        # Judge variants: (label_suffix, display_label, filtered_df)
        judge_variants: list[tuple[str, str, pd.DataFrame]] = [
            ("", "All Judges", df),
        ]
        if "judge_model" in df.columns:
            for jm in sorted(df["judge_model"].dropna().unique()):
                jm_df_raw = df[df["judge_model"] == jm]
                assert isinstance(jm_df_raw, pd.DataFrame)
                jm_df: pd.DataFrame = jm_df_raw
                # Derive a short file-safe suffix from the judge name
                suffix = "_" + display_name(jm).lower().replace(" ", "_").replace(
                    "-", ""
                ).replace(".", "")
                judge_variants.append((suffix, display_name(jm), jm_df))

        for j_suffix, j_label, j_df in judge_variants:
            print(f"\n📊 Generating plots for judge filter: {j_label}")
            cmp = compare_max_turns_by_provider(j_df, thresholds=TURN_THRESHOLDS)
            cmp_user = compare_max_turns_by_provider_and_user(
                j_df, thresholds=TURN_THRESHOLDS
            )

            judge_title = "" if j_label == "All Judges" else f" ({j_label})"

            provider_plot = plot_dir / f"{base_stem}_by_provider{j_suffix}.png"
            provider_user_plot = (
                plot_dir / f"{base_stem}_by_provider_and_user{j_suffix}.png"
            )

            if not cmp.empty:
                plot_comparison(
                    cmp,
                    provider_plot,
                    title=f"VERA Score by Provider: Simulated Max Turns{judge_title}",
                    by_provider=True,
                    thresholds=TURN_THRESHOLDS,
                )

            if not cmp_user.empty:
                plot_comparison_by_provider_and_user(
                    cmp_user,
                    provider_user_plot,
                    title=f"VERA Score vs Simulated Max Turns by Provider and User Agent{judge_title}",
                    thresholds=TURN_THRESHOLDS,
                )

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY - BY PROVIDER")
    print("=" * 80)
    if not comparison_df_provider.empty:
        print(comparison_df_provider.to_string(index=False))
    else:
        print("No provider comparison data")

    print("\n" + "=" * 80)
    print("RESULTS - BY PROVIDER AND USER")
    print("=" * 80)
    if not results_df_provider_user.empty:
        print(results_df_provider_user.to_string(index=False))
    else:
        print("No provider+user results data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
