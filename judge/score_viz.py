"""
Visualization functions for scoring results.

This module contains all matplotlib-based visualization functions for creating
charts and graphs from scoring results.
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle

matplotlib.use("Agg")  # Use non-interactive backend

from .constants import (
    BEST_PRACTICE,
    BEST_PRACTICE_KEY,
    DAMAGING,
    DAMAGING_KEY,
    MUTED_GRAY,
    MUTED_GREEN,
    MUTED_RED,
    MUTED_YELLOW,
    NEUTRAL,
    NEUTRAL_KEY,
    NOT_RELEVANT,
    NOT_RELEVANT_KEY,
)
from .score_utils import DIMENSIONS

# Risk level order for consistent sorting
RISK_LEVEL_ORDER = ["None", "Low", "High", "Imminent"]

# Visualization constants
VIZ_FIG_SIZE = (8, 12)
VIZ_PIE_HEIGHT_RATIO = 1
VIZ_BAR_HEIGHT_RATIO = 2
VIZ_BAR_HEIGHT = 0.6
VIZ_MIN_LABEL_PCT = 5.0


def _create_pie_chart(ax, results: Dict[str, Any]):
    """Create pie chart for overall percentages."""
    agg = results["aggregates"]
    pie_labels = [DAMAGING, NEUTRAL, BEST_PRACTICE]
    pie_sizes = [
        agg["overall_damaging_pct"],
        agg["overall_neutral_pct"],
        agg["overall_best_practice_pct"],
    ]
    colors = [MUTED_RED, MUTED_YELLOW, MUTED_GREEN]

    overall_vera_score = agg.get("vera_score", 0.0)
    pie_title = (
        f"Overall VERA-MH v1 Score: {overall_vera_score:.1f}\n\nRating Distribution"
    )

    _, _, autotexts = ax.pie(
        pie_sizes,
        labels=pie_labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title(pie_title, fontsize=14, fontweight="bold", pad=20)

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")


def _create_stacked_bar_chart(ax, results: Dict[str, Any]):
    """Create stacked bar chart for dimension breakdown."""
    dimensions = list(results["dimensions"].keys())[::-1]  # Reverse order
    best_practice_pcts = [
        results["dimensions"][dim][BEST_PRACTICE_KEY + "_pct"] for dim in dimensions
    ]
    neutral_pcts = [
        results["dimensions"][dim][NEUTRAL_KEY + "_pct"] for dim in dimensions
    ]
    damaging_pcts = [
        results["dimensions"][dim][DAMAGING_KEY + "_pct"] for dim in dimensions
    ]

    y_pos = range(len(dimensions))
    ax.barh(
        y_pos, damaging_pcts, VIZ_BAR_HEIGHT, label=DAMAGING, color=MUTED_RED, left=0
    )
    ax.barh(
        y_pos,
        neutral_pcts,
        VIZ_BAR_HEIGHT,
        left=damaging_pcts,
        label=NEUTRAL,
        color=MUTED_YELLOW,
    )
    ax.barh(
        y_pos,
        best_practice_pcts,
        VIZ_BAR_HEIGHT,
        left=[d + n for d, n in zip(damaging_pcts, neutral_pcts)],
        label=BEST_PRACTICE,
        color=MUTED_GREEN,
    )

    ax.set_xlabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dimension", fontsize=12, fontweight="bold")
    ax.set_title(
        "Rating Breakdown by Dimension", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dimensions, fontsize=9, ha="right")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.55, 1.02), fontsize=10, frameon=True)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    _add_bar_labels(ax, best_practice_pcts, neutral_pcts, damaging_pcts)


def _add_bar_labels(
    ax, best_practice_pcts: list, neutral_pcts: list, damaging_pcts: list
):
    """Add percentage labels to stacked bar chart."""
    for i, (bp, neu, dmg) in enumerate(
        zip(best_practice_pcts, neutral_pcts, damaging_pcts)
    ):
        if dmg > VIZ_MIN_LABEL_PCT:
            ax.text(
                dmg / 2,
                i,
                f"{dmg:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )
        if neu > VIZ_MIN_LABEL_PCT:
            ax.text(
                dmg + neu / 2,
                i,
                f"{neu:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )
        if bp > VIZ_MIN_LABEL_PCT:
            ax.text(
                dmg + neu + bp / 2,
                i,
                f"{bp:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )


def create_visualizations(results: Dict[str, Any], output_path: Path):
    """
    Create visualizations: pie chart for overall percentages and
    stacked bar chart for dimension breakdown.

    Args:
        results: Dictionary containing all scores
        output_path: Path to save the visualization (PDF or PNG)
    """
    judge_model = results.get("judge_model", "Unknown")
    persona_model = results.get("persona_model", "Unknown")
    agent_model = results.get("agent_model", "Unknown")
    title = f"Judge: {judge_model} | Persona: {persona_model} | Agent: {agent_model}"

    fig = plt.figure(figsize=VIZ_FIG_SIZE)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[VIZ_PIE_HEIGHT_RATIO, VIZ_BAR_HEIGHT_RATIO], hspace=0.3
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    _create_pie_chart(ax1, results)
    _create_stacked_bar_chart(ax2, results)

    plt.tight_layout(rect=[0.18, 0, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Visualizations saved to: {output_path}")


def _create_risk_dimension_chart(ax, dimension: str, risk_level_scores: Dict):
    """Create stacked bar chart for a single dimension across risk levels."""
    risk_levels = []
    best_practice_pcts = []
    neutral_pcts = []
    damaging_pcts = []
    not_relevant_pcts = []

    for risk_level in RISK_LEVEL_ORDER:
        if risk_level not in risk_level_scores:
            continue

        dim_scores = risk_level_scores[risk_level].get("dimensions", {}).get(dimension)
        if not dim_scores:
            continue

        risk_levels.append(risk_level)
        best_practice_pcts.append(dim_scores[BEST_PRACTICE_KEY + "_pct"])
        neutral_pcts.append(dim_scores[NEUTRAL_KEY + "_pct"])
        damaging_pcts.append(dim_scores[DAMAGING_KEY + "_pct"])
        not_relevant_pcts.append(dim_scores[NOT_RELEVANT_KEY + "_pct"])

    if not risk_levels:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(dimension, fontsize=12, fontweight="bold")
        return

    x_pos = range(len(risk_levels))
    width = 0.7

    ax.bar(x_pos, damaging_pcts, width, label=DAMAGING, color=MUTED_RED, bottom=0)
    ax.bar(
        x_pos,
        neutral_pcts,
        width,
        bottom=damaging_pcts,
        label=NEUTRAL,
        color=MUTED_YELLOW,
    )
    ax.bar(
        x_pos,
        best_practice_pcts,
        width,
        bottom=[d + n for d, n in zip(damaging_pcts, neutral_pcts)],
        label=BEST_PRACTICE,
        color=MUTED_GREEN,
    )
    ax.bar(
        x_pos,
        not_relevant_pcts,
        width,
        bottom=[
            d + n + bp
            for d, n, bp in zip(damaging_pcts, neutral_pcts, best_practice_pcts)
        ],
        label=NOT_RELEVANT,
        color=MUTED_GRAY,
    )

    _add_risk_bar_labels(
        ax, damaging_pcts, neutral_pcts, best_practice_pcts, not_relevant_pcts
    )

    ax.set_xlabel("Persona Risk Level", fontsize=10, fontweight="bold")
    ax.set_ylabel("Proportion", fontsize=10, fontweight="bold")
    ax.set_title(dimension, fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(risk_levels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def _add_risk_bar_labels(
    ax,
    damaging_pcts: list,
    neutral_pcts: list,
    best_practice_pcts: list,
    not_relevant_pcts: list,
):
    """Add percentage labels to risk level bars."""
    for i, (dmg, neu, bp, nr) in enumerate(
        zip(damaging_pcts, neutral_pcts, best_practice_pcts, not_relevant_pcts)
    ):
        if dmg > VIZ_MIN_LABEL_PCT:
            ax.text(
                i,
                dmg / 2,
                f"{dmg:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )
        if neu > VIZ_MIN_LABEL_PCT:
            ax.text(
                i,
                dmg + neu / 2,
                f"{neu:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )
        if bp > VIZ_MIN_LABEL_PCT:
            ax.text(
                i,
                dmg + neu + bp / 2,
                f"{bp:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )
        if nr > VIZ_MIN_LABEL_PCT:
            ax.text(
                i,
                dmg + neu + bp + nr / 2,
                f"{nr:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )


def _add_risk_legend(fig, gs, n_dims: int, n_rows: int, n_cols: int):
    """Add legend to risk level visualization."""
    if n_dims < n_rows * n_cols:
        legend_row = n_rows - 1
        legend_col = n_cols - 1
        ax_legend = fig.add_subplot(gs[legend_row, legend_col])
        ax_legend.axis("off")

        handles = [
            Rectangle((0, 0), 1, 1, facecolor=MUTED_RED, edgecolor="black"),
            Rectangle((0, 0), 1, 1, facecolor=MUTED_YELLOW, edgecolor="black"),
            Rectangle((0, 0), 1, 1, facecolor=MUTED_GREEN, edgecolor="black"),
            Rectangle((0, 0), 1, 1, facecolor=MUTED_GRAY, edgecolor="black"),
        ]
        labels = [DAMAGING, NEUTRAL, BEST_PRACTICE, NOT_RELEVANT]
        ax_legend.legend(handles, labels, loc="center", fontsize=10, frameon=True)


def create_risk_level_visualizations(results: Dict[str, Any], output_path: Path):
    """
    Create visualizations split by risk level with all rating
    categories including Not Relevant.

    Args:
        results: Dictionary containing scores by risk level
        output_path: Path to save the visualization
    """
    risk_level_scores = results.get("risk_level_scores", {})
    if not risk_level_scores:
        print("⚠️  No risk level data to visualize")
        return

    judge_model = results.get("judge_model", "Unknown")
    persona_model = results.get("persona_model", "Unknown")
    agent_model = results.get("agent_model", "Unknown")
    title = f"Judge: {judge_model} | Persona: {persona_model} | Agent: {agent_model}"

    n_dims = len(DIMENSIONS)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(18, 6 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.3)

    for dim_idx, dimension in enumerate(DIMENSIONS):
        row = dim_idx // n_cols
        col = dim_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        _create_risk_dimension_chart(ax, dimension, risk_level_scores)

    _add_risk_legend(fig, gs, n_dims, n_rows, n_cols)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Risk level visualizations saved to: {output_path}")
