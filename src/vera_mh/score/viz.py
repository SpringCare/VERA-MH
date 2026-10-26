"""
Visualization functions for scoring results.

This module contains all matplotlib-based visualization functions for creating
charts and graphs from scoring results.
"""

from pathlib import Path
from typing import Any, Dict, NamedTuple, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle

matplotlib.use("Agg")  # Use non-interactive backend

from vera_mh.score.utils import (
    BEST_PRACTICE_KEY,
    DAMAGING_KEY,
    MUTED_GRAY,
    MUTED_GREEN,
    MUTED_RED,
    MUTED_YELLOW,
    NEUTRAL_KEY,
    NOT_RELEVANT_KEY,
    RISK_LEVEL_ORDER,
    SCORE_NAME,
)
from vera_mh.utils.ratings import BEST_PRACTICE, DAMAGING, NEUTRAL, NOT_RELEVANT

# Visualization constants
VIZ_FIG_SIZE = (8, 12)
VIZ_PIE_HEIGHT_RATIO = 1
VIZ_BAR_HEIGHT_RATIO = 2
VIZ_BAR_HEIGHT = 0.6
VIZ_MIN_LABEL_PCT = 5.0


class Segment(NamedTuple):
    """One rating's slice of a pie or stacked bar."""

    label: str
    key: str
    color: str


# Stacking order, from the bottom (or left) of a bar up.
RATED_SEGMENTS = [
    Segment(DAMAGING, DAMAGING_KEY, MUTED_RED),
    Segment(NEUTRAL, NEUTRAL_KEY, MUTED_YELLOW),
    Segment(BEST_PRACTICE, BEST_PRACTICE_KEY, MUTED_GREEN),
]
# The risk-level charts also show how often a dimension was Not Relevant.
ALL_SEGMENTS = [*RATED_SEGMENTS, Segment(NOT_RELEVANT, NOT_RELEVANT_KEY, MUTED_GRAY)]


def _models_title(results: Dict[str, Any]) -> str:
    judge_model = results.get("judge_model", "Unknown")
    persona_model = results.get("persona_model", "Unknown")
    agent_model = results.get("agent_model", "Unknown")
    return f"Judge: {judge_model} | Persona: {persona_model} | Agent: {agent_model}"


def _save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _safe_pie_value(value: Any) -> float:
    """Coerce aggregate percentage to a finite non-negative float for pie charts."""
    try:
        v = float(value)
    except TypeError, ValueError:
        return 0.0
    if v != v or v < 0:  # NaN or negative
        return 0.0
    return v


def _draw_stacked_bars(
    ax,
    positions: Sequence[int],
    segments: Sequence[Segment],
    percentages: Dict[str, list],
    *,
    horizontal: bool,
    bar_width: float,
    label_size: int,
) -> None:
    """Draw one stacked bar per position, labeling each segment with its percent.

    `percentages` maps a segment key to one value per position. Segments under
    `VIZ_MIN_LABEL_PCT` are left unlabeled so the text doesn't overflow them.
    """
    offsets = [0.0] * len(positions)
    for segment in segments:
        values = percentages[segment.key]
        if horizontal:
            ax.barh(
                positions,
                values,
                bar_width,
                left=offsets,
                label=segment.label,
                color=segment.color,
            )
        else:
            ax.bar(
                positions,
                values,
                bar_width,
                bottom=offsets,
                label=segment.label,
                color=segment.color,
            )

        for position, offset, value in zip(positions, offsets, values):
            if value <= VIZ_MIN_LABEL_PCT:
                continue
            middle = offset + value / 2
            x, y = (middle, position) if horizontal else (position, middle)
            ax.text(
                x,
                y,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=label_size,
                fontweight="bold",
                color="white",
            )

        offsets = [offset + value for offset, value in zip(offsets, values)]


def _create_pie_chart(ax, results: Dict[str, Any]):
    """Create pie chart for overall percentages."""
    agg = results["aggregates"]
    pie_sizes = [
        _safe_pie_value(agg.get(f"overall_{segment.key}_pct"))
        for segment in RATED_SEGMENTS
    ]

    overall_vera_score = agg.get("vera_score", 0.0)
    try:
        score_display = float(overall_vera_score)
        if score_display != score_display:
            score_display = 0.0
    except TypeError, ValueError:
        score_display = 0.0
    pie_title = (
        f"Overall {SCORE_NAME} Score: {score_display:.1f}\n\nRating Distribution"
    )

    # Matplotlib pie fails (NaN wedge angles) when every slice is zero.
    if sum(pie_sizes) <= 0:
        ax.set_title(pie_title, fontsize=14, fontweight="bold", pad=20)
        ax.text(
            0.5,
            0.45,
            "No rating percentages\n(no scored dimension evaluations)",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
            color=MUTED_GRAY,
        )
        ax.set_axis_off()
        return

    _, _, autotexts = ax.pie(
        pie_sizes,
        labels=[segment.label for segment in RATED_SEGMENTS],
        colors=[segment.color for segment in RATED_SEGMENTS],
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
    percentages = {
        segment.key: [
            results["dimensions"][dim][segment.key + "_pct"] for dim in dimensions
        ]
        for segment in RATED_SEGMENTS
    }

    y_pos = range(len(dimensions))
    _draw_stacked_bars(
        ax,
        y_pos,
        RATED_SEGMENTS,
        percentages,
        horizontal=True,
        bar_width=VIZ_BAR_HEIGHT,
        label_size=8,
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


def create_visualizations(results: Dict[str, Any], output_path: Path):
    """
    Create visualizations: pie chart for overall percentages and
    stacked bar chart for dimension breakdown.

    Args:
        results: Dictionary containing all scores
        output_path: Path to save the visualization (PDF or PNG)
    """
    fig = plt.figure(figsize=VIZ_FIG_SIZE)
    fig.suptitle(_models_title(results), fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[VIZ_PIE_HEIGHT_RATIO, VIZ_BAR_HEIGHT_RATIO], hspace=0.3
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    _create_pie_chart(ax1, results)
    _create_stacked_bar_chart(ax2, results)

    plt.tight_layout(rect=(0.18, 0, 1, 1))
    _save_figure(output_path)

    print(f"📊 Visualizations saved to: {output_path}")


def _create_risk_dimension_chart(ax, dimension: str, risk_level_scores: Dict):
    """Create stacked bar chart for a single dimension across risk levels."""
    risk_levels = []
    percentages: Dict[str, list] = {segment.key: [] for segment in ALL_SEGMENTS}

    for risk_level in RISK_LEVEL_ORDER:
        if risk_level not in risk_level_scores:
            continue

        dim_scores = risk_level_scores[risk_level].get("dimensions", {}).get(dimension)
        if not dim_scores:
            continue

        risk_levels.append(risk_level)
        for segment in ALL_SEGMENTS:
            percentages[segment.key].append(dim_scores[segment.key + "_pct"])

    if not risk_levels:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(dimension, fontsize=12, fontweight="bold")
        return

    x_pos = range(len(risk_levels))
    _draw_stacked_bars(
        ax,
        x_pos,
        ALL_SEGMENTS,
        percentages,
        horizontal=False,
        bar_width=0.7,
        label_size=7,
    )

    ax.set_xlabel("Persona Risk Level", fontsize=10, fontweight="bold")
    ax.set_ylabel("Proportion", fontsize=10, fontweight="bold")
    ax.set_title(dimension, fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(risk_levels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def _add_risk_legend(fig, gs, n_dims: int, n_rows: int, n_cols: int):
    """Add legend to risk level visualization."""
    if n_dims < n_rows * n_cols:
        legend_row = n_rows - 1
        legend_col = n_cols - 1
        ax_legend = fig.add_subplot(gs[legend_row, legend_col])
        ax_legend.axis("off")

        handles = [
            Rectangle((0, 0), 1, 1, facecolor=segment.color, edgecolor="black")
            for segment in ALL_SEGMENTS
        ]
        labels = [segment.label for segment in ALL_SEGMENTS]
        ax_legend.legend(handles, labels, loc="center", fontsize=10, frameon=True)


def _dimensions_in(risk_level_scores: Dict) -> list[str]:
    """Every dimension scored at any risk level, in first-seen order."""
    dimensions: list[str] = []
    for level_scores in risk_level_scores.values():
        for dimension in level_scores.get("dimensions", {}):
            if dimension not in dimensions:
                dimensions.append(dimension)
    return dimensions


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

    dimensions = _dimensions_in(risk_level_scores)
    if not dimensions:
        print("⚠️  No risk level data to visualize")
        return

    n_dims = len(dimensions)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(18, 6 * n_rows))
    fig.suptitle(_models_title(results), fontsize=14, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.3)

    for dim_idx, dimension in enumerate(dimensions):
        row = dim_idx // n_cols
        col = dim_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        _create_risk_dimension_chart(ax, dimension, risk_level_scores)

    _add_risk_legend(fig, gs, n_dims, n_rows, n_cols)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    _save_figure(output_path)

    print(f"📊 Risk level visualizations saved to: {output_path}")
