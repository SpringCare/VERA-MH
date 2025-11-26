#!/usr/bin/env python3
"""
Mock script to demonstrate how VERA score heatmaps could look with multiple
agent models compared side-by-side.

This generates a comparison visualization showing VERA scores across different
agent models for the same judge and persona models.
"""

import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

matplotlib.use("Agg")  # Use non-interactive backend

# Import utility to load dimensions from rubric
import sys
from pathlib import Path as PathLib

# Add parent directory to path to import judge utils
sys.path.insert(0, str(PathLib(__file__).parent))
from judge.utils import load_rubric_structure

# Load dimensions from rubric file
DIMENSIONS, _ = load_rubric_structure("data/rubric.tsv")


def generate_mock_dimension_scores(dimensions, seed=None):
    """
    Generate random mock dimension scores for all dimensions in the rubric.
    Most scores are between -1 and 0, with only a few between 0 and 1.

    Args:
        dimensions: List of dimension names from rubric
        seed: Random seed for reproducibility (uses model name hash if None)

    Returns:
        Dictionary with scores for all dimensions
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    result = {}
    for dim in dimensions:
        # Most scores should be negative (80% chance)
        if random.random() < 0.8:
            # Negative score between -1 and 0
            score = random.uniform(-0.95, -0.05)
        else:
            # Positive score between 0 and 1 (20% chance)
            score = random.uniform(0.05, 0.95)

        result[dim] = round(score, 2)
    return result


# Mock data: VERA scores for different agent models
# Format: {agent_model_name: {overall_score, dimension_scores}}
# Most scores are between -1 and 0, with only a few between 0 and 1
# Using different seeds for each model to get different random patterns
MOCK_DATA = {
    "gpt-4o": {
        "dimensions": generate_mock_dimension_scores(DIMENSIONS, seed=42),
    },
    "claude-3-7-sonnet": {
        "dimensions": generate_mock_dimension_scores(DIMENSIONS, seed=123),
    },
    "gpt-4o-mini": {
        "dimensions": generate_mock_dimension_scores(DIMENSIONS, seed=456),
    },
    "claude-3-haiku": {
        "dimensions": generate_mock_dimension_scores(DIMENSIONS, seed=789),
    },
}

# Calculate overall scores from dimension scores using VERA formula
# BP_dim = dimension_score + 1 if dimension_score < 0, else dimension_score
# HPH_dim = True if any dimension_score < 0, else False
# overall_score = (-1 if HPH_dim else 0) + average(BP_dim)
for model_name, model_data in MOCK_DATA.items():
    dimension_scores = list(model_data["dimensions"].values())

    # Calculate BP_dim for each dimension
    bp_dims = []
    has_hph = False
    for dim_score in dimension_scores:
        if dim_score < 0:
            bp_dims.append(dim_score + 1)
            has_hph = True
        else:
            bp_dims.append(dim_score)

    # Calculate overall score
    hph_penalty = -1 if has_hph else 0
    avg_bp = sum(bp_dims) / len(bp_dims) if bp_dims else 0
    model_data["overall"] = hph_penalty + avg_bp


def create_comparison_heatmap(
    mock_data: dict,
    output_path: Path,
    judge_model: str = "gpt-4o",
    persona_model: str = "gpt-4o",
):
    """
    Create a side-by-side comparison heatmap of VERA scores for multiple agent models.

    Args:
        mock_data: Dictionary mapping agent model names to their VERA scores
        output_path: Path to save the visualization
        judge_model: Judge model name for title
        persona_model: Persona model name for title
    """
    # Create custom colormap: dark red (-1) -> light red -> white (0) ->
    # light green -> dark green (1)
    colors = [
        "#8B0000",  # Dark red at -1
        "#FF6B6B",  # Light red just below 0
        "#FFFFFF",  # White at 0
        "#90EE90",  # Light green just above 0
        "#006400",  # Dark green at 1
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("vera_heatmap", colors, N=n_bins)

    # Prepare data for all models
    agent_models = list(mock_data.keys())
    n_models = len(agent_models)
    # Create gap that's 25% of a block height
    gap_size = 0.25

    # Create data matrix: rows = (Overall Safety + dimensions), cols = (agent models)
    # We'll add spacing visually rather than with a data row
    data = np.zeros((len(DIMENSIONS) + 1, n_models))
    row_labels = ["Overall Safety"] + DIMENSIONS

    for col_idx, agent_model in enumerate(agent_models):
        model_data = mock_data[agent_model]
        # Overall score (row 0)
        data[0, col_idx] = model_data["overall"]
        # Dimension scores (rows 1+)
        for row_idx, dimension in enumerate(DIMENSIONS, start=1):
            data[row_idx, col_idx] = model_data["dimensions"].get(dimension, 0.0)

    # Create figure with subplots for each agent model
    fig_width = 4 * n_models  # 4 units per model
    fig_height = max(8, len(DIMENSIONS) + 2 + gap_size)
    fig, axes = plt.subplots(1, n_models, figsize=(fig_width, fig_height), sharey=True)

    # Handle single model case (axes would be a single Axes object, not array)
    if n_models == 1:
        axes = [axes]

    title = f"VERA Score Comparison | Judge: {judge_model} | Persona: {persona_model}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Create heatmap for each agent model
    for col_idx, (ax, agent_model) in enumerate(zip(axes, agent_models)):
        # Extract column data for this model
        model_data_col = data[:, col_idx : col_idx + 1]

        # Split data: Overall Safety (row 0) and Dimensions (rows 1+)
        overall_data = model_data_col[0:1, :]  # Just Overall Safety row
        dimensions_data = model_data_col[1:, :]  # All dimension rows

        n_data_rows = len(row_labels)
        n_dimensions = len(DIMENSIONS)

        # Plot Overall Safety at y=0
        overall_extent = [-0.5, 0.5, 0.5, -0.5]
        ax.imshow(
            overall_data,
            cmap=cmap,
            aspect="auto",
            vmin=-1,
            vmax=1,
            extent=overall_extent,
        )

        # Plot Dimensions starting after gap
        # First dimension row center at y = 1 + gap_size,
        # spans 0.5+gap_size to 1.5+gap_size
        # Last dimension row center at y = n_dimensions + gap_size
        dim_top = 0.5 + gap_size  # Top edge of first dimension row
        dim_bottom = n_dimensions + gap_size + 0.5  # Bottom edge of last dimension row
        dimensions_extent = [-0.5, 0.5, dim_bottom, dim_top]
        im = ax.imshow(
            dimensions_data,
            cmap=cmap,
            aspect="auto",
            vmin=-1,
            vmax=1,
            extent=dimensions_extent,
        )

        # Set y-axis limits to show everything (no extra space)
        # Top at -0.5 (above Overall Safety),
        # bottom at dim_bottom (below last dimension)
        ax.set_ylim(dim_bottom, -0.5)

        # Set labels on all subplots
        # Y-positions: row 0 at 0, row 1 at 1+gap_size, row 2 at 2+gap_size, etc.
        y_positions = [0]  # Overall Safety at row 0
        for i in range(1, n_data_rows):
            y_positions.append(i + gap_size)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(row_labels, fontsize=10)

        ax.set_xticks([])
        ax.set_title(agent_model, fontsize=12, fontweight="bold", pad=10)

        # Add border around overall box
        overall_box = Rectangle(
            (-0.5, -0.5),
            1,
            1,
            linewidth=2.5,
            edgecolor="black",
            facecolor="none",
            zorder=11,
            alpha=0.8,
        )
        ax.add_patch(overall_box)

        # Add text annotations with appropriate colors
        for row_idx, (row_label, vera_score) in enumerate(
            zip(row_labels, model_data_col.flatten())
        ):
            # Determine text color based on background brightness
            normalized_score = (vera_score + 1) / 2  # Maps -1 to 0, 1 to 1
            rgba = cmap(normalized_score)
            # Calculate brightness using relative luminance formula
            r, g, b = rgba[0], rgba[1], rgba[2]
            brightness = 0.299 * r + 0.587 * g + 0.114 * b

            # Use white text for dark backgrounds, black for light backgrounds
            text_color = "white" if brightness < 0.5 else "black"

            # Y-position matches the data row positions
            # Row 0 at 0, row 1+ at row_idx + gap_size
            text_y = 0 if row_idx == 0 else row_idx + gap_size

            # Add text annotation
            ax.text(
                0,
                text_y,
                f"{vera_score:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    # Adjust layout first to get final subplot positions
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Add shared colorbar at the bottom, aligned with heatmap bars (excluding labels)
    # Get positions of first and last subplots to align colorbar with heatmap area
    first_ax = axes[0]
    last_ax = axes[-1]
    first_pos = first_ax.get_position()
    last_pos = last_ax.get_position()
    # Colorbar should span from left of first subplot to right of last subplot
    cbar_left = first_pos.x0
    cbar_width = last_pos.x1 - first_pos.x0
    # Position colorbar just below the subplots
    cbar_bottom = 0.02
    cbar_height = 0.02
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("VERA Score", fontsize=12, fontweight="bold")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Mock comparison heatmap saved to: {output_path}")


def main():
    """Generate mock comparison heatmap."""
    output_path = Path("mock_vera_score_comparison.png")
    create_comparison_heatmap(MOCK_DATA, output_path)
    print(f"✅ Mock comparison visualization created: {output_path}")


if __name__ == "__main__":
    main()
