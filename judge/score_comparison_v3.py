#!/usr/bin/env python3
"""
Compare VERA scores across multiple evaluation runs (v3).

Modern card-based visualization showing:
- Overall VERA Safety Score (with numbers in colored boxes)
- Dimension scores as colored circles
- Horizontal legend at top right

Usage:
    python -m judge.score_comparison_v3
    python -m judge.score_comparison_v3 --input evaluations_to_compare.csv
    python -m judge.score_comparison_v3 -i my_evaluations.csv -o output.png
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend

from .score_utils import (
    BG_COLOR,
    COLOR_GREEN,
    COLOR_RED,
    DIMENSION_SHORT_NAMES,
    DIMENSIONS,
    TEXT_COLOR,
    calculate_dimension_scores,
    calculate_vera_score,
    ensure_results_csv,
    save_detailed_breakdown_csv,
)

# V3 additional color for gradient
COLOR_WHITE = "#FFFFFF"


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple (0-255)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB tuple (0-255) to hex color."""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def interpolate_color(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors. t=0 returns color1, t=1 returns color2."""
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    rgb = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * t for i in range(3))
    return rgb_to_hex(rgb)


def get_color_for_score_v3(score: float) -> str:
    """
    Get the color for a VERA score using gradient (v3 uses 0-100 range).

    Uses a gradient from COLOR_RED (0) to white (50) to COLOR_GREEN (100).
    """
    if score < 50:
        # Gradient from COLOR_RED (0) to white (50)
        t = score / 50  # 0 -> 0, 50 -> 1
        return interpolate_color(COLOR_RED, COLOR_WHITE, t)
    else:
        # Gradient from white (50) to COLOR_GREEN (100)
        t = (score - 50) / 50  # 50 -> 0, 100 -> 1
        return interpolate_color(COLOR_WHITE, COLOR_GREEN, t)


# Layout colors (additional colors specific to this visualization)
CARD_COLOR = "#FFFFFF"  # White card
HEADER_BAR_COLOR = "#D4D9D4"  # Light gray for dimension header bar
SUBTLE_TEXT = "#666666"  # Lighter text for subtitles


def load_evaluation_data(
    input_path: Path,
) -> List[Dict[str, Any]]:
    """
    Load evaluation data from input CSV file.

    Args:
        input_path: Path to CSV file with "Provider Model" and "Path" columns.
                    Path can be a single path or multiple paths separated by ";"

    Returns:
        List of dicts with model_name, vera_score, and dimensions data
    """
    # Read the input CSV file
    input_df = pd.read_csv(input_path)

    # Normalize column names (handle potential whitespace)
    input_df.columns = input_df.columns.str.strip()

    results = []

    for _, row in input_df.iterrows():
        model_name = str(row.get("Provider Model", "")).strip()
        eval_paths_str = str(row.get("Path", "")).strip()

        if not model_name or not eval_paths_str:
            continue

        # Split paths by semicolon (supports single path or multiple paths)
        eval_paths = [p.strip() for p in eval_paths_str.split(";") if p.strip()]

        # Collect all dataframes from all paths
        all_dfs = []
        for eval_path in eval_paths:
            # Ensure results.csv exists (regenerate from TSV files if needed)
            try:
                df = ensure_results_csv(eval_path)
                all_dfs.append(df)
            except FileNotFoundError as e:
                print(f"⚠️  Warning: {e}")
                continue
            except Exception as e:
                print(f"⚠️  Warning: Error loading {eval_path}: {e}")
                continue

        if not all_dfs:
            print(f"⚠️  Warning: No valid data found for {model_name}")
            continue

        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Calculate dimension scores and overall counts from combined data
        dimension_scores, overall_counts = calculate_dimension_scores(combined_df)

        # Calculate overall percentages
        total = overall_counts.get("total", 0)
        bp_count = overall_counts.get("bp_count", 0)
        hph_count = overall_counts.get("hph_count", 0)
        overall_bp_pct = round(100.0 * bp_count / total, 1) if total > 0 else 0.0
        overall_hph_pct = round(100.0 * hph_count / total, 1) if total > 0 else 0.0

        # Calculate VERA score from overall raw counts
        vera_score = calculate_vera_score(overall_counts)

        results.append(
            {
                "model_name": model_name,
                "vera_score": vera_score,
                "overall_bp_pct": overall_bp_pct,
                "overall_hph_pct": overall_hph_pct,
                "dimensions": dimension_scores,
            }
        )

    return results


def create_comparison_graphic(model_data: List[Dict[str, Any]], output_path: Path):
    """
    Create a modern card-based comparison graphic.

    Args:
        model_data: List of dicts with model_name, vera_score, and dimensions
        output_path: Path to save the visualization
    """
    if not model_data:
        print("❌ No data to visualize")
        return

    # Sort models by VERA score (highest to lowest)
    sorted_data = sorted(model_data, key=lambda m: m["vera_score"], reverse=True)

    # Get dimension short names in order
    dim_headers = []
    for dim in DIMENSIONS:
        short_name = DIMENSION_SHORT_NAMES.get(dim, dim)
        if short_name not in dim_headers:  # Avoid duplicates (Confirms risk)
            dim_headers.append(short_name)

    n_models = len(sorted_data)
    n_dims = len(dim_headers)

    # Figure dimensions
    fig_width = 16
    row_height = 0.55
    fig_height = 3.5 + n_models * row_height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # Layout constants
    margin = 0.5
    header_height = 1.8
    card_top = fig_height - header_height - 0.3
    card_left = margin
    card_right = fig_width - margin
    card_width = card_right - card_left

    # New layout: Models | Dimensions | Score (at end)
    model_col_width = 2.8
    score_col_width = 1.8  # Wider score column at end with gradient background

    # Dimensions section (between models and score)
    dim_section_left = card_left + model_col_width + 0.3
    dim_section_right = card_right - score_col_width - 0.3
    dim_section_width = dim_section_right - dim_section_left
    dim_col_width = dim_section_width / n_dims

    # Score section at the far right
    score_section_left = card_right - score_col_width

    # Row layout
    header_row_height = 0.7
    dim_header_bar_height = 0.45
    data_row_height = row_height
    card_bottom = (
        card_top
        - header_row_height
        - dim_header_bar_height
        - (n_models * data_row_height)
        - 0.6
    )

    # === HEADER SECTION (outside card) ===

    # Title
    ax.text(
        margin,
        fig_height - 0.4,
        "VERA-MH safety score: Suicide risk",
        fontsize=22,
        fontweight="bold",
        color=TEXT_COLOR,
        va="top",
        fontfamily="sans-serif",
    )

    # Subtitle
    ax.text(
        margin,
        fig_height - 0.95,
        "Scores indicate how well models detect and respond to suicide risk",
        fontsize=11,
        color=SUBTLE_TEXT,
        va="top",
        fontfamily="sans-serif",
    )

    # Legend (top right) - Gradient bar
    legend_width = 4.0
    legend_height = 0.35
    legend_right = card_right
    legend_y = fig_height - 0.5
    legend_left = legend_right - legend_width

    # Draw gradient bar using many thin rectangles
    n_segments = 100
    segment_width = legend_width / n_segments
    for i in range(n_segments):
        # Score goes from 0 (left) to 100 (right)
        score = (i / n_segments) * 100
        color = get_color_for_score_v3(score)
        segment_x = legend_left + i * segment_width
        rect = mpatches.Rectangle(
            (segment_x, legend_y - legend_height),
            segment_width + 0.01,  # Slight overlap to avoid gaps
            legend_height,
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(rect)

    # Draw thin gray line at 50 (center)
    center_x = legend_left + legend_width / 2
    ax.plot(
        [center_x, center_x],
        [legend_y - legend_height, legend_y],
        color="#999999",
        linewidth=1,
        zorder=10,
    )

    # Legend border
    legend_border = mpatches.FancyBboxPatch(
        (legend_left, legend_y - legend_height),
        legend_width,
        legend_height,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor="none",
        edgecolor="#CCCCCC",
        linewidth=1,
    )
    ax.add_patch(legend_border)

    # Value labels: 0, 50, 100
    ax.text(
        legend_left, legend_y + 0.08, "0", fontsize=8, color=SUBTLE_TEXT, ha="center"
    )
    ax.text(
        legend_left + legend_width / 2,
        legend_y + 0.08,
        "50",
        fontsize=8,
        color=SUBTLE_TEXT,
        ha="center",
    )
    ax.text(
        legend_right, legend_y + 0.08, "100", fontsize=8, color=SUBTLE_TEXT, ha="center"
    )

    # Description labels: "Unsafe" centered at 25, "Safe" centered at 75
    ax.text(
        legend_left + legend_width * 0.25,
        legend_y - legend_height - 0.12,
        "Unsafe",
        fontsize=9,
        color=SUBTLE_TEXT,
        ha="center",
        va="top",
    )
    ax.text(
        legend_left + legend_width * 0.75,
        legend_y - legend_height - 0.12,
        "Safe",
        fontsize=9,
        color=SUBTLE_TEXT,
        ha="center",
        va="top",
    )

    # === MAIN CARD ===

    card = mpatches.FancyBboxPatch(
        (card_left, card_bottom),
        card_width,
        card_top - card_bottom,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=CARD_COLOR,
        edgecolor="#E0E0E0",
        linewidth=1,
    )
    ax.add_patch(card)

    # === DIMENSION HEADER BAR ===
    dim_bar_y = card_top - 0.15
    dim_bar = mpatches.FancyBboxPatch(
        (dim_section_left - 0.1, dim_bar_y - dim_header_bar_height),
        dim_section_width + 0.2,
        dim_header_bar_height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=HEADER_BAR_COLOR,
        edgecolor="none",
    )
    ax.add_patch(dim_bar)

    # Dimension header bar label
    ax.text(
        dim_section_left + dim_section_width / 2,
        dim_bar_y - dim_header_bar_height / 2,
        "Safety measures: Suicide risk",
        fontsize=10,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="center",
        va="center",
    )

    # === COLUMN HEADERS ===
    col_header_y = dim_bar_y - dim_header_bar_height - 0.15

    # "Models" header
    ax.text(
        card_left + 0.4,
        col_header_y,
        "Models",
        fontsize=11,
        fontweight="bold",
        color=TEXT_COLOR,
        va="top",
    )

    # Dimension column headers with manual line wrapping
    dim_header_wrapped = {
        "Detects potential risk": "Detects\npotential risk",
        "Confirms risk": "Confirms\nrisk",
        "Guides to human care": "Guides to\nhuman care",
        "Supportive conversation": "Supportive\nconversation",
        "Follows AI boundaries": "Follows AI\nboundaries",
    }

    for i, dim_name in enumerate(dim_headers):
        dim_x = dim_section_left + i * dim_col_width + dim_col_width / 2
        wrapped_name = dim_header_wrapped.get(dim_name, dim_name)
        ax.text(
            dim_x,
            col_header_y,
            wrapped_name,
            fontsize=9,
            color=TEXT_COLOR,
            ha="center",
            va="top",
            linespacing=1.1,
        )

    # "Score" header - centered over score section at end
    ax.text(
        score_section_left + score_col_width / 2,
        col_header_y,
        "Score",
        fontsize=11,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="center",
        va="top",
    )

    # === DATA ROWS ===
    first_row_y = col_header_y - header_row_height

    for row_idx, model in enumerate(sorted_data):
        row_y = first_row_y - row_idx * data_row_height

        # Model name
        ax.text(
            card_left + 0.4,
            row_y,
            model["model_name"],
            fontsize=10,
            fontweight="bold",
            color=TEXT_COLOR,
            va="center",
        )

        # Dimension circles with score labels (smaller circles)
        circle_radius = 0.11
        for dim_idx, dim_name in enumerate(dim_headers):
            # Find the actual dimension key
            actual_dim = None
            for d in DIMENSIONS:
                if DIMENSION_SHORT_NAMES.get(d, d) == dim_name:
                    actual_dim = d
                    break

            if actual_dim and actual_dim in model["dimensions"]:
                dim_score = model["dimensions"][actual_dim].get("vera_score", 0.0)
            else:
                dim_score = 0.0

            dim_color = get_color_for_score_v3(dim_score)
            dim_x = dim_section_left + dim_idx * dim_col_width + dim_col_width / 2

            # Border color: green for > 50, red for < 50, grey for exactly 50
            if dim_score > 50:
                border_color = COLOR_GREEN
            elif dim_score < 50:
                border_color = COLOR_RED
            else:
                border_color = "#CCCCCC"  # Grey for exactly 50 (neutral)

            circle = mpatches.Circle(
                (dim_x, row_y),
                circle_radius,
                facecolor=dim_color,
                edgecolor=border_color,
                linewidth=1.5,
            )
            ax.add_patch(circle)

            # Score label to the right of circle (light gray)
            ax.text(
                dim_x + circle_radius + 0.08,
                row_y,
                f"{int(round(dim_score))}",
                fontsize=8,
                color="#999999",
                ha="left",
                va="center",
            )

        # Score section at end with gradient background
        score = model["vera_score"]
        score_color = get_color_for_score_v3(score)

        # Light purple/blue gradient background for score area
        score_bg_rect = mpatches.FancyBboxPatch(
            (score_section_left + 0.1, row_y - data_row_height / 2 + 0.05),
            score_col_width - 0.2,
            data_row_height - 0.1,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="#F0F0FF",
            edgecolor="none",
        )
        ax.add_patch(score_bg_rect)

        # Large score number
        ax.text(
            score_section_left + score_col_width / 2 - 0.15,
            row_y,
            f"{int(round(score))}",
            fontsize=14,
            fontweight="bold",
            color=TEXT_COLOR,
            ha="center",
            va="center",
        )

        # Small colored circle next to the score
        small_circle_radius = 0.12
        small_circle_x = score_section_left + score_col_width - 0.35
        if score > 50:
            border_color = COLOR_GREEN
        elif score < 50:
            border_color = COLOR_RED
        else:
            border_color = "#CCCCCC"

        score_circle = mpatches.Circle(
            (small_circle_x, row_y),
            small_circle_radius,
            facecolor=score_color,
            edgecolor=border_color,
            linewidth=1.5,
        )
        ax.add_patch(score_circle)

    # === FOOTER NOTE ===
    # Check if all models are below 50 (indicating harm detected)
    all_have_harm = all(m["vera_score"] < 50 for m in sorted_data)
    if all_have_harm:
        footer_text = (
            "All evaluated models scored below 50\n(harmful responses detected)."
        )
        ax.text(
            card_left + 0.4,
            card_bottom + 0.25,
            footer_text,
            fontsize=9,
            fontstyle="italic",
            color=SUBTLE_TEXT,
            va="bottom",
        )

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()

    print(f"📊 Comparison graphic saved to: {output_path}")

    # Build and save CSV data
    rows = []
    for model in sorted_data:
        row = {
            "Model": model["model_name"],
            "VERA Safety Index": round(model["vera_score"], 2),
        }
        for dim in DIMENSIONS:
            short_name = DIMENSION_SHORT_NAMES.get(dim, dim)
            vera_dim = model["dimensions"].get(dim, {}).get("vera_score", 0.0)
            col_name = f"VERA: {short_name}"
            if col_name not in row:  # Avoid duplicate columns
                row[col_name] = round(vera_dim)
        rows.append(row)

    display_df = pd.DataFrame(rows)

    csv_path = output_path.with_suffix(".csv")
    display_df.to_csv(csv_path, index=False)
    print(f"📄 Comparison data saved to: {csv_path}")

    # Save detailed breakdown CSV using shared function
    save_detailed_breakdown_csv(sorted_data, output_path, "v3")

    print("\n" + "=" * 80)
    print("VERA SCORE COMPARISON DATA (v3)")
    print("=" * 80)
    print(display_df.to_string(index=False))
    print("=" * 80)


def main():
    """Main entry point for score comparison v3."""
    parser = argparse.ArgumentParser(
        description="Compare VERA scores across multiple evaluation runs (v3)"
    )

    parser.add_argument(
        "--input",
        "-i",
        default="evaluations_to_compare.csv",
        help=(
            "Path to CSV file with 'Provider Model' and 'Path' columns "
            "(default: evaluations_to_compare.csv)"
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Path to save the output visualization "
            "(default: score_comparisons/{input_filename}_v3_output.png)"
        ),
    )

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    # Set default output path based on input filename
    if args.output:
        output_path = Path(args.output)
    else:
        # Use input filename with _output suffix in score_comparisons directory
        output_path = Path("score_comparisons") / f"{input_path.stem}_v3_output.png"

    print(f"📥 Loading evaluations from: {input_path}")

    # Load evaluation data
    model_data = load_evaluation_data(input_path)

    if not model_data:
        print("❌ Error: No valid evaluation data found")
        return 1

    print(f"✅ Loaded {len(model_data)} evaluations")

    # Create comparison graphic
    create_comparison_graphic(model_data, output_path)

    print(f"✅ Comparison complete: {output_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
