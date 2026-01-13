#!/usr/bin/env python3
"""
Analyze VERA score variability across different sampling strategies.

This script samples evaluated conversations from evaluations/score_variability_20260109
and calculates VERA scores for different combinations of:
- N: number of conversations per persona (5, 10, 20, 50, 100)
- J: number of judge iterations per conversation (1, 3, 5)

For each sampling strategy, it calculates mean and standard deviation of VERA scores.

Usage:
    python scripts/analyze_score_variability.py
    python scripts/analyze_score_variability.py --num-samples 100
    python scripts/analyze_score_variability.py -o results/variability_analysis.csv
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from judge.score_utils import (
    BG_COLOR,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    DIMENSIONS,
    TEXT_COLOR,
    calculate_scores_from_df,
    combine_evaluations,
    parse_evaluation_filename,
)


def discover_evaluation_files(
    base_dir: Path = Path("evaluations/score_variability_20260109"),
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Discover all evaluation files organized by PROVIDER model and persona.

    Aggregates all user models together under each provider.

    Args:
        base_dir: Base directory (evaluations/score_variability_20260109)

    Returns:
        Nested dict: {provider_model: {persona: [list of evaluation files]}}
    """
    results: Dict[str, Dict[str, List[Path]]] = {}

    # Find all model directories (provider/user structure)
    for provider_dir in base_dir.iterdir():
        if not provider_dir.is_dir():
            continue

        provider_model = provider_dir.name

        if provider_model not in results:
            results[provider_model] = defaultdict(list)

        for user_dir in provider_dir.iterdir():
            if not user_dir.is_dir():
                continue

            # Find evaluation folders (j_* directories)
            for eval_folder in user_dir.iterdir():
                if not eval_folder.is_dir() or not eval_folder.name.startswith("j_"):
                    continue

                # Find all TSV files
                for tsv_file in eval_folder.glob("*.tsv"):
                    if tsv_file.name == "results.csv":
                        continue

                    parsed = parse_evaluation_filename(tsv_file.name)
                    if parsed and "persona" in parsed:
                        results[provider_model][parsed["persona"]].append(tsv_file)

    return results


def group_files_by_conversation(
    files: List[Path],
) -> Dict[str, List[Path]]:
    """
    Group evaluation files by conversation (same id + run).

    Args:
        files: List of evaluation file paths

    Returns:
        Dict mapping conversation_key to list of judge iteration files
    """
    conversations = defaultdict(list)

    for f in files:
        parsed = parse_evaluation_filename(f.name)
        if parsed:
            conv_key = f"{parsed['id']}_run{parsed['run']}"
            conversations[conv_key].append((parsed["judge_iteration"], f))

    # Sort by iteration within each conversation
    result = {}
    for conv_key, iter_files in conversations.items():
        iter_files.sort(key=lambda x: x[0])
        result[conv_key] = [f for _, f in iter_files]

    return result


def partition_conversations(
    files_by_persona: Dict[str, List[Path]],
    n_conversations: int,
    j_iterations: int,
    random_seed: Optional[int] = 42,
) -> List[List[Path]]:
    """
    Partition conversations into non-overlapping samples.

    If there are 100 conversations and N=10, creates 10 disjoint samples
    of 10 conversations each (no conversation appears in multiple samples).

    Judge iterations are also partitioned when possible:
    - J=1 with 5 iterations: 5 disjoint judge partitions
    - J=5 with 5 iterations: 1 partition using all judges
    - J=3 with 5 iterations: samples WITH replacement (can't partition cleanly)

    Args:
        files_by_persona: Dict mapping persona -> list of evaluation files
        n_conversations: Number of conversations per sample per persona
        j_iterations: Number of judge iterations to include per conversation
        random_seed: Random seed for reproducibility

    Returns:
        List of samples, where each sample is a list of evaluation file paths
    """
    if random_seed is not None:
        random.seed(random_seed)

    # First, organize conversations by persona
    persona_conversations: Dict[str, Dict[str, List[Path]]] = {}
    min_num_conv_samples = float("inf")
    available_judge_iterations = 5  # We have 5 judge iterations per conversation

    for persona, files in files_by_persona.items():
        conversations = group_files_by_conversation(files)
        conv_keys = list(conversations.keys())

        if len(conv_keys) < n_conversations:
            raise ValueError(
                f"Not enough conversations for persona '{persona}': "
                f"have {len(conv_keys)}, need {n_conversations}"
            )

        # Check judge iterations for all conversations
        for conv_key, iter_files in conversations.items():
            if len(iter_files) < j_iterations:
                raise ValueError(
                    f"Not enough judge iterations for conversation '{conv_key}': "
                    f"have {len(iter_files)}, need {j_iterations}"
                )
            # Track actual available iterations
            available_judge_iterations = min(
                available_judge_iterations, len(iter_files)
            )

        # Shuffle conversation keys
        random.shuffle(conv_keys)
        persona_conversations[persona] = {key: conversations[key] for key in conv_keys}

        # Calculate how many complete conversation samples we can make
        num_possible_conv_samples = len(conv_keys) // n_conversations
        min_num_conv_samples = min(min_num_conv_samples, num_possible_conv_samples)

    if min_num_conv_samples == 0 or min_num_conv_samples == float("inf"):
        raise ValueError("Cannot create any complete samples with the given parameters")

    num_conv_samples = int(min_num_conv_samples)

    # Calculate judge partitions
    # If J divides evenly into available iterations, partition without replacement
    # Otherwise, sample with replacement
    can_partition_judges = available_judge_iterations % j_iterations == 0
    num_judge_partitions = (
        available_judge_iterations // j_iterations if can_partition_judges else 1
    )

    # Total samples = conversation partitions * judge partitions
    total_samples = num_conv_samples * num_judge_partitions

    # Create the partitioned samples
    samples: List[List[Path]] = [[] for _ in range(total_samples)]

    for persona, conversations in persona_conversations.items():
        conv_keys = list(conversations.keys())

        for conv_sample_idx in range(num_conv_samples):
            # Get the slice of conversations for this sample
            start_idx = conv_sample_idx * n_conversations
            end_idx = start_idx + n_conversations
            sample_conv_keys = conv_keys[start_idx:end_idx]

            for judge_partition_idx in range(num_judge_partitions):
                sample_idx = (
                    conv_sample_idx * num_judge_partitions + judge_partition_idx
                )

                for conv_key in sample_conv_keys:
                    iter_files = conversations[conv_key]

                    if can_partition_judges:
                        # Partition judges without replacement
                        j_start = judge_partition_idx * j_iterations
                        j_end = j_start + j_iterations
                        samples[sample_idx].extend(iter_files[j_start:j_end])
                    else:
                        # Sample judges WITH replacement (e.g., J=3 from 5)
                        samples[sample_idx].extend(
                            random.choices(iter_files, k=j_iterations)
                        )

    return samples


def calculate_sample_scores(
    sampled_files: List[Path],
) -> Dict[str, Any]:
    """
    Calculate VERA scores for a sample of evaluation files.

    Args:
        sampled_files: List of evaluation file paths

    Returns:
        Dict with overall_score and dimension_scores
    """
    if not sampled_files:
        return {"overall_score": 0.0, "dimension_scores": {}}

    # Combine all evaluations into a DataFrame
    df = combine_evaluations([str(f) for f in sampled_files])

    if df.empty:
        return {"overall_score": 0.0, "dimension_scores": {}}

    # Calculate scores
    return calculate_scores_from_df(df)


def run_sampling_analysis(
    model_data: Dict[str, Dict[str, List[Path]]],
    n_values: List[int],
    j_values: List[int],
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Run sampling analysis for all models and sampling strategies.

    Creates non-overlapping partitions of conversations. For example,
    if there are 100 conversations and N=10, creates 10 disjoint samples.

    Args:
        model_data: Nested dict from discover_evaluation_files
        n_values: List of N values (conversations per persona)
        j_values: List of J values (judge iterations per conversation)
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with results
    """
    results = []

    for model_key, files_by_persona in model_data.items():
        print(f"\n📊 Analyzing: {model_key}")

        # Count total available
        total_files = sum(len(files) for files in files_by_persona.values())
        personas = list(files_by_persona.keys())
        print(f"   Personas: {len(personas)}, Total files: {total_files}")

        for n in n_values:
            for j in j_values:
                print(f"   Sampling N={n}, J={j}...", end=" ")

                try:
                    # Get non-overlapping partitions
                    samples = partition_conversations(
                        files_by_persona,
                        n_conversations=n,
                        j_iterations=j,
                        random_seed=random_seed,
                    )
                except ValueError as e:
                    print(f"SKIPPED: {e}")
                    continue

                num_samples = len(samples)
                sample_scores = []
                sample_dim_scores = defaultdict(list)

                for sample in samples:
                    scores = calculate_sample_scores(sample)
                    sample_scores.append(scores["overall_score"])

                    for dim, dim_data in scores.get("dimension_scores", {}).items():
                        sample_dim_scores[dim].append(dim_data.get("vera_score", 0.0))

                # Calculate statistics
                mean_score = (
                    sum(sample_scores) / len(sample_scores) if sample_scores else 0
                )
                std_score = (
                    (
                        sum((s - mean_score) ** 2 for s in sample_scores)
                        / len(sample_scores)
                    )
                    ** 0.5
                    if sample_scores
                    else 0
                )
                min_score = min(sample_scores) if sample_scores else 0
                max_score = max(sample_scores) if sample_scores else 0

                result_row = {
                    "model": model_key,
                    "N": n,
                    "J": j,
                    "num_samples": num_samples,
                    "overall_mean": round(mean_score, 2),
                    "overall_std": round(std_score, 2),
                    "overall_min": round(min_score, 2),
                    "overall_max": round(max_score, 2),
                }

                # Add dimension statistics
                for dim in DIMENSIONS:
                    if dim in sample_dim_scores and sample_dim_scores[dim]:
                        dim_scores = sample_dim_scores[dim]
                        dim_mean = sum(dim_scores) / len(dim_scores)
                        dim_std = (
                            sum((s - dim_mean) ** 2 for s in dim_scores)
                            / len(dim_scores)
                        ) ** 0.5
                        result_row[f"{dim}_mean"] = round(dim_mean, 2)
                        result_row[f"{dim}_std"] = round(dim_std, 2)
                        result_row[f"{dim}_min"] = round(min(dim_scores), 2)
                        result_row[f"{dim}_max"] = round(max(dim_scores), 2)

                results.append(result_row)
                print(
                    f"samples={num_samples}, mean={mean_score:.1f}, std={std_score:.2f}"
                )

    return pd.DataFrame(results)


def plot_std_vs_n(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "VERA Score Standard Deviation vs Sample Size",
):
    """
    Plot standard deviation of VERA scores as a function of N,
    with separate lines for each J value.

    Args:
        results_df: DataFrame with analysis results
        output_path: Path to save the plot
        title: Plot title
    """
    # Colors for different J values (matching score_comparison_v3.py)
    colors = {1: COLOR_RED, 3: COLOR_YELLOW, 5: COLOR_GREEN}
    markers = {1: "o", 3: "s", 5: "^"}

    # Get unique models
    models = results_df["model"].unique()

    # Create subplots - one per model
    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(6 * n_models, 5), squeeze=False, sharey=True
    )

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        model_df = results_df[results_df["model"] == model]

        # Get unique J values
        j_values = sorted(model_df["J"].unique())

        for j in j_values:
            j_df = model_df[model_df["J"] == j].sort_values("N")

            ax.plot(
                j_df["N"],
                j_df["overall_std"],
                marker=markers.get(j, "o"),
                color=colors.get(j, "#666666"),
                linewidth=2,
                markersize=8,
                label=f"J={j}",
            )

        ax.set_xlabel("N (conversations per persona)", fontsize=11, color=TEXT_COLOR)
        if idx == 0:
            ax.set_ylabel(
                "Standard Deviation of VERA Score", fontsize=11, color=TEXT_COLOR
            )
        ax.set_title(
            model.replace("/", "\n"), fontsize=10, fontweight="bold", color=TEXT_COLOR
        )
        ax.legend(title="Judge iterations")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_facecolor("white")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02, color=TEXT_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()

    print(f"📈 Plot saved to: {output_path}")


def plot_std_vs_n_combined(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "VERA Score Standard Deviation vs Sample Size",
):
    """
    Plot standard deviation of VERA scores as a function of N,
    with all models on one plot and separate lines for each J value.

    Args:
        results_df: DataFrame with analysis results
        output_path: Path to save the plot
        title: Plot title
    """
    # Specific colors for each model
    model_color_map = {
        "claude-opus-4-1-20250805": "#E63946",  # Red
        "gemini-3-pro-preview": "#F4A261",  # Orange
        "gemini-2.5-flash": "#E9C46A",  # Yellow
        "claude-sonnet-4-5-20250929": "#2A9D8F",  # Turquoise
        "gpt-5": "#264653",  # Dark green/blue
        "gpt-4o": "#7209B7",  # Purple
    }
    # Fallback colors for any unlisted models
    fallback_colors = ["#3A86FF", "#FF006E", "#06D6A0", "#118AB2"]

    # Markers for different J values
    j_markers = {1: "o", 3: "s", 5: "^"}
    j_styles = {1: "-", 3: "--", 5: "-."}
    j_labels = {1: "J=1", 3: "J=3", 5: "J=5"}

    fig, ax = plt.subplots(figsize=(12, 7))

    models = list(results_df["model"].unique())  # type: ignore[union-attr]
    j_values = sorted(results_df["J"].unique())  # type: ignore[union-attr]

    # Track fallback color index for unlisted models
    fallback_idx = 0

    # Plot each model with different colors, J with different markers/styles
    for model_idx, model in enumerate(models):
        # Use specific color if defined, otherwise use fallback
        if model in model_color_map:
            model_color = model_color_map[model]
        else:
            model_color = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1
        short_model = model.split("/")[-1] if "/" in model else model

        for j in j_values:
            model_j_df = results_df[
                (results_df["model"] == model) & (results_df["J"] == j)
            ].sort_values(by="N")  # type: ignore[union-attr]

            if model_j_df.empty:
                continue

            ax.plot(
                model_j_df["N"],
                model_j_df["overall_std"],
                marker=j_markers.get(j, "o"),
                color=model_color,
                linestyle=j_styles.get(j, "-"),
                linewidth=2,
                markersize=8,
                label=f"{short_model}, {j_labels.get(j, f'J={j}')}",
                alpha=0.85,
            )

    ax.set_xlabel("N (conversations per persona)", fontsize=12, color=TEXT_COLOR)
    ax.set_ylabel("Standard Deviation of VERA Score", fontsize=12, color=TEXT_COLOR)
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR)

    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor("white")
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()

    print(f"📈 Combined plot saved to: {output_path}")


def plot_score_ranges(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "VERA Score Range Across Samples",
):
    """
    Plot the range of VERA scores (min to max) for each N,J combination.

    Shows mean with shaded area indicating the min-max range.

    Args:
        results_df: DataFrame with results (must include overall_min, overall_max)
        output_path: Path to save the plot
        title: Plot title
    """
    models = list(results_df["model"].unique())  # type: ignore[union-attr]
    n_models = len(models)

    # Create subplots - one per model
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    # J value styling
    j_colors = {1: COLOR_RED, 3: COLOR_YELLOW, 5: COLOR_GREEN}
    j_labels = {1: "J=1 (single judge)", 3: "J=3", 5: "J=5 (5 judges)"}
    j_alphas = {1: 0.2, 3: 0.25, 5: 0.3}

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        model_df = results_df[results_df["model"] == model]

        j_values = sorted(model_df["J"].unique())  # type: ignore[union-attr]

        for j in j_values:
            model_j_df = model_df[model_df["J"] == j].sort_values(by="N")  # type: ignore[union-attr]

            if model_j_df.empty:
                continue

            n_vals = model_j_df["N"].values
            means = model_j_df["overall_mean"].values
            mins = model_j_df["overall_min"].values
            maxs = model_j_df["overall_max"].values

            color = j_colors.get(j, "#888888")
            alpha = j_alphas.get(j, 0.2)
            label = j_labels.get(j, f"J={j}")

            # Plot shaded range (min to max)
            ax.fill_between(
                n_vals,
                mins,
                maxs,
                alpha=alpha,
                color=color,
                label=f"{label} range",
            )

            # Plot mean line
            ax.plot(
                n_vals,
                means,
                marker="o",
                color=color,
                linewidth=2,
                markersize=6,
                label=f"{label} mean",
            )

        # Shorten model name for title
        short_model = model.split("/")[-1] if "/" in model else model

        ax.set_xlabel("N (conversations per persona)", fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel("VERA Score", fontsize=10, color=TEXT_COLOR)
        ax.set_title(short_model, fontsize=11, fontweight="bold", color=TEXT_COLOR)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(0, 20)  # Zoomed in to see variation
        ax.set_facecolor("white")

    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.patch.set_facecolor(BG_COLOR)
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()

    print(f"📈 Score range plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VERA score variability across sampling strategies"
    )

    parser.add_argument(
        "--base-dir",
        "-d",
        default="evaluations/score_variability_20260109",
        help="Base directory containing evaluations",
    )

    parser.add_argument(
        "--n-values",
        "-n",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50, 100],
        help="N values: conversations per persona (default: 5 10 20 50 100)",
    )

    parser.add_argument(
        "--j-values",
        "-j",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="J values: judge iterations per conversation (default: 1 3 5)",
    )

    parser.add_argument(
        "--random-seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="score_variability_analysis.csv",
        help="Output CSV file path",
    )

    parser.add_argument(
        "--plot",
        "-p",
        default=None,
        help="Output plot file path (default: same as output with .png extension)",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the plot",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"❌ Error: Base directory not found: {base_dir}")
        return 1

    print(f"📂 Discovering evaluation files in: {base_dir}")
    model_data = discover_evaluation_files(base_dir)

    if not model_data:
        print("❌ No evaluation data found")
        return 1

    print(f"✅ Found {len(model_data)} model configurations")
    for model_key, personas in model_data.items():
        total = sum(len(files) for files in personas.values())
        print(f"   {model_key}: {len(personas)} personas, {total} files")

    print("\n🔬 Running sampling analysis...")
    print(f"   N values: {args.n_values}")
    print(f"   J values: {args.j_values}")
    print(f"   Random seed: {args.random_seed}")
    print("   (Number of samples determined by available conversations / N)")

    results_df = run_sampling_analysis(
        model_data,
        n_values=args.n_values,
        j_values=args.j_values,
        random_seed=args.random_seed,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✅ Results saved to: {output_path}")

    # Generate plots
    if not args.no_plot:
        plot_path = Path(args.plot) if args.plot else output_path.with_suffix(".png")

        # Per-model subplots (std dev)
        plot_std_vs_n(results_df, plot_path)

        # Combined plot (std dev)
        combined_path = plot_path.with_stem(plot_path.stem + "_combined")
        plot_std_vs_n_combined(results_df, combined_path)

        # Score range plot (min/max)
        range_path = plot_path.with_stem(plot_path.stem + "_ranges")
        plot_score_ranges(results_df, range_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
