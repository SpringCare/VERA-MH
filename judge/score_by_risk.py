#!/usr/bin/env python3
"""
Score evaluation results by risk level from judge/runner.py output.

Run with `python -m judge.score_by_risk -r evaluations/[eval_folder]/results.csv`

Reads results.csv, re-calculates the dataframe from the tsv files in the same 
folder if the results.csv is empty, adds risk level information from personas.tsv,
and creates visualizations split by risk level.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from .utils import load_rubric_structure, extract_model_names_from_path
from .constants import (
    BEST_PRACTICE,
    NEUTRAL,
    DAMAGING,
    NOT_RELEVANT,
    BEST_PRACTICE_KEY,
    NEUTRAL_KEY,
    DAMAGING_KEY,
    NOT_RELEVANT_KEY,
    MUTED_RED,
    MUTED_YELLOW,
    MUTED_GREEN,
    MUTED_GRAY
)

DIMENSIONS, _ = load_rubric_structure("data/rubric.tsv")

# Risk level order for consistent sorting
RISK_LEVEL_ORDER = ['None', 'Low', 'High', 'Imminent']


def load_personas_risk_levels(personas_tsv_path: Path) -> Dict[str, str]:
    """
    Load persona names and their risk levels from personas.tsv.
    
    Args:
        personas_tsv_path: Path to personas.tsv file
        
    Returns:
        Dictionary mapping persona name to risk level, or empty dict if error
    """
    df = pd.read_csv(personas_tsv_path, sep='\t', keep_default_na=False)
    # Map persona name to risk level
    # Use keep_default_na=False to prevent pandas from converting "None" string to NaN
    risk_map = (
        df[['Name', 'Current Risk Level']]
        .set_index('Name')['Current Risk Level']
        .astype(str).str.strip()
        .to_dict()
    )
    return risk_map


def extract_persona_name_from_filename(filename: str) -> Optional[str]:
    """
    Extract persona name from TSV filename.
    
    Filename format: {hash}_{persona_name}_{model}_run{number}_iterative.tsv
    Example: 1a84d1_Brian_g4o_run3_iterative.tsv -> "Brian"
    
    Args:
        filename: TSV filename (with or without extension)
        
    Returns:
        Persona name or None if not found
    """
    # Format: {hash}_{persona}_{model}_run{number} or {hash}_{persona}_{model}_run{number}_iterative.tsv
    try:
        parts = filename.split('_')
        if len(parts) >= 2:
            # Return just the persona name (index 1), not the list
            return parts[1]
        return None
    except Exception as e:
        print(f"Error extracting persona name from filename {filename}: {e}")
        return None
    

def build_dataframe_from_tsv_files_with_risk(
    evaluations_dir: Path,
    personas_tsv_path: Path
) -> pd.DataFrame:
    """
    Build a dataframe from TSV evaluation files with risk level information.
    
    Args:
        evaluations_dir: Directory containing TSV evaluation files
        personas_tsv_path: Path to personas.tsv file
        
    Returns:
        DataFrame with columns: filename, run_id, persona_name, risk_level, and each dimension
    """
    results = []
    
    # Load risk level mapping
    risk_map = load_personas_risk_levels(personas_tsv_path)
    
    # Get run_id from directory name
    run_id = evaluations_dir.name.split('__')[-1] if '__' in evaluations_dir.name else evaluations_dir.name
    
    # Find all TSV files in the directory
    tsv_files = list(evaluations_dir.glob("*.tsv"))
    
    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in: {evaluations_dir}")
    
    for tsv_file in tsv_files:
        # Extract persona name from filename
        persona_name = extract_persona_name_from_filename(tsv_file.name)
        risk_level = risk_map.get(persona_name, 'Unknown') if persona_name else 'Unknown'
        
        # Extract filename from TSV file name
        tsv_stem = tsv_file.stem
        if tsv_stem.endswith('_iterative'):
            tsv_stem = tsv_stem[:-10]
        filename = f"{tsv_stem}.txt"
        
        # Read TSV file
        try:
            tsv_df = pd.read_csv(tsv_file, sep='\t')
            
            # Build row dictionary
            row = {
                "filename": filename,
                "run_id": run_id,
                "persona_name": persona_name or 'Unknown',
                "risk_level": risk_level
            }
            
            # Extract dimension -> score mapping
            for _, tsv_row in tsv_df.iterrows():
                dimension = str(tsv_row.get('Dimension', '')).strip()
                score = str(tsv_row.get('Score', '')).strip()
                
                if dimension in DIMENSIONS:
                    row[dimension] = score
            
            # Ensure all dimensions are present (fill with empty string if missing)
            for dimension in DIMENSIONS:
                if dimension not in row:
                    row[dimension] = ''
            
            results.append(row)
            
        except Exception as e:
            print(f"Warning: Error reading TSV file {tsv_file}: {e}")
            continue
    
    # Build dataframe with correct column order
    columns = ["filename", "run_id", "persona_name", "risk_level"] + DIMENSIONS
    df = pd.DataFrame(results, columns=columns)
    
    return df


def score_results_by_risk(
    results_csv_path: str,
    personas_tsv_path: str,
    output_json_path: str = None
) -> Dict[str, Any]:
    """
    Score evaluation results grouped by risk level.
    
    Args:
        results_csv_path: Path to results.csv file
        personas_tsv_path: Path to personas.tsv file
        output_json_path: Optional path to save JSON output
        
    Returns:
        Dictionary containing all scores grouped by risk level
    """
    # Always rebuild the dataframe from TSV files to ensure risk levels are correct
    print(f"📊 Rebuilding dataframe with risk levels from TSV files...")
    evaluations_dir = Path(results_csv_path).parent
    df = build_dataframe_from_tsv_files_with_risk(evaluations_dir, Path(personas_tsv_path))
    # Save the updated dataframe
    df.to_csv(results_csv_path, index=False)
    print(f"✅ Rebuilt dataframe with {len(df)} rows and saved to {results_csv_path}")
    
    # Calculate scores by risk level (including Unknown)
    risk_level_scores = {}
    
    for risk_level in RISK_LEVEL_ORDER:
        risk_df = df[df['risk_level'] == risk_level].copy()
        
        if len(risk_df) == 0:
            continue
        
        dimension_scores = {}
        
        for dimension in DIMENSIONS:
            if dimension not in risk_df.columns:
                continue
            
            # Include "Not Relevant" in the analysis
            dim_df = risk_df[risk_df[dimension].notna() & (risk_df[dimension] != '')].copy()
            
            if len(dim_df) == 0:
                continue
            
            total_count = len(dim_df)
            
            # Count each option (including Not Relevant)
            counts = {
                BEST_PRACTICE_KEY: len(dim_df[dim_df[dimension] == BEST_PRACTICE]),
                NEUTRAL_KEY: len(dim_df[dim_df[dimension] == NEUTRAL]),
                DAMAGING_KEY: len(dim_df[dim_df[dimension] == DAMAGING]),
                NOT_RELEVANT_KEY: len(dim_df[dim_df[dimension] == NOT_RELEVANT])
            }
            
            # Calculate percentages
            dimension_scores[dimension] = {
                'total_count': total_count,
                BEST_PRACTICE_KEY + '_pct': round(100.0 * counts[BEST_PRACTICE_KEY] / total_count, 2),
                NEUTRAL_KEY + '_pct': round(100.0 * counts[NEUTRAL_KEY] / total_count, 2),
                DAMAGING_KEY + '_pct': round(100.0 * counts[DAMAGING_KEY] / total_count, 2),
                NOT_RELEVANT_KEY + '_pct': round(100.0 * counts[NOT_RELEVANT_KEY] / total_count, 2),
                'counts': counts
            }
        
        risk_level_scores[risk_level] = {
            'total_conversations': len(risk_df),
            'dimensions': dimension_scores
        }
    
    # Extract model names
    model_names = extract_model_names_from_path(results_csv_path)
    
    # Build results dictionary
    results = {
        'judge_model': model_names['judge'],
        'persona_model': model_names['persona'],
        'agent_model': model_names['agent'],
        'risk_level_scores': risk_level_scores
    }
    
    # Save to JSON if path provided
    if output_json_path is None:
        csv_path = Path(results_csv_path)
        output_json_path = csv_path.parent / "scores_by_risk.json"
    
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def create_risk_level_visualizations(results: Dict[str, Any], output_path: Path):
    """
    Create visualizations split by risk level with all rating categories including Not Relevant.
    
    Args:
        results: Dictionary containing scores by risk level
        output_path: Path to save the visualization
    """
    risk_level_scores = results.get('risk_level_scores', {})
    
    if not risk_level_scores:
        print("⚠️  No risk level data to visualize")
        return
    
    # Colors are imported from utils
    
    # Get model names for title
    judge_model = results.get('judge_model', 'Unknown')
    persona_model = results.get('persona_model', 'Unknown')
    agent_model = results.get('agent_model', 'Unknown')
    title = f'Judge: {judge_model} | Persona: {persona_model} | Agent: {agent_model}'
    
    # Create figure with subplots for each dimension
    from matplotlib import gridspec
    n_dims = len(DIMENSIONS)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=(18, 6 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.3)
    
    for dim_idx, dimension in enumerate(DIMENSIONS):
        row = dim_idx // n_cols
        col = dim_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Prepare data for this dimension across all risk levels
        risk_levels = []
        best_practice_pcts = []
        neutral_pcts = []
        damaging_pcts = []
        not_relevant_pcts = []
        
        for risk_level in RISK_LEVEL_ORDER:
            if risk_level not in risk_level_scores:
                continue
            
            dim_scores = risk_level_scores[risk_level].get('dimensions', {}).get(dimension)
            if not dim_scores:
                continue
            
            risk_levels.append(risk_level)
            best_practice_pcts.append(dim_scores[BEST_PRACTICE_KEY + '_pct'])
            neutral_pcts.append(dim_scores[NEUTRAL_KEY + '_pct'])
            damaging_pcts.append(dim_scores[DAMAGING_KEY + '_pct'])
            not_relevant_pcts.append(dim_scores[NOT_RELEVANT_KEY + '_pct'])
        
        if not risk_levels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dimension, fontsize=12, fontweight='bold')
            continue
        
        # Create stacked bar chart
        x_pos = range(len(risk_levels))
        width = 0.7
        
        # Stack bars: Best Practice at bottom, then Neutral, then Damaging, then Not Relevant at top
        ax.bar(x_pos, best_practice_pcts, width, label=BEST_PRACTICE, color=MUTED_GREEN, bottom=0)
        ax.bar(x_pos, neutral_pcts, width, bottom=best_practice_pcts, 
               label=NEUTRAL, color=MUTED_YELLOW)
        ax.bar(x_pos, damaging_pcts, width,
               bottom=[bp + n for bp, n in zip(best_practice_pcts, neutral_pcts)],
               label=DAMAGING, color=MUTED_RED)
        ax.bar(x_pos, not_relevant_pcts, width,
               bottom=[bp + n + d for bp, n, d in zip(best_practice_pcts, neutral_pcts, damaging_pcts)],
               label=NOT_RELEVANT, color=MUTED_GRAY)
        
        # Format chart
        ax.set_xlabel('Persona Risk Level', fontsize=10, fontweight='bold')
        ax.set_ylabel('Proportion', fontsize=10, fontweight='bold')
        ax.set_title(dimension, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(risk_levels, fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend in the empty 6th subplot position (row 1, col 2)
    if n_dims < n_rows * n_cols:
        legend_row = n_rows - 1
        legend_col = n_cols - 1
        ax_legend = fig.add_subplot(gs[legend_row, legend_col])
        ax_legend.axis('off')  # Turn off axes
        
        # Create legend manually with the colors and labels
        from matplotlib.patches import Rectangle
        handles = [
            Rectangle((0, 0), 1, 1, facecolor=MUTED_GREEN, edgecolor='black'),
            Rectangle((0, 0), 1, 1, facecolor=MUTED_YELLOW, edgecolor='black'),
            Rectangle((0, 0), 1, 1, facecolor=MUTED_RED, edgecolor='black'),
            Rectangle((0, 0), 1, 1, facecolor=MUTED_GRAY, edgecolor='black')
        ]
        labels = [BEST_PRACTICE, NEUTRAL, DAMAGING, NOT_RELEVANT]
        ax_legend.legend(handles, labels, loc='center', fontsize=10, frameon=True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Risk level visualizations saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Score evaluation results by risk level from judge/runner.py output"
    )
    
    parser.add_argument(
        "--results-csv",
        "-r",
        required=True,
        help="Path to results.csv file from judge evaluation"
    )
    
    parser.add_argument(
        "--personas-tsv",
        "-p",
        default="data/personas.tsv",
        help="Path to personas.tsv file (default: data/personas.tsv)"
    )
    
    parser.add_argument(
        "--output-json",
        "-o",
        default=None,
        help="Path to save JSON output (default: scores_by_risk.json in same directory as CSV)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    results_csv_path = Path(args.results_csv)
    if not results_csv_path.exists():
        print(f"Error: Results CSV file not found: {args.results_csv}")
        return 1
    
    personas_tsv_path = Path(args.personas_tsv)
    if not personas_tsv_path.exists():
        print(f"Error: Personas TSV file not found: {args.personas_tsv}")
        return 1
    
    # Score the results (this will rebuild the CSV automatically)
    results = score_results_by_risk(
        str(results_csv_path),
        str(personas_tsv_path),
        args.output_json
    )
    
    # Create visualizations
    viz_path = Path(args.results_csv).parent / "scores_by_risk_visualization.png"
    try:
        create_risk_level_visualizations(results, viz_path)
    except Exception as e:
        print(f"⚠️  Warning: Could not create visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Print JSON path
    if args.output_json:
        json_path = args.output_json
    else:
        json_path = Path(args.results_csv).parent / "scores_by_risk.json"
    
    print(f"\n✅ Scores saved to: {json_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

