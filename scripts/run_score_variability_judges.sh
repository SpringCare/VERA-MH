#!/bin/bash

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Base directory to search for conversation folders
CONV_BASE_DIR="conversations/score_variability_20260109"
EVAL_BASE_DIR="evaluations/score_variability_20260109"

judges=(
    'gpt-4o:5'
    # 'claude-sonnet-4-5-20250929'
)

# Find all conversation run folders (3 levels deep: provider/user/run_folder)
for folder in "$CONV_BASE_DIR"/*/*/p_*; do
    if [[ -d "$folder" ]]; then
        # Extract provider/user from path (e.g., gpt-4o/gemini-3-pro-preview)
        relative_path="${folder#$CONV_BASE_DIR/}"  # Remove base dir prefix
        provider_user="${relative_path%/p_*}"       # Remove run folder suffix
        
        output_dir="$EVAL_BASE_DIR/$provider_user"
        
        for judge in "${judges[@]}"; do
            echo "Running with folder: $folder, judge: $judge"
            echo "Output: $output_dir"
            echo "python3 judge.py -f $folder -j $judge -o $output_dir"
            python3 judge.py -f "$folder" -j "$judge" -o "$output_dir" -m 50
        done
    fi
done
