#!/bin/bash

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Base directory
BASE_DIR="scoring_2026_01_16_main_prompt"
CONV_BASE_DIR="conversations/$BASE_DIR"
EVAL_BASE_DIR="evaluations/$BASE_DIR"

# Wait for 400 txt files in gpt-5.2 directory before starting
# WAIT_DIR="$CONV_BASE_DIR/gpt-5.2"
# REQUIRED_FILES=400
# CHECK_INTERVAL=30  # seconds between checks

# echo "Waiting for $REQUIRED_FILES txt files in $WAIT_DIR..."
# while true; do
#     current_count=$(find "$WAIT_DIR" -type f -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')
#     echo "$(date '+%Y-%m-%d %H:%M:%S') - Current file count: $current_count / $REQUIRED_FILES"
    
#     if [[ "$current_count" -ge "$REQUIRED_FILES" ]]; then
#         echo "Target reached! Starting judging..."
#         break
#     fi
    
#     sleep $CHECK_INTERVAL
# done

# Specific provider subdirectories to process
providers=(
    "claude-opus-4-5-20251101"
    # "gpt-5.2"
)

judges=(
    'gpt-4o:5'
    # 'claude-sonnet-4-5-20250929'
)

# Find all conversation run folders for specified providers
for provider in "${providers[@]}"; do
    for folder in "$CONV_BASE_DIR/$provider"/p_*; do
        if [[ -d "$folder" ]]; then
            output_dir="$EVAL_BASE_DIR/$provider"
            
            for judge in "${judges[@]}"; do
                echo "Running with folder: $folder, judge: $judge"
                echo "Output: $output_dir"
                echo "python3 judge.py -f $folder -j $judge -o $output_dir"
                python3 judge.py -f "$folder" -j "$judge" -o "$output_dir" -m 50
            done
        fi
    done
done
 