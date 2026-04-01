#!/bin/bash

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-n|--dry-run]"
            echo "  -n, --dry-run  Print commands without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-n|--dry-run]"
            exit 1
            ;;
    esac
done

[[ $DRY_RUN -eq 1 ]] && echo "[DRY RUN] Commands will not be executed."

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Base directory to search for conversation folders
# CONV_BASE_DIR="conversations/score_variability_20260120"
CONV_BASE_DIR="conversations/HEOR_CLAUDE_4"
EVAL_BASE_DIR="evaluations/HEOR_CLAUDE_4_EVAL"

MAX_CONCURRENT=20

judges=(
    'gpt-4o'
    'claude-sonnet-4-5-20250929'
)

# Find all conversation run folders
# Pattern 1: nested (provider/user/run_folder) e.g. .../HEOR/gpt-4o/gemini-3-pro-preview/p_*
# Pattern 2: flat (run_folder directly under base) e.g. .../HEOR/Set_01-...
FOUND=0
for folder in "$CONV_BASE_DIR"/*/p_*; do
    if [[ -d "$folder" ]]; then
        relative_path="${folder#$CONV_BASE_DIR/}"
        provider_user="${relative_path%/p_*}"
        output_dir="$EVAL_BASE_DIR/$provider_user"
        FOUND=1
        for judge in "${judges[@]}"; do
            echo "Running with folder: $folder, judge: $judge"
            echo "Output: $output_dir"
            echo "python3 judge.py -f $folder -j $judge -o $output_dir -m $MAX_CONCURRENT"
            if [[ $DRY_RUN -eq 0 ]]; then
                python3 judge.py -f "$folder" -j "$judge" -o "$output_dir" -m $MAX_CONCURRENT
            fi
        done
    fi
done
if [[ $FOUND -eq 0 ]]; then
    # Flat layout: run folders directly under CONV_BASE_DIR
    for folder in "$CONV_BASE_DIR"/*; do
        if [[ -d "$folder" ]]; then
            provider_user="$(basename "$folder")"
            output_dir="$EVAL_BASE_DIR/$provider_user"
            FOUND=1
            for judge in "${judges[@]}"; do
                echo "Running with folder: $folder, judge: $judge"
                echo "Output: $output_dir"
                echo "python3 judge.py -f $folder -j $judge -o $output_dir -m $MAX_CONCURRENT"
                if [[ $DRY_RUN -eq 0 ]]; then
                    python3 judge.py -f "$folder" -j "$judge" -o "$output_dir" -m $MAX_CONCURRENT
                fi
            done
        fi
    done
fi
if [[ $FOUND -eq 0 ]]; then
    echo "No conversation folders found under $CONV_BASE_DIR"
    exit 1
fi
