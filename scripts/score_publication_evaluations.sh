#!/bin/bash
# Score all evaluation folders in publication_evaluations/vera_mh_v1_scores/
#
# This script finds all j_* evaluation folders and runs score.py on each one.
# It will generate scores.json and visualization PNGs for each evaluation.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Base directory for publication evaluations
EVAL_BASE_DIR="publication_evaluations/vera_mh_v1_scores"

if [ ! -d "$EVAL_BASE_DIR" ]; then
    echo "❌ Error: Directory not found: $EVAL_BASE_DIR"
    exit 1
fi

# Find all j_* evaluation folders
echo "🔍 Finding evaluation folders in $EVAL_BASE_DIR..."
eval_folders=$(find "$EVAL_BASE_DIR" -type d -name "j_*" | sort)

if [ -z "$eval_folders" ]; then
    echo "❌ No evaluation folders found matching pattern j_*"
    exit 1
fi

# Count total folders
total=$(echo "$eval_folders" | wc -l | tr -d ' ')
echo "📊 Found $total evaluation folders"
echo ""

# Process each folder
count=0
success=0
failed=0

while IFS= read -r eval_folder; do
    count=$((count + 1))
    results_csv="$eval_folder/results.csv"
    
    # Check if results.csv exists
    if [ ! -f "$results_csv" ]; then
        echo "[$count/$total] ⚠️  Skipping $eval_folder (no results.csv found)"
        continue
    fi
    
    echo "[$count/$total] 📈 Scoring: $eval_folder"
    
    # Run score.py
    if python3 -m judge.score -r "$results_csv" 2>&1; then
        success=$((success + 1))
        echo "   ✅ Success"
    else
        failed=$((failed + 1))
        echo "   ❌ Failed"
    fi
    echo ""
    
done <<< "$eval_folders"

# Summary
echo "=" | tr -d '\n' | head -c 80
echo ""
echo "📊 Summary:"
echo "   Total folders: $total"
echo "   ✅ Successful: $success"
echo "   ❌ Failed: $failed"
echo "   ⚠️  Skipped: $((total - success - failed))"

if [ $failed -gt 0 ]; then
    exit 1
fi
