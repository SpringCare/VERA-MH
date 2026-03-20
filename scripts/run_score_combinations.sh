#!/bin/bash

# Parse --dry-run or -n
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        -n|--dry-run) DRY_RUN=true ;;
    esac
done

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONV_BASE_DIR="conversations/HEOR_GROK_3"


# Parameters
RUNS_TURNS_PAIRS=(
    "1 30"
    # "1 100"
)
MAX_CONCURRENT=20
MAX_PERSONAS=100


users=(
    "gpt-5.2 -uep max_completion_tokens=5000"
    "claude-opus-4-5-20251101"
    # "claude-opus-4-6"
    # "claude-sonnet-4-5-20250929"
)

providers=(
    # "gpt-4o"
    # "claude-sonnet-4-6"
    # "gemini-2.5-flash"
    # "gpt-5.2 -pep max_completion_tokens=5000"
    # "claude-opus-4-5-20251101"
    # "claude-opus-4-6"
    # "gemini-3.1-pro-preview"
    "azure-grok-3"
    # "azure-grok-4"
    # "claude-sonnet-4-5-20250929"
)

if $DRY_RUN; then
    echo "=== DRY RUN: configuration ==="
    echo "CONV_BASE_DIR=$CONV_BASE_DIR"
    echo "MAX_CONCURRENT=$MAX_CONCURRENT"
    echo "MAX_PERSONAS=$MAX_PERSONAS"
    echo "RUNS_TURNS_PAIRS=(${RUNS_TURNS_PAIRS[*]})"
    echo "users=(${users[*]})"
    echo "providers=(${providers[*]})"
    echo ""
    echo "=== DRY RUN: commands that would run ==="
fi

for provider in "${providers[@]}"; do
    for runs_turns in "${RUNS_TURNS_PAIRS[@]}"; do
        read -r RUNS TURNS <<< "$runs_turns"
        for user in "${users[@]}"; do
            # Extract model name (first word, before any flags)
            provider_model=$(echo "$provider" | awk '{print $1}')
            output_dir="$CONV_BASE_DIR/$provider_model"

            echo "Running with:"
            echo "provider: $provider"
            echo "user: $user"
            echo "max personas: $MAX_PERSONAS"
            echo "conversations: $RUNS"
            echo "max turns per conversation: $TURNS"
            echo "output dir: $output_dir"
            echo "python3 generate.py -f $output_dir -u $user -p $provider -r $RUNS -t $TURNS -c $MAX_CONCURRENT -mp $MAX_PERSONAS"
            if ! $DRY_RUN; then
                python3 generate.py -f "$output_dir" -u $user -p $provider -r $RUNS -t $TURNS -c $MAX_CONCURRENT -mp $MAX_PERSONAS
            fi
            echo ""
        done
    done
done

