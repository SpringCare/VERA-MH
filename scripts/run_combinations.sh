#!/bin/bash

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parameters
RUNS=5
TURNS=20
MAX_WORDS=4000

users=(
    "claude-opus-4-5-20251101"
    "claude-sonnet-4-5-20250929"
    "chatgpt-4o-latest"
    "gpt-5.2-2025-12-11 -uep max_completion_tokens=5000"
    "gemini-3-pro-preview"
    "gemini-2.5-flash"
)

providers=(
    "claude-opus-4-5-20251101"
    "claude-sonnet-4-5-20250929"
    "chatgpt-4o-latest"
    "gpt-5.2-2025-12-11 -pep max_completion_tokens=5000"
    "claude-opus-4-5-20251101"
    "gemini-3-pro-preview"
    "gemini-2.5-flash"
)

for user in "${users[@]}"; do
    for provider in "${providers[@]}"; do
        echo "Running with user: $user, provider: $provider"
        echo "python3 generate.py -u $user -p $provider -r $RUNS -t $TURNS -m"
        python3 generate.py -u $user -p $provider -r $RUNS -t $TURNS -m
    done
done

wait
