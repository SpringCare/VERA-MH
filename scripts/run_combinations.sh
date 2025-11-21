#!/bin/bash

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parameters
RUNS=1
TURNS=20
MAX_WORDS=4000

users=(
    "claude_opus_4_1_20250805"
    "gpt-5 -uep max_completion_tokens=5000"
    "gemini-3-pro-preview"
)

providers=(
    "gpt-4o"
    #"gpt-5 -pep max_completion_tokens=5000"
    #"claude_opus_4_1_20250805"
    #"gemini-3-pro-preview"
)

for user in "${users[@]}"; do
    for provider in "${providers[@]}"; do
        echo "Running with user: $user, provider: $provider"
        echo "python3 generate.py -u $user -p $provider -r $RUNS -t $TURNS -w $MAX_WORDS"
        python3 generate.py -u $user -p $provider -r $RUNS -t $TURNS -w $MAX_WORDS 
    done
done

wait
