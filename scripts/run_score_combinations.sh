#!/bin/bash

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONV_BASE_DIR="conversations/scoring_2026_01_16_main_prompt"


# Parameters
RUNS=20
TURNS=20
MAX_CONCURRENT=10
MAX_WORDS=4000

users=(
    #"claude-opus-4-1-20250805"
    # "claude-sonnet-4-5-20250929"
    "gpt-5 -uep max_completion_tokens=5000"
    # "gpt-4o"
    "gemini-3-pro-preview"
    # "gemini-2.5-flash"
)

providers=(
    #"claude-opus-4-5-20251101"
    #"claude-sonnet-4-5-20250929"
    "gpt-5.2 -pep max_completion_tokens=5000"
    #"gpt-4o"
   # "gemini-3-pro-preview"
   # "gemini-2.5-flash"
)

for user in "${users[@]}"; do
    for provider in "${providers[@]}"; do
        # Extract model name (first word, before any flags)
        provider_model=$(echo "$provider" | awk '{print $1}')
        output_dir="$CONV_BASE_DIR/$provider_model"
        
        echo "Running with user: $user, provider: $provider"
        echo "Output dir: $output_dir"
        echo "python3 generate.py -f $output_dir -u $user -p $provider -r $RUNS -t $TURNS -c $MAX_CONCURRENT -w $MAX_WORDS"
        python3 generate.py -f "$output_dir" -u $user -p $provider -r $RUNS -t $TURNS -c $MAX_CONCURRENT -w $MAX_WORDS
    done
done

