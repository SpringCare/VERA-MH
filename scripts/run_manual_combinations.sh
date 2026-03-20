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

CONV_BASE_DIR="conversations/HEOR_CATCH_UP_2_NAMES"
MAX_CONCURRENT=1
MAX_PERSONAS=100

# -----------------------------------------------------------------------------
# Model aliases (include -uep/-pep flags as needed)
# -----------------------------------------------------------------------------
user_gpt5_2="gpt-5.2 -uep max_completion_tokens=5000"
provider_gpt5_2="gpt-5.2 -pep max_completion_tokens=5000"
opus4_5="claude-opus-4-5-20251101"
opus4_6="claude-opus-4-6"
grok4="azure-grok-4"

# -----------------------------------------------------------------------------
# Combinations: "provider|user|runs|turns|personas"
# Use | as delimiter so model args with spaces work. Personas: comma-delimited
# names (e.g. Mateo,Omar) or empty for all.
# -----------------------------------------------------------------------------
provider_user_runs_turns_personas_sets=(
    # "$opus4_6|$opus4_5|4|20|Mateo"
    # "$grok4|$user_gpt5_2|2|20|William,Luna"
    # "$grok4|$opus4_5|2|20|Bella"
    # "$grok4|$opus4_5|1|100|Emily,Hazel,Grayson,Hudson,Ellie,Leo,Daniel,Ella,Jackson,John,Leah,Ethan,Ezra,Chloe,James,Grace,Elijah,Henry,Asher,David,Isaac"
    "$grok4|$opus4_5|1|100|Asher"
)

# -----------------------------------------------------------------------------

if $DRY_RUN; then
    echo "=== DRY RUN: configuration ==="
    echo "CONV_BASE_DIR=$CONV_BASE_DIR"
    echo "MAX_CONCURRENT=$MAX_CONCURRENT"
    echo "MAX_PERSONAS=$MAX_PERSONAS"
    echo ""
    echo "provider_user_runs_turns_personas_sets:"
    for i in "${!provider_user_runs_turns_personas_sets[@]}"; do
        echo "  [$i] ${provider_user_runs_turns_personas_sets[$i]}"
    done
    echo ""
    echo "=== DRY RUN: commands that would run ==="
fi

for combo in "${provider_user_runs_turns_personas_sets[@]}"; do
    IFS='|' read -r provider user runs turns personas <<< "$combo"
    # Extract model name (first word) for output directory
    provider_model=$(echo "$provider" | awk '{print $1}')
    output_dir="$CONV_BASE_DIR/$provider_model"

    persona_args=""
    if [[ -n "${personas:-}" ]]; then
        persona_args="-pn $personas"
    fi
    echo "Running with:"
    echo "  provider: $provider"
    echo "  user: $user"
    echo "  max personas: $MAX_PERSONAS"
    [[ -n "${personas:-}" ]] && echo "  persona names: $personas"
    echo "  conversations: $runs"
    echo "  max turns per conversation: $turns"
    echo "  output dir: $output_dir"
    echo "  python3 generate.py -f $output_dir -u $user -p $provider -r $runs -t $turns -c $MAX_CONCURRENT -mp $MAX_PERSONAS $persona_args"
    if ! $DRY_RUN; then
        python3 generate.py -f "$output_dir" -u $user -p $provider -r "$runs" -t "$turns" -c $MAX_CONCURRENT -mp $MAX_PERSONAS $persona_args
    fi
    echo ""
done
