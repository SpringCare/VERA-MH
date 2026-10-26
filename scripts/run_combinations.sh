#!/bin/bash
set -uo pipefail

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parameters (override from the environment, e.g. TURNS=5 ./scripts/run_combinations.sh)
RUNS="${RUNS:-1}"
TURNS="${TURNS:-30}"
TARGET="${TARGET:-HFO}"
DRY_RUN="${DRY_RUN:-0}"

users=(
    gpt-5.2
)

providers=(
    claude-sonnet-5
    gemini-3.6-flash
)

failed=()

for user in "${users[@]}"; do
    for provider in "${providers[@]}"; do
        cmd=(
            uv run vera generate
            --user "$user:$RUNS"
            --chatbot "$provider"
            --turns "$TURNS"
            --target "$TARGET"
        )

        echo "==> user=$user provider=$provider"
        printf '    '; printf '%q ' "${cmd[@]}"; echo

        [[ "$DRY_RUN" == 1 ]] && continue

        if ! "${cmd[@]}"; then
            echo "!!! FAILED: user=$user provider=$provider" >&2
            failed+=("$user/$provider")
        fi
    done
done

if ((${#failed[@]})); then
    echo "Failed combinations: ${failed[*]}" >&2
    exit 1
fi
echo "All ${#users[@]}x${#providers[@]} combinations completed."

