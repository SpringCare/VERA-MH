#!/usr/bin/env bash
# Batch-run judge.score on every results.csv under a HEOR judging directory tree.
#
# Expected layout (example):
#   ROOT/
#     p_<conv_run>/
#       j_<judge>__p_<conv_run>/
#         results.csv
#         *.tsv
#
# Usage:
#   ./scripts/batch_score_heor_judging.sh
#   ./scripts/batch_score_heor_judging.sh "/path/to/GPT4o + S4.5 Judging"
#
# Optional env:
#   VERA_MH_ROOT  — repo root (default: parent of scripts/)
#   SKIP_ON_ERROR — if set to 1, continue after failures (default: stop on first error)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERA_ROOT="${VERA_MH_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

DEFAULT_ROOT="/Users/josh.gieringer/Desktop/HEOR AIM 3/GROK_3_EVAL"
JUDGING_ROOT="${1:-$DEFAULT_ROOT}"

if [[ ! -d "$JUDGING_ROOT" ]]; then
  echo "Error: directory not found: $JUDGING_ROOT" >&2
  exit 1
fi

cd "$VERA_ROOT" || exit 1

# macOS bash 3.2: no mapfile; use process substitution
tmp="$(mktemp)"
find "$JUDGING_ROOT" -name results.csv -type f | sort >"$tmp"
if [[ ! -s "$tmp" ]]; then
  rm -f "$tmp"
  echo "No results.csv found under: $JUDGING_ROOT" >&2
  exit 1
fi

total=$(wc -l <"$tmp" | tr -d ' ')
echo "VERA-MH root: $VERA_ROOT"
echo "Judging root: $JUDGING_ROOT"
echo "Found $total results.csv file(s)"
echo ""

failed=0
n=0
while IFS= read -r csv; do
  n=$((n + 1))
  echo "========================================"
  echo "[$n/$total] Scoring: $csv"
  echo "========================================"
  if python3 -m judge.score -r "$csv"; then
    echo ""
  else
    echo "FAILED: $csv" >&2
    failed=$((failed + 1))
    if [[ "${SKIP_ON_ERROR:-0}" != "1" ]]; then
      rm -f "$tmp"
      exit 1
    fi
  fi
done <"$tmp"
rm -f "$tmp"

if [[ $failed -gt 0 ]]; then
  echo "$failed run(s) failed." >&2
  exit 1
fi

echo "Done. Scored $total directory(ies)."
