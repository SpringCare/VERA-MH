#!/usr/bin/env bash
#
# run_recommended_vera_pipeline.sh
# --------------------------------
# Orchestrates the README-style “recommended” VERA flow in three phases:
#
#   1) Generate + judge + score with user model A (default: GPT 5.2) talking to
#      the provider agent you pass on the command line.
#   2) Same pipeline with user model B (default: Claude Opus 4.5).
#   3) Merge the two evaluation folders into one pooled bundle (results.csv
#      and scores/) using pool_vera_scores.py.
#
# Implementation detail: each phase runs run_pipeline.py; the evaluation
# directory path printed by the pipeline is captured from the log via
# pool_vera_scores.py --extract-from-log so the final pooling step knows which
# folders to merge.
#
# Usage:
#   ./scripts/run_recommended_vera_pipeline.sh <provider-agent-model> [extra run_pipeline.py args...]
#
# Example:
#   ./scripts/run_recommended_vera_pipeline.sh gpt-4o
#
# Optional environment (override defaults without editing this file):
#   VERA_OUTPUT_PARENT     Where new p_* run folders go (default: output)
#   VERA_USER_A          User agent for the first suite (default: gpt-5.2)
#   VERA_USER_B          User agent for the second suite (default: claude-opus-4-5-20251101)
#   VERA_JUDGE           Judge model (default: gpt-5.4)
#   VERA_MAX_CONCURRENT  Forwarded as --max-concurrent (default: 10)
#   VERA_MAX_PERSONAS    Forwarded as --max-personas (default: 100)
#   VERA_POOL_OUTPUT     Parent dir for pooled j_<judge>__p_* output (default: same as VERA_OUTPUT_PARENT)
#   VERA_POOL_SKIP_RISK  If set (non-empty), pooled run skips risk-level analysis (--skip-risk-analysis)

set -euo pipefail
# -e: exit on first failing command
# -u: treat unset variables as errors
# -o pipefail: pipeline fails if any stage fails (needed for PIPESTATUS below)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root: run_pipeline.py and output dirs are relative to project root.
cd "$SCRIPT_DIR/.."

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <provider-agent-model> [extra run_pipeline.py arguments...]" >&2
  exit 2
fi

# First positional: the “provider” side model (the agent being evaluated).
# Remaining args are forwarded verbatim to run_pipeline.py for both suites.
PROVIDER_AGENT="$1"
shift

OUTPUT_PARENT="${VERA_OUTPUT_PARENT:-output}"
USER_A="${VERA_USER_A:-gpt-5.2}"
USER_B="${VERA_USER_B:-claude-opus-4-5-20251101}"
JUDGE="${VERA_JUDGE:-gpt-5.4}"

POOL_PARENT="${VERA_POOL_OUTPUT:-$OUTPUT_PARENT}"

# Shared run_pipeline.py flags: provider, conversation shape, judge model,
# parent for new p_* runs (--conversation-output / -co → generate.py --output).
COMMON_ARGS=(
  --provider-agent "$PROVIDER_AGENT"
  --turns 30
  --runs 1
  --judge-model "$JUDGE"
  --conversation-output "$OUTPUT_PARENT"
)

# Throttling and persona cap (defaults here; override with VERA_* env vars).
COMMON_ARGS+=(--max-concurrent "${VERA_MAX_CONCURRENT:-10}" --max-personas "${VERA_MAX_PERSONAS:-100}")

# Arguments for the final pool_vera_scores.py invocation only.
POOL_ARGS=(-o "$POOL_PARENT")
if [[ -n "${VERA_POOL_SKIP_RISK:-}" ]]; then
  POOL_ARGS+=(--skip-risk-analysis)
fi

# Run one full pipeline (generate → judge → score), tee full log to stderr for
# visibility, then parse the log to recover the evaluation directory path.
# stdout of this function is only the path string (for command substitution).
run_pipeline_capture_eval() {
  local log
  log="$(mktemp)"
  # tee duplicates stream to stderr so the user sees progress; stdout would
  # otherwise pollute the captured eval path from command substitution.
  uv run python run_pipeline.py "$@" 2>&1 | tee "$log" >&2
  local st="${PIPESTATUS[0]}"
  if [[ "$st" -ne 0 ]]; then
    rm -f "$log"
    return "$st"
  fi
  local ev
  ev="$(uv run python "$SCRIPT_DIR/pool_vera_scores.py" --extract-from-log "$log")" || {
    rm -f "$log"
    return 1
  }
  rm -f "$log"
  printf '%s\n' "$ev"
}

echo "== VERA recommended pipeline: user $USER_A → provider $PROVIDER_AGENT =="
EVAL_A="$(run_pipeline_capture_eval --user-agent "$USER_A" "${COMMON_ARGS[@]}" "$@")"

echo ""
echo "== VERA recommended pipeline: user $USER_B → provider $PROVIDER_AGENT =="
EVAL_B="$(run_pipeline_capture_eval --user-agent "$USER_B" "${COMMON_ARGS[@]}" "$@")"

echo ""
echo "== Pooling evaluation scores into $POOL_PARENT =="
# Merge the two evaluation roots into one j_<judge>__p_* folder under POOL_PARENT.
uv run python "$SCRIPT_DIR/pool_vera_scores.py" "${POOL_ARGS[@]}" "$EVAL_A" "$EVAL_B"
