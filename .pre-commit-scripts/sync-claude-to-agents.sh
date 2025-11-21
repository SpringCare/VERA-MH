#!/bin/bash
# Pre-commit hook to sync CLAUDE.md to AGENTS.md
# Fails if AGENTS.md exists and differs from CLAUDE.md

set -e

CLAUDE_FILE="CLAUDE.md"
AGENTS_FILE="AGENTS.md"

# Check if CLAUDE.md exists
if [ ! -f "$CLAUDE_FILE" ]; then
    echo "Error: $CLAUDE_FILE not found"
    exit 1
fi

# If AGENTS.md doesn't exist, create it
if [ ! -f "$AGENTS_FILE" ]; then
    echo "Creating $AGENTS_FILE from $CLAUDE_FILE"
    cp "$CLAUDE_FILE" "$AGENTS_FILE"
    git add "$AGENTS_FILE"
    exit 0
fi

# If AGENTS.md exists, check if it differs from CLAUDE.md
if ! diff -q "$CLAUDE_FILE" "$AGENTS_FILE" > /dev/null 2>&1; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "❌ ERROR: AGENTS.md must be identical to CLAUDE.md"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "PROBLEM:"
    echo "  You modified CLAUDE.md, but AGENTS.md already exists with different content."
    echo "  These files MUST remain synchronized."
    echo ""
    echo "REQUIRED ACTION:"
    echo "  AGENTS.md must either:"
    echo "  • Be deleted (so it can be auto-created from CLAUDE.md), OR"
    echo "  • Be manually updated to match CLAUDE.md exactly"
    echo ""
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "RESOLUTION OPTIONS:"
    echo "────────────────────────────────────────────────────────────────────────────"
    echo ""
    echo "Option 1: Delete AGENTS.md (it will be recreated automatically)"
    echo "  $ rm AGENTS.md"
    echo "  $ git add AGENTS.md"
    echo "  $ git commit"
    echo ""
    echo "Option 2: Review differences and reconcile manually"
    echo "  $ diff CLAUDE.md AGENTS.md          # See what's different"
    echo "  # Manually edit AGENTS.md to match CLAUDE.md (or vice versa)"
    echo "  # Then copy CLAUDE.md to AGENTS.md:"
    echo "  $ cp CLAUDE.md AGENTS.md"
    echo "  $ git add AGENTS.md"
    echo "  $ git commit"
    echo ""
    echo "Option 3: Move/rename AGENTS.md if you need to preserve it"
    echo "  $ mv AGENTS.md AGENTS.md.backup"
    echo "  $ git add AGENTS.md AGENTS.md.backup"
    echo "  # Reconcile changes later, then:"
    echo "  $ cp CLAUDE.md AGENTS.md"
    echo "  $ git add AGENTS.md"
    echo "  $ git commit"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    exit 1
fi

# Files are identical, no action needed
exit 0
