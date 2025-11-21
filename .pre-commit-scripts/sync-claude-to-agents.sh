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
    echo "❌ ERROR: $AGENTS_FILE exists and differs from $CLAUDE_FILE"
    echo ""
    echo "You are attempting to modify $CLAUDE_FILE, but $AGENTS_FILE already exists"
    echo "with different content."
    echo ""
    echo "To resolve this issue, choose one of the following options:"
    echo ""
    echo "  1. If $AGENTS_FILE should be overwritten:"
    echo "     rm $AGENTS_FILE"
    echo "     git add $AGENTS_FILE"
    echo ""
    echo "  2. If you want to manually sync the files:"
    echo "     # Review differences:"
    echo "     diff $CLAUDE_FILE $AGENTS_FILE"
    echo "     # Then manually edit $AGENTS_FILE"
    echo ""
    echo "  3. If $CLAUDE_FILE changes should not affect $AGENTS_FILE:"
    echo "     # Keep both files separate (not recommended)"
    echo ""
    exit 1
fi

# Files are identical, no action needed
exit 0
