# Pre-commit Hooks

## Setup

```bash
uv sync              # Installs pre-commit
pre-commit install   # Activates hooks
```

## Hooks

### Standard: Ruff
Auto-formats and lints Python code. Configuration in `pyproject.toml`.

### Custom: CLAUDE.md → AGENTS.md Sync

**What:** Automatically keeps `AGENTS.md` identical to `CLAUDE.md`.

**Why:** Both files must contain the same project instructions - `CLAUDE.md` for Claude Code, `AGENTS.md` for custom agents.

**Behavior:**
- If `AGENTS.md` doesn't exist → creates it from `CLAUDE.md`
- If `AGENTS.md` exists and matches → passes
- If `AGENTS.md` exists and differs → **fails with error**

### Resolving Sync Conflicts

If the hook fails, choose one:

```bash
# Option 1: Delete AGENTS.md (simplest - auto-recreated)
rm AGENTS.md
git add AGENTS.md
git commit

# Option 2: Reconcile manually
diff CLAUDE.md AGENTS.md
cp CLAUDE.md AGENTS.md
git add AGENTS.md
git commit

# Option 3: Preserve current AGENTS.md temporarily
mv AGENTS.md AGENTS.md.backup
git add AGENTS.md AGENTS.md.backup
git commit
```

## Manual Usage

```bash
pre-commit run --all-files    # Run all hooks
```

## Configuration

- `.pre-commit-config.yaml` - Hook configuration
- `.pre-commit-scripts/sync-claude-to-agents.sh` - Custom sync script
