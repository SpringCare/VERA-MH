# Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) hooks to automatically enforce code quality standards and maintain file synchronization.

## Installation

Pre-commit hooks are automatically installed when you set up the project:

```bash
uv sync              # Installs pre-commit as a dev dependency
pre-commit install   # Installs the git hooks
```

## Configured Hooks

### 1. Ruff (Code Formatting & Linting)

Automatically formats and lints Python code before each commit.

**What it does:**
- **`ruff`**: Lints code and auto-fixes issues (unused imports, etc.)
- **`ruff-format`**: Formats code to match project style (88 char line length, double quotes)

**Configuration:** See `[tool.ruff]` in `pyproject.toml`

**Manual usage:**
```bash
ruff check --fix .   # Lint and auto-fix
ruff format .        # Format code
```

### 2. CLAUDE.md → AGENTS.md Sync

**Purpose:** Automatically keeps `AGENTS.md` synchronized with `CLAUDE.md`.

This hook ensures that any changes to `CLAUDE.md` are reflected in `AGENTS.md`. The files must remain identical.

**Behavior:**

| Scenario | Action |
|----------|--------|
| `AGENTS.md` doesn't exist | ✅ Automatically created from `CLAUDE.md` |
| `AGENTS.md` exists and is identical | ✅ Hook passes (no changes needed) |
| `AGENTS.md` exists and differs | ❌ **Commit blocked with error** |

**Why?**
- `CLAUDE.md` contains project instructions for Claude Code
- `AGENTS.md` is used by custom agents and must stay in sync
- Manual synchronization is error-prone

### Resolving AGENTS.md Sync Conflicts

If you modify `CLAUDE.md` and the hook fails, you'll see this error:

```
❌ ERROR: AGENTS.md must be identical to CLAUDE.md
```

**Resolution steps:**

#### Option 1: Delete AGENTS.md (simplest)

The hook will recreate it automatically:

```bash
rm AGENTS.md
git add AGENTS.md
git commit
```

#### Option 2: Review and reconcile differences

If you have intentional changes in `AGENTS.md`:

```bash
# See what's different
diff CLAUDE.md AGENTS.md

# Manually reconcile the differences
# Edit one or both files as needed

# Copy CLAUDE.md to AGENTS.md when ready
cp CLAUDE.md AGENTS.md
git add AGENTS.md
git commit
```

#### Option 3: Preserve AGENTS.md temporarily

If you need to keep the current `AGENTS.md` for reference:

```bash
# Move it out of the way
mv AGENTS.md AGENTS.md.backup
git add AGENTS.md AGENTS.md.backup

# Commit (AGENTS.md will be recreated from CLAUDE.md)
git commit

# Later, reconcile your changes:
diff CLAUDE.md AGENTS.md.backup
# Apply any needed changes to CLAUDE.md
cp CLAUDE.md AGENTS.md
git add AGENTS.md
git commit -m "Reconcile AGENTS.md changes"
```

## Bypassing Hooks (Not Recommended)

In rare cases, you may need to bypass hooks:

```bash
git commit --no-verify
```

**⚠️ Warning:** Only use `--no-verify` when absolutely necessary (e.g., during initial tooling setup). Regular commits should pass all hooks.

## Running Hooks Manually

You can run all hooks on all files without committing:

```bash
pre-commit run --all-files
```

Or run a specific hook:

```bash
pre-commit run ruff --all-files
pre-commit run sync-claude-to-agents --all-files
```

## Updating Hooks

To update hook versions:

```bash
pre-commit autoupdate
```

## Troubleshooting

### Hook fails but files look correct

Try reinstalling the hooks:

```bash
pre-commit uninstall
pre-commit install
pre-commit run --all-files
```

### Permission denied on script

Ensure the sync script is executable:

```bash
chmod +x .pre-commit-scripts/sync-claude-to-agents.sh
```

### Hook not running

Check that pre-commit is installed:

```bash
pre-commit --version
git config --list | grep hook
```

## Configuration Files

- **`.pre-commit-config.yaml`**: Pre-commit hook configuration
- **`.pre-commit-scripts/sync-claude-to-agents.sh`**: CLAUDE.md sync script
- **`pyproject.toml`**: Ruff configuration (`[tool.ruff]` section)

## Best Practices

1. **Don't bypass hooks** unless absolutely necessary
2. **Run hooks locally** before pushing: `pre-commit run --all-files`
3. **Keep CLAUDE.md and AGENTS.md in sync** - never edit AGENTS.md directly
4. **Update hooks regularly**: `pre-commit autoupdate`
5. **Test hooks** after modifying `.pre-commit-config.yaml`

## See Also

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [CLAUDE.md](../CLAUDE.md) - Project instructions for Claude Code
