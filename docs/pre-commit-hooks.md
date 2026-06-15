# Pre-commit Hooks

## Setup

```bash
uv sync              # Installs pre-commit
pre-commit install   # Activates hooks
```

## Hooks

### Standard: Ruff

Auto-formats and lints Python code. Configuration in `pyproject.toml`.

## Agent Documentation

`AGENTS.md` and `CLAUDE.md` are **intentionally separate**:

- **[AGENTS.md](../AGENTS.md)** — agent-agnostic guide (architecture, testing, domain guardrails, key commands)
- **[CLAUDE.md](../CLAUDE.md)** — Claude Code slash commands and `.claude/` maintenance

Update the file that matches your audience. There is no pre-commit sync between them.

## Manual Usage

```bash
pre-commit run --all-files    # Run all hooks
```

## Configuration

- `.pre-commit-config.yaml` — Hook configuration
