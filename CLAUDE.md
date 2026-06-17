# VERA-MH: Claude Code Guide

Framework for generating and evaluating LLM conversations in mental health contexts. This file is for **Claude Code** (slash commands, `.claude/` config). For agent-agnostic guidance (architecture, testing, domain guardrails, CLI commands, git conventions), see [AGENTS.md](./AGENTS.md).

## Slash Commands

Prefer these over retyping CLI commands from [AGENTS.md](./AGENTS.md):

| Area | Commands |
|------|----------|
| Setup | `/setup-dev` |
| Code quality | `/format` |
| VERA-MH | `/run-generator`, `/run-judge` |
| Testing | `/test`, `/fix-tests`, `/create-tests [module] [--layer=unit\|integration\|e2e]` |
| Git | `/create-commits`, `/create-pr` |

Full command docs: [.claude/commands/README.md](./.claude/commands/README.md)

## Git Workflow (Claude Code)

Use `/create-commits` to organize commits logically, then `/create-pr` for the pull request. All other git conventions are in [AGENTS.md](./AGENTS.md).

## Testing

See [AGENTS.md](./AGENTS.md) for full testing policy. Summary:

- `tests/unit/` and `tests/integration/`; fixtures in `tests/fixtures/`
- Default: `uv run pytest -m "not live"` (CI-safe, no API keys)
- Live API tests: `uv run pytest -m live`
- Coverage enforced via `pyproject.toml` (`--cov-fail-under=30`)

### Claude Code Testing Configuration

- **Slash commands** (`.claude/commands/`) — `/test`, `/fix-tests`, `/create-tests`
- **test-engineer agent** (`.claude/agents/`) — parallel test runs

**Maintenance guidelines:**

1. **When testing patterns change** (pytest config, fixtures, conventions):
   - Update relevant slash commands (`/test`, `/create-tests`, etc.)
   - Update [AGENTS.md](./AGENTS.md) if agent-facing policy changes
   - Only update `test-engineer` agent if commands are added/removed

2. **When adding new testing commands:**
   - Add to `.claude/commands/`
   - Update `.claude/commands/README.md` and `README.md`
   - Reference in `.claude/agents/test-engineer.md` if applicable

## Documentation Reference

See the **Documentation map** in [AGENTS.md](./AGENTS.md). Claude-specific entry points:

- **Slash commands:** [.claude/commands/README.md](./.claude/commands/README.md)
- **Team settings:** [`.claude/settings.json`](./.claude/settings.json) (shared); `.claude/settings.local.json` (personal, not committed)
