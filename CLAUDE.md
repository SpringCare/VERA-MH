# VERA-MH: Claude Code Guide

Framework for generating and evaluating LLM conversations in mental health contexts. This file is for **Claude Code** (slash commands, `.claude/` config). For agent-agnostic guidance (architecture, testing, domain guardrails), see [AGENTS.md](./AGENTS.md).

## Quick Start

```bash
pip install uv
uv sync
source .venv/bin/activate  # Windows: .venv\Scripts\activate
cp .env.example .env       # Add API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
```

**Python >= 3.11 required**

## Code Style

- Minimal print statements
- Prioritize clarity; match existing patterns in the module you touch
- Don't create example files unless asked
- Use `python3` or `uv run python` explicitly
- Add or update tests when changing behavior

## File Organization

- **Temporary tests**: `tmp_tests/` (not committed)
- **Main scripts**: `generate.py`, `judge.py`, `run_pipeline.py` at root
- **Packages**: `generate_conversations/`, `judge/`, `llm_clients/`, `utils/`
- **Permanent tests**: `tests/` (unit and integration)
- **Docs**: `docs/`; agent architecture map in [AGENTS.md](./AGENTS.md)

## Code Quality Tools

- **Formatting**: `uv run ruff format .`
- **Linting**: `uv run ruff check .`
- **Type checking**: `uv run pyright` (basic mode)
- **Pre-commit**: `pre-commit install` (auto-run checks on commit)
- All configuration in `pyproject.toml`
- See `docs/pre-commit-hooks.md` for pre-commit documentation

## Git Conventions

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <description>

[optional body]
```

**Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `style`, `perf`

**Guidelines:** Imperative mood, under 72 characters, no trailing period, atomic commits.

### Branch Naming

**Format:** `<type>/<brief-description>` (kebab-case), e.g. `feat/add-gpt4-support`

### Workflow

1. **Create branch from main**: `git checkout -b type/description`
2. **Make changes**: Follow code style; run `uv run pytest -m "not live"`
3. **Commit frequently**: Atomic, logical commits
4. **Run quality checks**: Pre-commit hooks run automatically
5. **Push and create PR**: `git push -u origin branch-name`
6. **Use `/create-commits`**: Let Claude Code organize commits logically

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

## Key Commands

```bash
# Slash-command alternatives: /run-generator, /run-judge, /test, /format

# End-to-end pipeline
uv run python run_pipeline.py \
  --user-agent claude-sonnet-4-5-20250929 \
  --provider-agent gpt-4o \
  --runs 1 --turns 10 \
  --judge-model claude-sonnet-4-5-20250929 \
  --max-personas 5

# Generate / judge (step by step)
uv run python generate.py -u claude-sonnet-4-5-20250929 -p gpt-4o -t 6 -r 1
uv run python judge.py -f output/{YOUR_P_RUN}/ -j claude-sonnet-4-5-20250929

# Development & quality
uv sync
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest -m "not live"
pre-commit run --all-files
```

## Documentation Reference

- **Agent guide (architecture, guardrails, testing):** [AGENTS.md](./AGENTS.md)
- **Setup & usage:** [README.md](./README.md)
- **Custom LLM providers:** [docs/evaluating.md](./docs/evaluating.md)
- **Slash commands:** [.claude/commands/README.md](./.claude/commands/README.md)

## Docker

```bash
docker-compose up
```
