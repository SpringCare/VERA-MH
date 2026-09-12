# VERA-MH: Agent Guide

Framework for generating and evaluating LLM conversations in mental health contexts. This file is for **any coding agent** (Cursor, Copilot, etc.). For Claude Code slash commands and `.claude/` maintenance, see [CLAUDE.md](./CLAUDE.md).

## Quick Start

```bash
pip install uv
uv sync
source .venv/bin/activate  # Windows: .venv\Scripts\activate
cp .env.example .env       # Add API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
```

**Python >= 3.11 required**

## Code Style

- Minimal print statements
- Prioritize clarity; match existing patterns in the module you touch
- **Check for existing code first** — before adding a function or helper, search the repo for something that already does the job; extend or reuse it when possible
- **Don't add abstractions unless asked** — avoid new base classes, wrappers, or indirection layers unless the task explicitly calls for them
- Keep changes **small and understandable** — one logical change per edit; avoid drive-by refactors or unrelated cleanup in the same diff
- When replacing behavior, **delete the old code** — don't leave dead paths, commented-out blocks, or "just in case" fallbacks behind
- Don't create example files unless asked
- Use `python3` or `uv run python` explicitly
- Add or update tests when changing behavior

## Architecture Map

| Area | Key paths | When to edit |
|------|-----------|--------------|
| **CLI** | `vera.py`, `vera_cli/`, `utils/config_schema.py` | Thin entry point, command adapters, and shared command/config wiring |
| **Generation** | `generate_conversations/` | Conversation simulation, turns, personas |
| **Judging** | `judge/` | Rubric scoring, TSV output, question navigation |
| **LLM providers** | `llm_clients/`, `llm_clients/llm_factory.py` | New models, custom HTTP/API providers |
| **Pipeline helpers** | `scripts/` | Pooling and automation until absorbed into `vera pool` |
| **Data** | `data/` (personas, rubrics) | Evaluation inputs (committed) |
| **Output** | `output/` (gitignored) | Generated transcripts, evaluations, logs |
| **Config** | `utils/model_config_loader.py`, `llm_clients/config.py` | Model name resolution, API keys |
| **Shared utils** | `utils/` | Naming, logging, conversation layout |

**Entry points:** `vera.py generate` is the first unified CLI feature. `generate.py`
remains temporarily as a compatibility adapter, while `judge.py` and
`run_pipeline.py` remain the current entry points for features not migrated yet.
Each legacy script is removed after its replacement feature is available; see
[docs/architecture.md](docs/architecture.md#migration-from-current-layout).

**Temporary experiments:** `tmp_tests/` (not committed). **Permanent tests:** `tests/`.

## Architecture compliance

Read [docs/architecture.md](docs/architecture.md) before structural changes. See its [ESCALATE section](docs/architecture.md#escalate-stop-and-ask) for when to stop and ask, and [Enforcement](docs/architecture.md#enforcement) for the pre-push verification commands.

When an architecture escalation requires a design record, create it under
[`docs/design/`](docs/design/) and follow the required format and lifecycle in
[`docs/design/README.md`](docs/design/README.md). Keep current normative rules in
the architecture documentation; design records preserve rationale and
consequences without duplicating that contract.

## Testing

The project uses [pytest](https://docs.pytest.org/) with unit and integration tests under `tests/`. Coverage is enforced (`--cov-fail-under=30` in `pyproject.toml`).

**Layout:**
- `tests/unit/` — fast, isolated tests
- `tests/integration/` — component interactions and CLI flows
- `tests/fixtures/` — rubrics, personas, sample conversations
- `tests/mocks/` — shared LLM mocks

The `e2e` marker exists in `pyproject.toml` but there is no `tests/e2e/` directory yet; use `integration` for workflow-level tests.

**Commands:**
```bash
# Default local/CI run (no API keys needed)
uv run pytest -m "not live"

# Full suite with coverage (default addopts include --cov)
uv run pytest

# Live tests only (requires API keys in .env)
uv run pytest -m live

# Single file or directory
uv run pytest tests/unit/judge/test_score.py
uv run pytest tests/integration/
```

**Markers:** `unit`, `integration`, `e2e`, `live` (see `pyproject.toml`). CI runs `pytest -m "not live"`; live tests run in a separate job when secrets are available.

**Scratch scripts:** use `tmp_tests/` for one-off experiments, not committed tests.

## Key Commands

Unified generation CLI:

```bash
uv run python vera.py generate \
  -c gpt-4o \
  -u claude-sonnet-4-5-20250929:1 \
  --target SI
```

Legacy commands for pipeline, judging, and compatibility:

```bash
uv run python run_pipeline.py \
  --user-agent claude-sonnet-4-5-20250929 \
  --provider-agent gpt-4o \
  --runs 1 \
  --turns 10 \
  --judge-model claude-sonnet-4-5-20250929 \
  --max-personas 5

uv run python generate.py -u claude-sonnet-4-5-20250929 -p gpt-4o -t 6 -r 1
uv run python judge.py -f output/{YOUR_P_RUN}/ -j claude-sonnet-4-5-20250929

# Recommended published-score profile (scripted; legacy)
./scripts/run_recommended_vera_pipeline.sh <provider-agent-model>

# Development
uv sync
uv add <package>
uv add --dev <pkg>

# Code quality
uv run ruff format .
uv run ruff check .
uv run pyright
pre-commit run --all-files
```

Use dated model IDs (e.g. `claude-sonnet-4-5-20250929`) as in README; shorthand aliases may not resolve.

## Code Quality Tools

- **Formatting:** `uv run ruff format .`
- **Linting:** `uv run ruff check .`
- **Type checking:** `uv run pyright` (basic mode)
- **Pre-commit:** `pre-commit install` — see [docs/pre-commit-hooks.md](docs/pre-commit-hooks.md)
- Configuration: `pyproject.toml`

## Git Conventions

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `style`, `perf`. Imperative mood, under 72 characters, no trailing period.

### Branch Naming

Format: `<type>/<brief-description>` (kebab-case), e.g. `feat/add-gpt4-support`, `fix/conversation-file-handling`.

### Workflow

1. Branch from `main`
2. Make changes; run `uv run pytest -m "not live"` for code changes
3. Atomic commits; pre-commit hooks run on commit
4. Push and open a PR

## Documentation Map

One canonical home per concern — cross-link, don't copy paragraphs.

| Doc | Audience | Use for |
|-----|----------|---------|
| [README.md](./README.md) | Humans | Setup, CLI usage, output layout |
| [docs/architecture.md](./docs/architecture.md) | Humans and agents | Target architecture, invariants, layer model |
| [docs/design/](./docs/design/) | Humans and agents | Historical design decisions, rationale, and compatibility consequences |
| **AGENTS.md** (this file) | All coding agents | Style, architecture map, testing, key commands, git conventions |
| [CLAUDE.md](./CLAUDE.md) | Claude Code only | Slash commands, `.claude/` maintenance |
| [docs/](./docs/) | Humans and agents | Topic deep dives (see links below) |

**When to update which file:** pytest/CI policy → AGENTS.md; new CLI flag or output layout → README (+ AGENTS key commands if agents run it often); LLM provider integration → [docs/evaluating.md](./docs/evaluating.md); Claude slash commands → `.claude/commands/` + CLAUDE.md + README command list.

**OpenSpec:** not used in this repo. Consider [OpenSpec](https://github.com/Fission-AI/OpenSpec) only for large multi-file features where you want agreed behavioral specs before coding (e.g. new judge dimensions, pipeline CLI changes). It complements — does not replace — AGENTS.md or README.

### Links

- **Architecture:** [docs/architecture.md](docs/architecture.md)
- **Design decision records:** [docs/design/README.md](docs/design/README.md)
- **Setup, pipeline, output layout:** [README.md](./README.md)
- **Custom LLM providers:** [docs/evaluating.md](./docs/evaluating.md)
- **Judge behavior:** [docs/judge.md](./docs/judge.md)
- **Structured output:** [docs/structured-output.md](./docs/structured-output.md)
- **Pre-commit hooks:** [docs/pre-commit-hooks.md](./docs/pre-commit-hooks.md)
- **Claude Code commands:** [CLAUDE.md](./CLAUDE.md), [.claude/commands/](./.claude/commands/)

## Docker

```bash
docker-compose up
```
