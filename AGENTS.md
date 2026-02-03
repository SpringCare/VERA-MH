# VERA-MH: Validation of Ethical and Responsible AI in Mental Health
Prototype for generating and evaluating LLM conversations in mental health contexts.

## Quick Start
```bash
# Install uv if not already installed
pip install uv

# Set up environment and install dependencies
uv sync
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Configure environment
cp .env.example .env  # Add your API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
```
**Python >= 3.11 required**

## Code Style
- Minimal print statements
- Prototype phase: prioritize clarity over perfection
- Don't overthink implementation
- Don't create example files
- Use `python3` command explicitly

## File Organization
- **Temporary tests**: `tmp_tests/` (not committed)
- **Main scripts**: `generate.py`, `judge.py` at root
- **Core modules**: Implementation in main directory
- **Docs**: See `docs/` for detailed guides

## Code Quality Tools
- **Formatting**: `uv run ruff format .`
- **Linting**: `uv run ruff check .`
- **Type checking**: `uv run pyright` (basic mode)
- **Pre-commit**: `pre-commit install` (auto-run checks on commit)
- All configuration in `pyproject.toml`
- **📖 See**: `docs/pre-commit-hooks.md` for pre-commit documentation

## Git Conventions

### Commit Message Format
Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <description>

[optional body]
```

**Types:**
- `feat`: New feature or significant enhancement
- `fix`: Bug fix
- `refactor`: Code restructuring without behavior change
- `test`: Adding or updating tests
- `docs`: Documentation changes only
- `chore`: Maintenance tasks (dependencies, config, tooling)
- `style`: Code style/formatting changes only
- `perf`: Performance improvements

**Guidelines:**
- Keep subject line under 72 characters
- Use imperative mood ("add feature" not "added feature")
- Don't end subject line with a period
- Separate subject from body with blank line
- Focus on *why* the change was made, not *what* changed
- Make atomic commits (one logical change per commit)

**Examples:**
```bash
feat: add support for GPT-4 model evaluation
fix: handle missing conversation files gracefully
docs: update README with new model options
chore: upgrade langchain to v0.1.0
test: add unit tests for judge scoring logic
```

### Branch Naming
Use descriptive branch names with type prefixes:

**Format:** `<type>/<brief-description>`

**Types:**
- `feat/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `test/` - Testing infrastructure
- `docs/` - Documentation updates
- `chore/` - Maintenance and tooling

**Examples:**
```bash
feat/add-gpt4-support
fix/conversation-file-handling
refactor/cleanup-judge-logic
test/unit-test-infrastructure
docs/update-api-examples
chore/upgrade-dependencies
```

**Guidelines:**
- Use kebab-case (lowercase with hyphens)
- Keep names concise but descriptive
- Avoid generic names like `fix/bug` or `feat/new-feature`
- Delete branches after merging

### Workflow
1. **Create branch from main**: `git checkout -b type/description`
2. **Make changes**: Follow code style and write tests
3. **Commit frequently**: Make atomic, logical commits
4. **Run quality checks**: Pre-commit hooks run automatically
5. **Push and create PR**: `git push -u origin branch-name`
6. **Use `/create-commits`**: Let Claude Code organize commits logically

**Tip:** Use `/create-commits` slash command to analyze changes and create well-organized, logical commits automatically.

## Testing
- No formal test suite yet (prototype phase)
- For temporary test scripts: use `tmp_tests/`
- When adding permanent tests: use `pytest` with `tests/` directory
- Run tests: `pytest` (when tests exist)
- Coverage: `pytest --cov` (when needed)

### Claude Code Testing Configuration
The project uses Claude Code with custom testing commands and agents:
- **Slash commands** (`.claude/commands/`) - User-facing testing workflows
- **test-engineer agent** (`.claude/agents/`) - Automated testing in parallel

**Maintenance guidelines:**
1. **When testing patterns change** (pytest config, fixtures, conventions):
   - Review and update relevant slash commands (`/test`, `/create-tests`, etc.)
   - Agent reads command files directly, so updates auto-propagate
   - Only update agent if commands are added/removed

2. **When adding new testing commands:**
   - Add to `.claude/commands/`
   - Update `.claude/commands/README.md` and main `README.md`
   - If it contains testing patterns, add reference to `.claude/agents/test-engineer.md`

**Why this matters:**
- Agents use slash commands as living documentation (via Read tool)
- Keeping them in sync ensures consistent testing patterns
- Single source of truth prevents duplication and drift

## Tech Stack
- **LLM Framework**: LangChain (multi-provider support)
- **Supported Providers**: Anthropic, OpenAI, Google GenAI
- **Data Validation**: Pydantic v2
- **Data Processing**: Pandas
- **Config Management**: python-dotenv

## Key Commands
```bash
# Generate conversations
python3 generate.py -u claude-3-7-sonnet -p claude-3-7-sonnet -t 6 -r 1

# Judge/evaluate conversations
python3 judge.py -f conversations/{YOUR_FOLDER} -j claude-3-7-sonnet

# Development
uv sync              # Install/update dependencies
uv add <package>     # Add new dependency
uv add --dev <pkg>   # Add dev dependency

# Code quality
uv run ruff format .   # Format code
uv run ruff check .    # Lint code
uv run pyright         # Type check
pre-commit run --all-files  # Run all pre-commit hooks

# Testing (when implemented)
pytest               # Run tests
pytest --cov         # Run with coverage
```

## Documentation Reference
- **Setup & Architecture**: See `README.md`
- **Pre-commit Hooks**: See `docs/pre-commit-hooks.md`
- **Custom LLM Providers**: See `docs/evaluating.md`
- **Usage Examples**: See `README.md` → "Usage" section
- **Model Configuration**: See `README.md` → "Models" section

## Docker
```bash
docker-compose up    # Run via Docker
```

---
For detailed information, see README.md and docs/
