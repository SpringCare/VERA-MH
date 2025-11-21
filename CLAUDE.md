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
- **Type checking**: `pyright` (basic mode, not overly strict during prototype)
- **Formatting**: `ruff format .` (auto-format code)
- **Linting**: `ruff check .` (lint with auto-fix: `ruff check --fix .`)
- **Pre-commit**: `pre-commit install` (auto-run checks on commit)
- All configuration in `pyproject.toml`

## Testing
- No formal test suite yet (prototype phase)
- For temporary test scripts: use `tmp_tests/`
- When adding permanent tests: use `pytest` with `tests/` directory
- Run tests: `pytest` (when tests exist)
- Coverage: `pytest --cov` (when needed)

## Tech Stack
- **LLM Framework**: LangChain (multi-provider support)
- **Supported Providers**: Anthropic, OpenAI, Google GenAI
- **Data Validation**: Pydantic v2
- **Data Processing**: Pandas
- **Config Management**: python-dotenv

## Key Commands
```bash
# Generate conversations
python3 generate.py --model claude-3-7-sonnet --num-conversations 5

# Judge/evaluate conversations
python3 judge.py --input-dir output/conversations --judge-model claude-3-7-sonnet

# Development
uv sync              # Install/update dependencies
uv add <package>     # Add new dependency
uv add --dev <pkg>   # Add dev dependency

# Code quality
ruff format .        # Format code
ruff check --fix .   # Lint and auto-fix
pyright              # Type check
pre-commit run --all-files  # Run all pre-commit hooks

# Testing (when implemented)
pytest               # Run tests
pytest --cov         # Run with coverage
```

## Documentation Reference
- **Setup & Architecture**: See `README.md`
- **Custom LLM Providers**: See `docs/evaluating.MD`
- **Usage Examples**: See `README.md` → "Usage" section
- **Model Configuration**: See `README.md` → "Models" section

## Docker
```bash
docker-compose up    # Run via Docker
```

---
For detailed information, see README.md and docs/
# Test
