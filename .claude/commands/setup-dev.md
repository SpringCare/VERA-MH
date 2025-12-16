Set up the complete development environment for VERA-MH:

1. Check if `uv` is installed, install if needed
2. Run `uv sync` to install dependencies
3. Check if `.env` exists, prompt to copy from `.env.example` if not
4. Remind about configuring API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
5. Run `pre-commit install` to set up git hooks
6. Initialize testing infrastructure:
   - Create `tests/` directory structure if it doesn't exist
   - Check if pytest and pytest-cov are installed
   - Create basic `tests/conftest.py` with common fixtures
   - Create `tests/__init__.py`
   - Create example test file `tests/test_example.py` as a template
   - Add pytest configuration to `pyproject.toml` if needed
7. Verify the setup by running a quick sanity check
8. Run a test to verify everything works
