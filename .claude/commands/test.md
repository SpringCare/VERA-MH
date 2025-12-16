Run the test suite for VERA-MH with comprehensive options:

1. Check if tests directory exists, suggest running /setup-dev if not
2. Check if pytest and pytest-cov are installed
3. Parse command arguments:
   - `--unit`: Run only unit tests (fast)
   - `--integration`: Run only integration tests
   - `--e2e`: Run only E2E tests (slow)
   - `--no-cov`: Skip coverage reporting
   - `[path]`: Run tests in specific file or directory
4. Run pytest with appropriate markers and coverage
5. Display test results with pass/fail counts
6. Summarize warnings if any are present:
   - Count total warnings
   - Categorize by type (deprecation, unused fixture, etc.)
   - Show specific warning messages with file locations
   - Example: "⚠️ 5 warnings: 3 deprecations, 2 unused fixtures"
7. Show coverage report with color-coded feedback:
   - 🔴 Red: <70% coverage (needs attention)
   - 🟡 Yellow: 70-85% coverage (approaching target)
   - 🟢 Green: ≥85% coverage (meets target)
   - Note: Project has dual coverage targets:
     - 30% = CI minimum (--cov-fail-under in pyproject.toml)
     - 75%+ = Quality target for new/changed code
8. If coverage below threshold, suggest specific files to add tests for
9. If tests fail, show failure details and suggest fixes
10. Remind about test pyramid structure (unit > integration > e2e)

Examples:
- `/test` - Run all tests with coverage
- `/test --unit` - Run only fast unit tests
- `/test tests/unit/llm_clients/` - Run specific test directory
- `/test --no-cov` - Run without coverage (faster)
