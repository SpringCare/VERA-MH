Fix failing tests iteratively until they pass, then show coverage focused on branch changes.

**Parameters:**
- Optional focus area (e.g., "focus on integration tests", "check async issues")
- `--all`: Show full project coverage (default: only branch-touched files)
- `--no-cov`: Skip coverage report entirely

**Workflow:**

1. **Initial Check**
   - Run `/test` to get baseline status
   - Count failing tests and identify specific failures
   - Note warnings if present
   - If user provided focus area, prioritize related failures

2. **Iterative Fix Loop**

   While tests are failing:

   a. **Analyze Failure**
      - Review error message and traceback from `/test` output
      - Identify failing test file and function
      - Determine failure type:
        - Assertion error (expected vs actual)
        - Exception/error in test
        - Exception/error in implementation
        - Missing fixture or setup
        - Import error

   b. **Investigate Root Cause**
      - Read the failing test code
      - Read the implementation code being tested
      - Check for recent changes that might have broken it
      - Identify whether issue is in test or implementation

   c. **Fix the Issue**
      - Make targeted fix (test code or implementation)
      - Follow existing patterns and conventions
      - Use appropriate fixtures (MockLLM, tmp_path, etc.)
      - Don't skip/ignore assertions - fix the root cause

   d. **Verify Fix**
      - Run specific test: `uv run pytest path/to/test.py::test_name -v`
      - If passes, mark as fixed
      - If fails, iterate on fix

   e. **Full Re-check**
      - Run `/test` again to verify no regressions
      - Update count of remaining failures
      - Continue to next failure if any remain

3. **Address Warnings**
   - After all tests pass, review warning summary from `/test` output
   - `/test` provides categorized warnings (deprecations, unused fixtures, etc.)
   - Address warnings systematically:
     - Deprecations: Update to new APIs
     - Unused fixtures: Remove or mark with underscore
     - Import issues: Fix import paths
   - Goal: clean test output (zero warnings if possible)

4. **Coverage Report (Branch-Focused)**

   **Default behavior** (focus on branch changes):
   - Get files changed in branch: `git diff --name-only main...HEAD`
   - Filter to Python source files (exclude tests, configs)
   - Parse coverage.json from final `/test` run
   - Extract coverage for branch-touched files only
   - Display focused report:
     ```
     Coverage Summary (Branch Files):
     Overall: 78.5% (4 files changed)

     Files touched in this branch:
     🟢 generate.py: 92% (lines 45-48, 156 missing)
     🟡 judge/runner.py: 81% (lines 89-103 missing)
     🟡 llm_clients/claude_llm.py: 75% (lines 120-135 missing)
     🔴 judge/score.py: 65% (lines 23-45, 78-92 missing)
     ```

   **With `--all` flag**:
   - Show full `/test` coverage report for entire project

   **With `--no-cov` flag**:
   - Skip coverage analysis entirely (faster)

5. **Final Summary**
   ```
   ✅ All tests passing (X passed, 0 failed)
   ⚠️ Y warnings (list specific warnings if any)

   ✏️ Fixed N failing tests
   📊 Coverage: 78.5% (branch files) | 75.2% (full project)

   Recommendations:
   - [Specific suggestions based on coverage gaps]
   - [Warnings that should be addressed]
   ```

**Coverage Color Coding:**
- 🔴 <70%: Needs attention
- 🟡 70-85%: Acceptable
- 🟢 ≥85%: Good coverage

**Key Principles:**
- Leverage `/test` for all test execution and coverage collection
- Fix root causes, not symptoms
- Maintain existing test patterns and conventions
- Branch-focused coverage tracks quality of changes
- Full suite must pass before reporting success

**Example Usage:**

```bash
# Fix all tests, show branch coverage
/fix-tests

# Focus on specific test types
/fix-tests focus on integration tests

# Fix tests and show full project coverage
/fix-tests --all

# Quick fix without coverage (faster)
/fix-tests --no-cov
```

**When to Use:**
- **Use `/fix-tests`**: When CI is red or tests are failing
- **Use `/test`**: For quick one-time test runs
- **Use `/create-tests`**: After tests pass, to add tests for gaps

**Relationship to `/test`:**
- `/test` runs once and reports (with pass/fail counts, warning summaries, coverage)
- `/fix-tests` runs `/test` repeatedly, fixing issues until clean
- Both use same pytest infrastructure and coverage tools
- `/fix-tests` leverages `/test`'s warning summaries to guide fixes
