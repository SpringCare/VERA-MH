Verify that a code change works correctly without hitting live APIs.

IMPORTANT: Never run `generate.py`, `judge.py`, or `run_pipeline.py` directly — these make live LLM API calls. Verification must use the test suite only.

Steps:
1. Identify what changed (git diff --stat) to understand the scope
2. Run the relevant tests with `uv run pytest -m "not live" -v` targeting changed modules where possible
3. Run the full non-live suite to check for regressions: `uv run pytest -m "not live"`
4. Report: which tests passed/failed, coverage delta if relevant, and whether the change behaves as expected based on test output

If the change cannot be verified without a live API call, say so explicitly and suggest what a targeted live test would look like (but do not run it).
