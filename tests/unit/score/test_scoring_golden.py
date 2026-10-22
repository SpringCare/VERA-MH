"""Scoring a committed results.csv must reproduce the committed scores exactly.

These guard the published numbers: any change to percentages, the VERA formula,
rounding, worst-band selection, or the risk breakdown fails a case here. For an
intended change, run `python -m tests.safe_changes.golden_scoring` and review
the diff of the `expected/` files. See docs/safe-changes.md.
"""

import json
from pathlib import Path

import pytest

from tests.safe_changes import golden_scoring


@pytest.mark.parametrize("case_dir", golden_scoring.cases(), ids=lambda p: p.name)
def test_scores_match_the_committed_golden_files(case_dir: Path, tmp_path: Path):
    scores = golden_scoring.score_case(case_dir, tmp_path)

    for name in golden_scoring.OUTPUTS:
        expected_file = case_dir / "expected" / name
        produced_file = scores / name
        assert produced_file.is_file() == expected_file.is_file(), name
        if expected_file.is_file():
            produced = json.loads(produced_file.read_text())
            expected = json.loads(expected_file.read_text())
            assert produced == expected, f"{case_dir.name}/{name} changed"
