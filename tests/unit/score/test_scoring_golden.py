"""Scoring a committed results.csv must reproduce the committed scores exactly.

These guard the published numbers: any change to percentages, the VERA formula,
rounding, worst-band selection, or the risk breakdown fails a case here. For an
intended change, run `tests/fixtures/golden_scoring/regenerate.py` and review
the diff of the `expected/` files.
"""

import importlib.util
import json
from pathlib import Path

import pytest

# The helpers live beside the golden data so the regenerate script and this
# test can't drift apart. tests/fixtures/ is not a package, so load by path.
GOLDEN = Path(__file__).resolve().parents[2] / "fixtures/golden_scoring"
_spec = importlib.util.spec_from_file_location("regenerate", GOLDEN / "regenerate.py")
assert _spec and _spec.loader
regenerate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regenerate)


@pytest.mark.parametrize("case_dir", regenerate.cases(), ids=lambda p: p.name)
def test_scores_match_the_committed_golden_files(case_dir: Path, tmp_path: Path):
    scores = regenerate.score_case(case_dir, tmp_path)

    for name in regenerate.OUTPUTS:
        expected_file = case_dir / "expected" / name
        produced_file = scores / name
        assert produced_file.is_file() == expected_file.is_file(), name
        if expected_file.is_file():
            produced = json.loads(produced_file.read_text())
            expected = json.loads(expected_file.read_text())
            assert produced == expected, f"{case_dir.name}/{name} changed"
