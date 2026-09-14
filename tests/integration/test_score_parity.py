"""Parity between the legacy `judge/score.py` CLI and `vera.py score`.

TEMPORARY — DELETE WITH `judge/score.py`'s `main`.

This module exists only for the window in which both entry points ship. Adding
`vera score` moved the body of `judge.score.main` into `judge.score.run_scoring`
so both callers share it; the one claim worth proving while the legacy script
still has users is that the move changed no numbers. `docs/architecture.md`
Phase 1 deletes the legacy entry points and makes `vera.py` the only one — at
that moment this comparison loses its second side, so delete this file rather
than adapting it.

Both sides run against copies of the same checked-in evaluation fixture, so the
comparison needs no API keys, no model calls, and no generated output.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import vera
from judge import score as score_domain

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_FIXTURE = REPO_ROOT / "tests/fixtures/eval_tsv_rebuild__20260415_140037"
PERSONAS_FIXTURE = REPO_ROOT / "tests/fixtures/personas_with_risk.tsv"


def _evaluation_copy(tmp_path: Path, name: str) -> Path:
    """An independent copy of the fixture, so each side writes its own scores."""
    run = tmp_path / name
    shutil.copytree(EVAL_FIXTURE, run)
    return run


def _scores(run: Path) -> dict:
    return json.loads((run / "scores/scores.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "extra_legacy,extra_unified",
    [
        # The risk step off on both sides: the narrowest comparison, isolating
        # the dimension and aggregate scoring that the extraction touched.
        (["--skip-risk-analysis"], ["--skip-risk-analysis"]),
        # The risk step on, with the same personas file named explicitly. This
        # is the configuration in which the two CLIs are meant to be identical;
        # they diverge only when `--personas` is omitted, which is asserted in
        # tests/unit/test_vera_score.py rather than here.
        (
            ["--personas-tsv", str(PERSONAS_FIXTURE)],
            ["--personas", str(PERSONAS_FIXTURE)],
        ),
    ],
    ids=["skip-risk", "with-personas"],
)
def test_both_entry_points_produce_the_same_scores(
    tmp_path: Path, extra_legacy: list[str], extra_unified: list[str]
) -> None:
    legacy_run = _evaluation_copy(tmp_path, "legacy")
    unified_run = _evaluation_copy(tmp_path, "unified")

    legacy_argv = [
        "score.py",
        "-r",
        str(legacy_run / "results.csv"),
        *extra_legacy,
    ]
    with patch.object(sys, "argv", legacy_argv):
        assert score_domain.main() == 0

    parser = vera.build_parser()
    args = parser.parse_args(
        ["score", "-r", str(unified_run / "results.csv"), *extra_unified]
    )
    assert args.handler(args) == 0

    assert _scores(unified_run) == _scores(legacy_run)


def test_legacy_main_still_reports_a_missing_results_csv(tmp_path: Path) -> None:
    """The extraction raises where `main` used to print and return 1.

    `main` catches that and keeps its old exit code, so the legacy contract is
    unchanged for callers that check the status rather than the exception.
    """
    with patch.object(sys, "argv", ["score.py", "-r", str(tmp_path / "absent.csv")]):
        assert score_domain.main() == 1
