"""Parity between the legacy pool script and ``vera pool``.

TEMPORARY — DELETE WITH `legacy/pool_vera_scores.py`.

Both entry points call `score.run_pooling`; the claim worth proving while the
legacy script still has users is that, given the same inputs and the same
personas file, they write the same pooled results and scores. They diverge only
when no personas file is named (the legacy script defaults to the SI file,
`vera pool` skips the risk breakdown), which tests/unit/test_vera_pool.py covers.
Its end is the deferred removal of `legacy/` (`docs/roadmap.md`).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import vera
from legacy import pool_vera_scores as legacy_pool

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_FIXTURE = REPO_ROOT / "tests/fixtures/eval_tsv_rebuild__20260415_140037"
PERSONAS_FIXTURE = REPO_ROOT / "tests/fixtures/personas_with_risk.tsv"


def _sources(root: Path) -> list[Path]:
    """Two independent copies of the fixture, as the two evaluations to pool."""
    runs = []
    for name in ("j_run_a", "j_run_b"):
        run = root / name
        shutil.copytree(EVAL_FIXTURE, run)
        runs.append(run)
    return runs


def _pooled_folder(output: Path) -> Path:
    (folder,) = [path for path in output.iterdir() if path.is_dir()]
    return folder


def _without_paths(scores: dict) -> dict:
    """Drop the provenance block, which names each side's own source folders."""
    return {key: value for key, value in scores.items() if key != "pooled"}


@pytest.mark.parametrize(
    "extra_legacy,extra_unified",
    [
        (["--skip-risk-analysis"], ["--skip-risk-analysis"]),
        (
            ["--personas-tsv", str(PERSONAS_FIXTURE)],
            ["--personas", str(PERSONAS_FIXTURE)],
        ),
    ],
    ids=["skip-risk", "with-personas"],
)
def test_both_entry_points_produce_the_same_pool(
    tmp_path: Path, extra_legacy: list[str], extra_unified: list[str]
) -> None:
    legacy_out = tmp_path / "legacy_out"
    unified_out = tmp_path / "unified_out"
    legacy_sources = _sources(tmp_path / "legacy_src")
    unified_sources = _sources(tmp_path / "unified_src")

    legacy_argv = [
        "pool_vera_scores.py",
        *map(str, legacy_sources),
        "-o",
        str(legacy_out),
        *extra_legacy,
    ]
    with patch.object(sys, "argv", legacy_argv):
        assert legacy_pool.main() == 0

    parser = vera.build_parser()
    args = parser.parse_args(
        [
            "pool",
            "--evaluations",
            *map(str, unified_sources),
            "-o",
            str(unified_out),
            *extra_unified,
        ]
    )
    assert args.handler(args) == 0

    legacy_folder = _pooled_folder(legacy_out)
    unified_folder = _pooled_folder(unified_out)

    pd.testing.assert_frame_equal(
        pd.read_csv(unified_folder / "results.csv"),
        pd.read_csv(legacy_folder / "results.csv"),
    )
    for name in ("scores.json", "scores_by_risk.json"):
        legacy_file = legacy_folder / "scores" / name
        unified_file = unified_folder / "scores" / name
        assert legacy_file.is_file() == unified_file.is_file()
        if legacy_file.is_file():
            assert _without_paths(json.loads(unified_file.read_text())) == (
                _without_paths(json.loads(legacy_file.read_text()))
            )
    legacy_meta = json.loads((legacy_folder / "pool_metadata.json").read_text())
    unified_meta = json.loads((unified_folder / "pool_metadata.json").read_text())
    assert unified_meta["rows_per_source"] == legacy_meta["rows_per_source"]


def test_legacy_extract_from_log_still_finds_the_evaluation_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    log = tmp_path / "pipeline.log"
    log.write_text(
        "noise\nEvaluations saved to: /a/first\nmore\nEvaluations saved to: /b/last/\n"
    )
    with patch.object(
        sys, "argv", ["pool_vera_scores.py", "--extract-from-log", str(log)]
    ):
        assert legacy_pool.main() == 0
    assert capsys.readouterr().out.strip() == str(Path("/b/last").resolve())
