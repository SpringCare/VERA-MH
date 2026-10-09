"""Tests for the ``vera pool`` command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import vera
from vera_cli import config as cli_config
from vera_cli import pool

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_FIXTURE = REPO_ROOT / "tests/fixtures/eval_tsv_rebuild__20260415_140037"
EVAL_FIXTURE_REL = "tests/fixtures/eval_tsv_rebuild__20260415_140037"
PERSONAS_FIXTURE = REPO_ROOT / "tests/fixtures/personas_with_risk.tsv"


@pytest.fixture(autouse=True)
def clear_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cli_config.VERA_RUN_CONFIG_ENV, raising=False)


def _evaluations(tmp_path: Path) -> list[Path]:
    """Two copies of the checked-in evaluation fixture, as two runs to pool."""
    runs = []
    for name in ("j_run_a", "j_run_b"):
        run = tmp_path / name
        shutil.copytree(EVAL_FIXTURE, run)
        runs.append(run)
    return runs


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _pooling_config(evaluations: list[str], **overrides: object) -> dict:
    pooling: dict[str, object] = {
        "evaluations": evaluations,
        "output": "output",
        "personas": None,
        "skip_risk_analysis": False,
    }
    pooling.update(overrides)
    return {"pooling": pooling}


def _pooled_folder(output: Path) -> Path:
    (folder,) = [path for path in output.iterdir() if path.is_dir()]
    return folder


def test_pool_is_registered_and_has_help() -> None:
    parser = vera.build_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["pool", "--help"])
    assert exit_info.value.code == 0


def test_pool_requires_evaluations() -> None:
    parser = vera.build_parser()
    with pytest.raises(cli_config.ConfigError, match="--evaluations"):
        pool.resolve_configs(parser.parse_args(["pool"]))


def test_pool_needs_at_least_two_evaluations(tmp_path: Path) -> None:
    run, _ = _evaluations(tmp_path)
    parser = vera.build_parser()
    with pytest.raises(cli_config.ConfigError, match="at least two"):
        pool.resolve_configs(parser.parse_args(["pool", "--evaluations", str(run)]))


def test_missing_evaluation_fails_before_any_work(tmp_path: Path) -> None:
    run, _ = _evaluations(tmp_path)
    parser = vera.build_parser()
    args = parser.parse_args(
        ["pool", "--evaluations", str(run), str(tmp_path / "absent")]
    )
    with pytest.raises(cli_config.ConfigError, match="evaluation folder"):
        pool.resolve_configs(args)


def test_evaluation_may_be_a_folder_or_its_results_csv(tmp_path: Path) -> None:
    run_a, run_b = _evaluations(tmp_path)
    parser = vera.build_parser()

    (resolved,) = pool.resolve_configs(
        parser.parse_args(
            ["pool", "--evaluations", str(run_a), str(run_b / "results.csv")]
        )
    )

    assert resolved.pooling is not None
    assert resolved.pooling.evaluations == [
        str(run_a.resolve()),
        str((run_b / "results.csv").resolve()),
    ]


def test_cli_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No personas (unlike the legacy script) and output/ in the working directory."""
    run_a, run_b = _evaluations(tmp_path)
    monkeypatch.chdir(tmp_path)
    parser = vera.build_parser()

    (resolved,) = pool.resolve_configs(
        parser.parse_args(["pool", "--evaluations", "j_run_a", "j_run_b"])
    )

    assert resolved.pooling is not None
    assert resolved.pooling.evaluations == [str(run_a.resolve()), str(run_b.resolve())]
    assert resolved.pooling.output == str((tmp_path / "output").resolve())
    assert resolved.pooling.personas is None
    assert resolved.pooling.skip_risk_analysis is False


def test_config_paths_resolve_from_repository_root(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        _pooling_config(
            [EVAL_FIXTURE_REL, f"{EVAL_FIXTURE_REL}/results.csv"],
            personas="tests/fixtures/personas_with_risk.tsv",
        ),
    )
    parser = vera.build_parser()

    (resolved,) = pool.resolve_configs(
        parser.parse_args(["pool", "--config", str(config)])
    )

    assert resolved.pooling is not None
    assert resolved.pooling.evaluations[0] == str(EVAL_FIXTURE.resolve())
    assert resolved.pooling.output == str((REPO_ROOT / "output").resolve())
    assert resolved.pooling.personas == str(PERSONAS_FIXTURE.resolve())


@pytest.mark.parametrize(
    "field", ["evaluations", "output", "personas", "skip_risk_analysis"]
)
def test_config_requires_every_pooling_field(tmp_path: Path, field: str) -> None:
    data = _pooling_config([EVAL_FIXTURE_REL, EVAL_FIXTURE_REL])
    del data["pooling"][field]  # type: ignore[union-attr]
    config = _write_config(tmp_path, data)
    parser = vera.build_parser()

    with pytest.raises(
        cli_config.ConfigError, match=f"missing required field: {field}"
    ):
        pool.resolve_configs(parser.parse_args(["pool", "--config", str(config)]))


def test_config_rejects_one_evaluation(tmp_path: Path) -> None:
    config = _write_config(tmp_path, _pooling_config([EVAL_FIXTURE_REL]))
    parser = vera.build_parser()

    with pytest.raises(cli_config.ConfigError, match="at least two"):
        pool.resolve_configs(parser.parse_args(["pool", "--config", str(config)]))


def test_config_rejects_any_run_defining_cli_flag(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path, _pooling_config([EVAL_FIXTURE_REL, EVAL_FIXTURE_REL])
    )
    parser = vera.build_parser()

    with pytest.raises(cli_config.ConfigError, match="cannot be combined"):
        pool.resolve_configs(
            parser.parse_args(["pool", "--config", str(config), "-o", "elsewhere"])
        )


def test_config_rejects_sections_pool_does_not_own(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"scoring": {}})
    parser = vera.build_parser()

    with pytest.raises(cli_config.ConfigError, match="unknown top-level config field"):
        pool.resolve_configs(parser.parse_args(["pool", "--config", str(config)]))


def test_print_round_trips_through_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    run_a, run_b = _evaluations(tmp_path)
    parser = vera.build_parser()
    args = parser.parse_args(
        ["pool", "--evaluations", str(run_a), str(run_b), "--print"]
    )
    (original,) = pool.resolve_configs(args)

    assert pool.run(args) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("uv run python vera.py pool")

    payload = printed.split("=", 1)[1].rsplit(" uv run", 1)[0].strip("'")
    monkeypatch.setenv(cli_config.VERA_RUN_CONFIG_ENV, payload)
    (replayed,) = pool.resolve_configs(parser.parse_args(["pool"]))

    assert replayed == original


def test_pool_writes_merged_results_and_scores(tmp_path: Path) -> None:
    run_a, run_b = _evaluations(tmp_path)
    output = tmp_path / "pooled"
    parser = vera.build_parser()

    exit_code = pool.run(
        parser.parse_args(
            ["pool", "--evaluations", str(run_a), str(run_b), "-o", str(output)]
        )
    )

    assert exit_code == 0
    folder = _pooled_folder(output)
    rows = len(pd.read_csv(EVAL_FIXTURE / "results.csv"))
    assert len(pd.read_csv(folder / "results.csv")) == 2 * rows
    metadata = json.loads((folder / "pool_metadata.json").read_text())
    assert metadata["rows_per_source"] == [rows, rows]
    scores = json.loads((folder / "scores/scores.json").read_text())
    assert scores["judge_model"] == "pooled"
    assert not (folder / "scores/scores_by_risk.json").exists()


def test_personas_adds_the_risk_breakdown(tmp_path: Path) -> None:
    run_a, run_b = _evaluations(tmp_path)
    output = tmp_path / "pooled"
    parser = vera.build_parser()

    exit_code = pool.run(
        parser.parse_args(
            [
                "pool",
                "--evaluations",
                str(run_a),
                str(run_b),
                "-o",
                str(output),
                "--personas",
                str(PERSONAS_FIXTURE),
            ]
        )
    )

    assert exit_code == 0
    assert (_pooled_folder(output) / "scores/scores_by_risk.json").is_file()


def test_skip_risk_analysis_wins_over_a_given_personas_file(tmp_path: Path) -> None:
    run_a, run_b = _evaluations(tmp_path)
    output = tmp_path / "pooled"
    parser = vera.build_parser()

    exit_code = pool.run(
        parser.parse_args(
            [
                "pool",
                "--evaluations",
                str(run_a),
                str(run_b),
                "-o",
                str(output),
                "--personas",
                str(PERSONAS_FIXTURE),
                "--skip-risk-analysis",
            ]
        )
    )

    assert exit_code == 0
    assert not (_pooled_folder(output) / "scores/scores_by_risk.json").exists()
