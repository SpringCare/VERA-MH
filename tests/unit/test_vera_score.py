"""Tests for the ``vera score`` command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import vera
from vera_cli import config as cli_config
from vera_cli import score

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_FIXTURE = REPO_ROOT / "tests/fixtures/eval_tsv_rebuild__20260415_140037"
PERSONAS_FIXTURE = REPO_ROOT / "tests/fixtures/personas_with_risk.tsv"


@pytest.fixture(autouse=True)
def clear_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cli_config.VERA_RUN_CONFIG_ENV, raising=False)


def _evaluation_run(tmp_path: Path) -> Path:
    """Copy the checked-in evaluation fixture so scoring can write beside it."""
    run = tmp_path / "j_haiku_run"
    shutil.copytree(EVAL_FIXTURE, run)
    return run


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _scoring_config(results: Path, **overrides: object) -> dict:
    scoring: dict[str, object] = {
        "results": str(results),
        "output": None,
        "personas": None,
        "skip_risk_analysis": False,
    }
    scoring.update(overrides)
    return {"scoring": scoring}


def test_score_is_registered_and_has_help() -> None:
    parser = vera.build_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["score", "--help"])
    assert exit_info.value.code == 0


def test_score_requires_a_results_path() -> None:
    """`-r` cannot be `required=True` — a config may supply it — so check it here."""
    parser = vera.build_parser()
    with pytest.raises(cli_config.ConfigError, match="-r/--results"):
        score.resolve_configs(parser.parse_args(["score"]))


def test_missing_results_file_fails_before_any_work(tmp_path: Path) -> None:
    parser = vera.build_parser()
    args = parser.parse_args(["score", "-r", str(tmp_path / "absent.csv")])
    with pytest.raises(cli_config.ConfigError, match="--results does not exist"):
        score.resolve_configs(args)


def test_cli_defaults_resolve_to_no_personas_and_no_output(tmp_path: Path) -> None:
    """The legacy `data/SI/personas.tsv` default is deliberately not carried over."""
    results = _evaluation_run(tmp_path) / "results.csv"
    parser = vera.build_parser()

    (resolved,) = score.resolve_configs(
        parser.parse_args(["score", "-r", str(results)])
    )

    assert resolved.scoring is not None
    assert resolved.scoring.results == str(results.resolve())
    assert resolved.scoring.personas is None
    assert resolved.scoring.output is None
    assert resolved.scoring.skip_risk_analysis is False


def test_cli_paths_resolve_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI paths follow the working directory; config paths follow the repo root."""
    run = _evaluation_run(tmp_path)
    monkeypatch.chdir(run)
    parser = vera.build_parser()

    (resolved,) = score.resolve_configs(
        parser.parse_args(["score", "-r", "results.csv"])
    )

    assert resolved.scoring is not None
    assert resolved.scoring.results == str((run / "results.csv").resolve())


def test_config_paths_resolve_from_repository_root(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        _scoring_config(
            Path("tests/fixtures/eval_tsv_rebuild__20260415_140037/results.csv"),
            personas="tests/fixtures/personas_with_risk.tsv",
        ),
    )
    parser = vera.build_parser()

    (resolved,) = score.resolve_configs(
        parser.parse_args(["score", "--config", str(config)])
    )

    assert resolved.scoring is not None
    assert resolved.scoring.results == str((EVAL_FIXTURE / "results.csv").resolve())
    assert resolved.scoring.personas == str(PERSONAS_FIXTURE.resolve())


@pytest.mark.parametrize(
    "field", ["results", "output", "personas", "skip_risk_analysis"]
)
def test_config_requires_every_scoring_field(tmp_path: Path, field: str) -> None:
    """A stored config states every value; nothing is filled in behind the caller."""
    results = _evaluation_run(tmp_path) / "results.csv"
    data = _scoring_config(results)
    del data["scoring"][field]  # type: ignore[union-attr]
    config = _write_config(tmp_path, data)
    parser = vera.build_parser()

    with pytest.raises(
        cli_config.ConfigError, match=f"missing required field: {field}"
    ):
        score.resolve_configs(parser.parse_args(["score", "--config", str(config)]))


@pytest.mark.parametrize(
    "flag",
    [
        ["-r", "tests/fixtures/eval_tsv_rebuild__20260415_140037/results.csv"],
        ["-o", "out.json"],
        ["--personas", "tests/fixtures/personas_with_risk.tsv"],
        ["--skip-risk-analysis"],
    ],
)
def test_config_rejects_any_run_defining_cli_flag(
    tmp_path: Path, flag: list[str]
) -> None:
    results = _evaluation_run(tmp_path) / "results.csv"
    config = _write_config(tmp_path, _scoring_config(results))
    parser = vera.build_parser()

    with pytest.raises(cli_config.ConfigError, match="cannot be combined"):
        score.resolve_configs(
            parser.parse_args(["score", "--config", str(config), *flag])
        )


def test_config_rejects_sections_score_does_not_own(tmp_path: Path) -> None:
    """A key this command would ignore is better rejected than silently dropped."""
    config = _write_config(tmp_path, {"judging": {}})
    parser = vera.build_parser()

    with pytest.raises(cli_config.ConfigError, match="unknown top-level config field"):
        score.resolve_configs(parser.parse_args(["score", "--config", str(config)]))


def test_debug_may_accompany_config(tmp_path: Path) -> None:
    """`--debug` is invocation-only, so the config-or-flags rule does not apply."""
    results = _evaluation_run(tmp_path) / "results.csv"
    config = _write_config(tmp_path, _scoring_config(results))
    parser = vera.build_parser()

    (resolved,) = score.resolve_configs(
        parser.parse_args(["score", "--config", str(config), "--debug"])
    )

    assert resolved.invocation.debug is True


def test_print_round_trips_through_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """`--print` emits a command that resolves to the same run."""
    results = _evaluation_run(tmp_path) / "results.csv"
    parser = vera.build_parser()
    args = parser.parse_args(["score", "-r", str(results), "--print"])
    (original,) = score.resolve_configs(args)

    assert score.run(args) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("uv run python vera.py score")

    payload = printed.split("=", 1)[1].rsplit(" uv run", 1)[0].strip("'")
    monkeypatch.setenv(cli_config.VERA_RUN_CONFIG_ENV, payload)
    (replayed,) = score.resolve_configs(parser.parse_args(["score"]))

    assert replayed == original


def test_scoring_writes_scores_beside_the_results_csv(tmp_path: Path) -> None:
    run = _evaluation_run(tmp_path)
    parser = vera.build_parser()

    assert score.run(parser.parse_args(["score", "-r", str(run / "results.csv")])) == 0

    assert (run / "scores/scores.json").is_file()


def test_no_personas_skips_the_risk_breakdown_entirely(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The one place `vera score` deliberately diverges from legacy `judge/score.py`.

    Legacy defaulted to `data/SI/personas.tsv` and, for any other target's
    personas file, wrote an empty `scores_by_risk.json` that looked like a
    result. No personas file now means no file at all, plus a message saying so.
    """
    run = _evaluation_run(tmp_path)
    parser = vera.build_parser()

    assert score.run(parser.parse_args(["score", "-r", str(run / "results.csv")])) == 0

    assert (run / "scores/scores.json").is_file()
    assert not (run / "scores/scores_by_risk.json").exists()
    assert "skipping risk-level analysis" in capsys.readouterr().out.lower()


def test_personas_restores_the_risk_breakdown(tmp_path: Path) -> None:
    """Passing `--personas` is exactly the legacy behavior."""
    run = _evaluation_run(tmp_path)
    parser = vera.build_parser()

    exit_code = score.run(
        parser.parse_args(
            [
                "score",
                "-r",
                str(run / "results.csv"),
                "--personas",
                str(PERSONAS_FIXTURE),
            ]
        )
    )

    assert exit_code == 0
    assert (run / "scores/scores_by_risk.json").is_file()


def test_skip_risk_analysis_wins_over_a_given_personas_file(tmp_path: Path) -> None:
    run = _evaluation_run(tmp_path)
    parser = vera.build_parser()

    exit_code = score.run(
        parser.parse_args(
            [
                "score",
                "-r",
                str(run / "results.csv"),
                "--personas",
                str(PERSONAS_FIXTURE),
                "--skip-risk-analysis",
            ]
        )
    )

    assert exit_code == 0
    assert not (run / "scores/scores_by_risk.json").exists()


def test_output_overrides_where_the_scores_json_lands(tmp_path: Path) -> None:
    run = _evaluation_run(tmp_path)
    destination = tmp_path / "elsewhere/custom.json"
    parser = vera.build_parser()

    exit_code = score.run(
        parser.parse_args(
            ["score", "-r", str(run / "results.csv"), "-o", str(destination)]
        )
    )

    assert exit_code == 0
    assert destination.is_file()
