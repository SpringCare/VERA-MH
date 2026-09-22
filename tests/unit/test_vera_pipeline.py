"""Tests for the ``vera pipeline`` command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import vera
from vera_cli import config as cli_config
from vera_cli import pipeline


@pytest.fixture(autouse=True)
def clear_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cli_config.VERA_RUN_CONFIG_ENV, raising=False)


def _cli(*extra: str) -> list[str]:
    return [
        "pipeline",
        "-c",
        "claude-sonnet-5",
        "-u",
        "gpt-5.2:1",
        "-j",
        "gpt-5.4:1",
        "--target",
        "SI",
        *extra,
    ]


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _resolved_config() -> dict:
    """The canonical config form, taken from the CLI path's own resolution."""
    parser = vera.build_parser()
    (resolved,) = pipeline.resolve_configs(parser.parse_args(_cli()))
    return resolved.to_dict()


def test_pipeline_is_registered_and_has_help() -> None:
    parser = vera.build_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["pipeline", "--help"])
    assert exit_info.value.code == 0


def test_shorthand_resolves_all_three_stages() -> None:
    parser = vera.build_parser()
    (resolved,) = pipeline.resolve_configs(parser.parse_args(_cli()))

    generation = resolved.generation.generation
    assert generation is not None
    assert generation.chatbot.name == "claude-sonnet-5"
    assert [model.name for model in generation.user] == ["gpt-5.2"]
    assert [model.name for model in resolved.judge_models] == ["gpt-5.4"]
    assert resolved.rubric.rubric_file.endswith("data/SI/rubric.tsv")


def test_one_target_supplies_both_personas_and_rubric() -> None:
    """The point of `--target` on a pipeline: both halves come from one bundle."""
    parser = vera.build_parser()
    (resolved,) = pipeline.resolve_configs(parser.parse_args(_cli()))

    generation = resolved.generation.generation
    assert generation is not None
    personas_dir = Path(generation.personas[0]).parent
    assert Path(resolved.rubric.rubric_file).parent == personas_dir


@pytest.mark.parametrize("flag", [["-j", "gpt-5.4:1"], ["--target", "SI"]])
def test_shorthand_requires_judge_and_target(flag: list[str]) -> None:
    """Neither can be `required=True`: a config may supply them instead."""
    argv = ["pipeline", "-c", "bot", "-u", "user:1", *flag]
    parser = vera.build_parser()
    with pytest.raises(cli_config.ConfigError, match="pipeline requires"):
        pipeline.resolve_configs(parser.parse_args(argv))


def test_judge_params_apply_to_every_judge_model() -> None:
    parser = vera.build_parser()
    argv = _cli("--judge-params", "reasoning_effort=low")
    (resolved,) = pipeline.resolve_configs(parser.parse_args(argv))

    assert all(
        model.extra_params == {"reasoning_effort": "low"}
        for model in resolved.judge_models
    )


def test_print_round_trips_through_a_config(tmp_path: Path) -> None:
    """`--print` emits a document that resolves back to the same pipeline."""
    original = _resolved_config()
    config = _write_config(tmp_path, original)
    parser = vera.build_parser()

    (replayed,) = pipeline.resolve_configs(
        parser.parse_args(["pipeline", "--config", str(config)])
    )

    assert replayed.to_dict() == original


def test_config_must_not_supply_conversations(tmp_path: Path) -> None:
    """Generation supplies them; stating them too is a contradiction.

    This is the one field a pipeline config deliberately omits, so naming it is
    rejected outright rather than silently overridden by the generated folder.
    """
    data = _resolved_config()
    data["judging"]["conversations"] = ["output/somewhere"]
    config = _write_config(tmp_path, data)
    parser = vera.build_parser()

    with pytest.raises(
        cli_config.ConfigError, match="must not set judging.conversations"
    ):
        pipeline.resolve_configs(
            parser.parse_args(["pipeline", "--config", str(config)])
        )


def test_config_rejects_run_defining_cli_flags(tmp_path: Path) -> None:
    config = _write_config(tmp_path, _resolved_config())
    parser = vera.build_parser()

    with pytest.raises(cli_config.ConfigError, match="cannot be combined"):
        pipeline.resolve_configs(
            parser.parse_args(
                ["pipeline", "--config", str(config), "-c", "other-model"]
            )
        )


def test_config_target_and_explicit_rubrics_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    """A target already determines the rubric; naming both is a contradiction."""
    data = _resolved_config()
    data["target"] = "SI"
    # A target is mutually exclusive with the explicit generation fields too, so
    # drop those to isolate the judging-side rule this test is about.
    for field in ("personas", "persona_context_template"):
        data["generation"].pop(field)
    config = _write_config(tmp_path, data)
    parser = vera.build_parser()

    with pytest.raises(cli_config.ConfigError, match="mutually exclusive"):
        pipeline.resolve_configs(
            parser.parse_args(["pipeline", "--config", str(config)])
        )


def test_scoring_section_is_optional_and_defaults_to_skipping_risk(
    tmp_path: Path,
) -> None:
    """With no personas file, scoring skips the risk breakdown (see score.py)."""
    data = _resolved_config()
    del data["scoring"]
    config = _write_config(tmp_path, data)
    parser = vera.build_parser()

    (resolved,) = pipeline.resolve_configs(
        parser.parse_args(["pipeline", "--config", str(config)])
    )

    assert resolved.scoring_personas is None
    assert resolved.skip_risk_analysis is False


def test_target_all_is_rejected(tmp_path: Path) -> None:
    """Deferred until a use case asks for it, as `vera judge` defers it."""
    parser = vera.build_parser()
    argv = ["pipeline", "-c", "bot", "-u", "user:1", "-j", "judge:1", "--target", "all"]

    with pytest.raises(cli_config.ConfigError, match="does not support --target all"):
        pipeline.resolve_configs(parser.parse_args(argv))

    # And through a config, where `generate` would otherwise fan it out.
    data = _resolved_config()
    data["target"] = "all"
    for field in ("personas", "persona_context_template"):
        data["generation"].pop(field)
    config = _write_config(tmp_path, data)
    with pytest.raises(cli_config.ConfigError, match="does not support --target all"):
        pipeline.resolve_configs(
            parser.parse_args(["pipeline", "--config", str(config)])
        )


def test_pipeline_declares_no_stage_specific_knobs() -> None:
    """The shorthand stays unambiguous: no flag that could mean either stage.

    `--output` and `--max-concurrent` mean different things to generation and
    judging, so pipeline offers neither rather than inventing a prefixed
    spelling for one side. They come from `--config`.
    """
    parser = vera.build_parser()
    for ambiguous in ["--output", "--max-concurrent", "--conversations"]:
        with pytest.raises(SystemExit):
            parser.parse_args(_cli(ambiguous, "x"))


def test_shipped_recommended_config_resolves() -> None:
    """The checked-in profile is an artifact that would otherwise rot silently.

    It publishes the same profile `scripts/run_recommended_vera_pipeline.sh`
    runs, so a field going stale has to fail here rather than at the start of
    an expensive run.
    """
    data = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/recommended-SI.json").read_text(
            encoding="utf-8"
        )
    )
    data["generation"]["chatbot"]["name"] = "claude-sonnet-5"

    parser = vera.build_parser()
    env = json.dumps(data)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(cli_config.VERA_RUN_CONFIG_ENV, env)
        (resolved,) = pipeline.resolve_configs(parser.parse_args(["pipeline"]))

    generation = resolved.generation.generation
    assert generation is not None
    # The two published user-side models, the published judge and its profile.
    assert [model.name for model in generation.user] == [
        "gpt-5.2",
        "claude-opus-4-5-20251101",
    ]
    assert [model.name for model in resolved.judge_models] == ["gpt-5.4"]
    assert resolved.judge_models[0].extra_params == {"reasoning_effort": "low"}
    assert generation.turns == 30


def test_pipeline_runs_three_stages_and_never_pools(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Pooling is a separate command; pipeline names the folders instead.

    `docs/architecture.md` defines this command as three stages and lists
    `vera pool` separately, so a multi-user-model run must not quietly produce a
    fourth artifact. It reports the evaluation folders so pooling needs no
    log-scraping.
    """

    async def fake_generate(run_configs):
        generation = run_configs[0].generation
        return [f"run_{index}" for index, _ in enumerate(generation.user)]

    async def fake_judge_and_score(run, run_folder):
        return f"{run_folder}/evaluations/j_x"

    monkeypatch.setattr(pipeline.generate_command, "_execute", fake_generate)
    monkeypatch.setattr(pipeline, "_judge_and_score", fake_judge_and_score)
    assert not hasattr(pipeline, "_pool")

    parser = vera.build_parser()
    asyncio.run(pipeline._execute(pipeline.resolve_configs(parser.parse_args(_cli()))))
    single = capsys.readouterr().out
    # One evaluation has nothing to pool, so nothing is reported.
    assert "folders to pool" not in single

    both = pipeline.resolve_configs(
        parser.parse_args(
            [
                "pipeline",
                "-c",
                "bot",
                "-u",
                "gpt-5.2:1",
                "claude-opus-4-5-20251101:1",
                "-j",
                "gpt-5.4:1",
                "--target",
                "SI",
            ]
        )
    )
    asyncio.run(pipeline._execute(both))
    reported = capsys.readouterr().out
    assert "folders to pool" in reported
    assert "run_0/evaluations/j_x" in reported
    assert "run_1/evaluations/j_x" in reported
