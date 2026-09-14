"""Tests for the ``vera pipeline`` command."""

from __future__ import annotations

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


def test_target_all_gives_each_run_its_own_rubric() -> None:
    """Each target's conversations must be judged against that target's rubric.

    With a single target checked in this is a one-element case, but it pins the
    invariant that the rubric is resolved per run rather than once and shared.
    """
    parser = vera.build_parser()
    argv = ["pipeline", "-c", "bot", "-u", "user:1", "-j", "judge:1", "--target", "all"]

    resolved = pipeline.resolve_configs(parser.parse_args(argv))

    for run in resolved:
        generation = run.generation.generation
        assert generation is not None
        assert (
            Path(run.rubric.rubric_file).parent == Path(generation.personas[0]).parent
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
