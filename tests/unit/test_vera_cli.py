"""Tests for the first unified CLI feature: ``vera generate``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import generate as generation_domain
import vera
from utils.config_schema import ModelSpec
from vera_cli import (
    config as cli_config,
)
from vera_cli import generate, targets


@pytest.fixture(autouse=True)
def clear_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cli_config.VERA_RUN_CONFIG_ENV, raising=False)


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _generation_config(**overrides: object) -> dict:
    generation: dict[str, object] = {
        "chatbot": {"name": "gpt-5", "repeats": 1},
        "user": [{"name": "claude-sonnet-5", "repeats": 1}],
        "personas": ["data/SI/personas.tsv"],
        "turns": 30,
        "output": "output",
        "max_concurrent": None,
        "max_total_words": None,
        "persona_speaks_first": True,
        "sessions": None,
        "persona_context_template": "data/SI/persona_context_template.txt",
    }
    generation.update(overrides)
    return {"generation": generation}


def _write_target(tmp_path: Path, *, persona_count: int = 1) -> Path:
    target_dir = tmp_path / "target"
    target_dir.mkdir(parents=True)
    for filename in (
        "rubric.tsv",
        "rubric_prompt.txt",
        "question_prompt.txt",
        "persona_context.txt",
    ):
        (target_dir / filename).write_text("fixture", encoding="utf-8")
    personas = []
    for index in range(persona_count):
        filename = f"personas_{index}.tsv"
        (target_dir / filename).write_text("Name\tPrompt\n", encoding="utf-8")
        personas.append(filename)
    manifest = target_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rubric_file": "rubric.tsv",
                "rubric_prompt_beginning_file": "rubric_prompt.txt",
                "question_prompt_file": "question_prompt.txt",
                "personas": personas,
                "persona_context_template_file": "persona_context.txt",
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_entry_point_stays_thin_and_generate_only() -> None:
    source = Path(vera.__file__).read_text(encoding="utf-8")
    parser = vera.build_parser()

    assert len(source.splitlines()) < 50
    assert "generate_conversations" not in source
    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["generate", "--help"])
    assert help_exit.value.code == 0


def test_model_shorthand_preserves_provider_colons() -> None:
    assert ModelSpec.from_shorthand("llama3:8b") == ModelSpec(
        name="llama3:8b", repeats=1, extra_params={}
    )
    assert ModelSpec.from_shorthand("gpt-5:3") == ModelSpec(
        name="gpt-5", repeats=3, extra_params={}
    )


def test_target_and_personas_resolve_same_generation_inputs() -> None:
    parser = vera.build_parser()
    target_args = parser.parse_args(
        ["generate", "-c", "gpt-5", "-u", "claude-sonnet-5", "--target", "SI"]
    )
    personas_args = parser.parse_args(
        [
            "generate",
            "-c",
            "gpt-5",
            "-u",
            "claude-sonnet-5",
            "--personas",
            "SI",
        ]
    )

    assert generate.resolve_configs(target_args) == generate.resolve_configs(
        personas_args
    )


def test_cli_defaults_resolve_before_print(capsys: pytest.CaptureFixture) -> None:
    result = vera.main(
        [
            "generate",
            "-c",
            "gpt-5",
            "-u",
            "claude-sonnet-5:2",
            "--personas",
            "SI",
            "--print",
        ]
    )

    rendered = capsys.readouterr().out.strip()
    assert result == 0
    assert rendered.startswith(f"{cli_config.VERA_RUN_CONFIG_ENV}=")
    assert '"turns":30' in rendered
    assert '"repeats":2' in rendered
    assert "data/SI/personas.tsv" in rendered


def test_generation_delegates_every_value_to_generate_main() -> None:
    with patch.object(
        generation_domain, "main", new_callable=AsyncMock
    ) as generate_main:
        generate_main.return_value = ([], "output/generated")
        result = vera.main(
            [
                "generate",
                "-c",
                "gpt-5",
                "-u",
                "claude-sonnet-5:3",
                "--target",
                "SI",
                "--turns",
                "4",
                "--max-concurrent",
                "2",
                "--max-total-words",
                "100",
                "--provider-speaks-first",
                "--sessions",
                "intake,coaching",
                "--sample",
                "2",
            ]
        )

    assert result == 0
    assert generate_main.await_count == 1
    assert generate_main.await_args is not None
    kwargs = generate_main.await_args.kwargs
    assert kwargs["persona_model_config"] == {"model": "claude-sonnet-5"}
    assert kwargs["agent_model_config"] == {"model": "gpt-5", "name": "gpt-5"}
    assert kwargs["runs_per_prompt"] == 3
    assert kwargs["max_turns"] == 4
    assert kwargs["max_concurrent"] == 2
    assert kwargs["max_total_words"] == 100
    assert kwargs["max_personas"] == 2
    assert kwargs["persona_speaks_first"] is False
    assert kwargs["session_types"] == ["intake", "coaching"]
    assert kwargs["resume"] is False
    assert kwargs["persona_files"] == [
        str((cli_config.ROOT / "data/SI/personas.tsv").resolve())
    ]
    assert kwargs["persona_context_template_path"] == str(
        (cli_config.ROOT / "data/SI/persona_context_template.txt").resolve()
    )


def test_each_user_model_gets_one_generate_main_call(tmp_path: Path) -> None:
    manifest = _write_target(tmp_path, persona_count=2)
    with patch.object(
        generation_domain, "main", new_callable=AsyncMock
    ) as generate_main:
        generate_main.return_value = ([], "output/generated")
        vera.main(
            [
                "generate",
                "-c",
                "gpt-5",
                "-u",
                "claude-sonnet-5:2",
                "gpt-4o:3",
                "--personas",
                str(manifest),
            ]
        )

    assert generate_main.await_count == 2
    repeats = [call.kwargs["runs_per_prompt"] for call in generate_main.await_args_list]
    assert repeats == [2, 3]
    assert all(
        len(call.kwargs["persona_files"]) == 2 for call in generate_main.await_args_list
    )


def test_config_requires_all_generation_behavior_fields(tmp_path: Path) -> None:
    config_data = _generation_config()
    del config_data["generation"]["turns"]
    config = _write_config(tmp_path, config_data)

    with pytest.raises(SystemExit) as error:
        vera.main(["generate", "--config", str(config)])

    assert error.value.code == 2


def test_config_paths_resolve_from_repository_root(tmp_path: Path) -> None:
    config = _write_config(tmp_path, _generation_config())
    args = vera.build_parser().parse_args(["generate", "--config", str(config)])

    generation = generate.resolve_configs(args)[0].generation
    assert generation.personas == [
        str((cli_config.ROOT / "data/SI/personas.tsv").resolve())
    ]
    assert generation.output == str((cli_config.ROOT / "output").resolve())
    assert generation.persona_context_template == str(
        (cli_config.ROOT / "data/SI/persona_context_template.txt").resolve()
    )


def test_config_target_expands_to_concrete_generation_paths(tmp_path: Path) -> None:
    generation = _generation_config()["generation"]
    del generation["personas"]
    del generation["persona_context_template"]
    config = _write_config(tmp_path, {"target": "SI", "generation": generation})
    args = vera.build_parser().parse_args(["generate", "--config", str(config)])

    resolved = generate.resolve_configs(args)[0].to_dict()
    assert "target" not in resolved
    assert resolved["generation"]["personas"] == [
        str((cli_config.ROOT / "data/SI/personas.tsv").resolve())
    ]


def test_config_target_rejects_explicit_persona_component(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"target": "SI", **_generation_config()})

    with pytest.raises(SystemExit) as error:
        vera.main(["generate", "--config", str(config)])

    assert error.value.code == 2


@pytest.mark.parametrize(
    "flag",
    [
        ["-c", "gpt-5"],
        ["-u", "gpt-5"],
        ["--target", "SI"],
        ["--personas", "SI"],
        ["--turns", "3"],
        ["--output", "elsewhere"],
        ["--max-concurrent", "2"],
        ["--max-total-words", "10"],
        ["--provider-speaks-first"],
        ["--sessions", "intake"],
    ],
)
def test_config_rejects_any_run_defining_cli_flag(
    tmp_path: Path, flag: list[str]
) -> None:
    """Every run-defining flag is refused alongside ``--config``.

    Covers each flag behaviorally rather than asserting a hand-maintained list,
    so a flag that is wrongly classified as invocation-only fails here.
    """
    config = _write_config(tmp_path, _generation_config())

    with pytest.raises(SystemExit) as error:
        vera.main(["generate", "--config", str(config), *flag])

    assert error.value.code == 2


def test_config_allows_debug_sample_and_print_controls(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = _write_config(tmp_path, _generation_config())

    result = vera.main(
        [
            "generate",
            "--config",
            str(config),
            "--sample",
            "1",
            "--debug",
            "--print",
        ]
    )

    assert result == 0
    rendered = capsys.readouterr().out
    assert rendered.startswith(cli_config.VERA_RUN_CONFIG_ENV)
    assert '"debug":true' in rendered
    assert '"sample":1' in rendered
    assert rendered.rstrip().endswith("uv run python vera.py generate")


def test_persisted_invocation_metadata_is_accepted(tmp_path: Path) -> None:
    config_data = _generation_config()
    config_data["invocation"] = {"debug": True, "sample": 2}
    config = _write_config(tmp_path, config_data)
    args = vera.build_parser().parse_args(["generate", "--config", str(config)])

    invocation = generate.resolve_configs(args)[0].invocation

    assert invocation.debug is True
    assert invocation.sample == 2


@pytest.mark.parametrize("field", ["sample", "debug", "print"])
def test_invocation_controls_are_rejected_inside_config(
    tmp_path: Path, field: str
) -> None:
    config_data = _generation_config()
    config_data[field] = True
    config = _write_config(tmp_path, config_data)

    with pytest.raises(SystemExit) as error:
        vera.main(["generate", "--config", str(config)])

    assert error.value.code == 2


def test_environment_config_is_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cli_config.VERA_RUN_CONFIG_ENV, json.dumps(_generation_config()))

    result = vera.main(["generate", "--print"])

    assert result == 0


def test_config_and_environment_config_are_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, _generation_config())
    monkeypatch.setenv(cli_config.VERA_RUN_CONFIG_ENV, json.dumps(_generation_config()))

    with pytest.raises(SystemExit) as error:
        vera.main(["generate", "--config", str(config)])

    assert error.value.code == 2


def test_incomplete_target_fails_before_dispatch(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"personas": ["personas.tsv"]}), encoding="utf-8")

    with (
        patch.object(generation_domain, "main", new_callable=AsyncMock) as runner,
        pytest.raises(SystemExit) as error,
    ):
        vera.main(
            [
                "generate",
                "-c",
                "gpt-5",
                "-u",
                "claude-sonnet-5",
                "--target",
                str(manifest),
            ]
        )

    assert error.value.code == 2
    runner.assert_not_awaited()


def test_target_all_produces_one_config_per_manifest(tmp_path: Path) -> None:
    first = _write_target(tmp_path / "first")
    second = _write_target(tmp_path / "second")
    args = vera.build_parser().parse_args(
        ["generate", "-c", "gpt-5", "-u", "claude-sonnet-5", "--target", "all"]
    )

    with patch.object(targets, "target_catalog", return_value=[first, second]):
        run_configs = generate.resolve_configs(args)

    assert len(run_configs) == 2


def test_generate_requires_explicit_target_or_personas() -> None:
    with pytest.raises(SystemExit) as error:
        vera.main(["generate", "-c", "gpt-5", "-u", "claude-sonnet-5"])

    assert error.value.code == 2


def test_config_rejects_judging_section(tmp_path: Path) -> None:
    """``generate`` rejects a judging block rather than silently ignoring it."""
    config = _write_config(
        tmp_path,
        {**_generation_config(), "judging": {"rubrics": ["data/SI/rubric.tsv"]}},
    )

    with pytest.raises(SystemExit) as error:
        vera.main(["generate", "--config", str(config)])

    assert error.value.code == 2


def test_sample_must_be_positive() -> None:
    with pytest.raises(SystemExit) as error:
        vera.main(
            [
                "generate",
                "-c",
                "gpt-5",
                "-u",
                "claude-sonnet-5",
                "--target",
                "SI",
                "--sample",
                "0",
            ]
        )

    assert error.value.code == 2
