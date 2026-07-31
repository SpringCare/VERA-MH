"""Tests for the unified ``vera.py`` CLI adapters and source contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import vera
from utils.config_schema import (
    GenerationConfig,
    JudgingConfig,
    ModelSpec,
    RubricSpec,
    RunConfig,
)


@pytest.fixture(autouse=True)
def clear_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(vera.VERA_RUN_CONFIG_ENV, raising=False)


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_model_shorthand_preserves_colon_model_names() -> None:
    assert ModelSpec.from_shorthand("llama3:8b") == ModelSpec(name="llama3:8b")
    assert ModelSpec.from_shorthand("gpt-5:3") == ModelSpec(name="gpt-5", repeats=3)


def test_judge_rejects_config_mixed_with_conversations(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        {
            "judging": {
                "models": [{"name": "gpt-5"}],
                "rubrics": [{"name": "SI"}],
                "conversations": ["output/conversations"],
            }
        },
    )

    with pytest.raises(SystemExit) as error:
        vera.main(
            [
                "judge",
                "--config",
                str(config),
                "--conversations",
                "other/conversations",
            ]
        )

    assert error.value.code == 2


def test_config_allows_cli_sample_only(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = _write_config(
        tmp_path,
        {
            "generation": {
                "chatbot": {"name": "gpt-5"},
                "user": [{"name": "claude-sonnet-5"}],
                "personas": ["data/SI/personas.tsv"],
            }
        },
    )

    with patch.object(
        vera, "_run_generation", new_callable=AsyncMock
    ) as run_generation:
        result = vera.main(["generate", "--config", str(config), "--sample", "1"])

    assert result == 0
    rendered = json.loads(capsys.readouterr().out)
    assert "sample" not in rendered
    assert "debug" not in rendered
    assert rendered["generation"]["personas"] == [
        str((vera.ROOT / "data/SI/personas.tsv").resolve())
    ]
    assert run_generation.await_args is not None
    assert run_generation.await_args.kwargs["sample"] == 1


@pytest.mark.parametrize("control", ["--debug", "--print"])
def test_config_rejects_non_sample_cli_controls(tmp_path: Path, control: str) -> None:
    config = _write_config(
        tmp_path,
        {
            "generation": {
                "chatbot": {"name": "gpt-5"},
                "user": [{"name": "claude-sonnet-5"}],
                "personas": ["data/SI/personas.tsv"],
            }
        },
    )

    with pytest.raises(SystemExit) as error:
        vera.main(["generate", "--config", str(config), control])

    assert error.value.code == 2


@pytest.mark.parametrize("field", ["sample", "debug", "print"])
def test_debug_controls_are_not_config_fields(tmp_path: Path, field: str) -> None:
    config = _write_config(
        tmp_path,
        {
            "generation": {
                "chatbot": {"name": "gpt-5"},
                "user": [{"name": "claude-sonnet-5"}],
                "personas": ["data/SI/personas.tsv"],
            },
            field: 1,
        },
    )

    with pytest.raises(SystemExit) as error:
        vera.main(["generate", "--config", str(config), "--print"])

    assert error.value.code == 2


def test_judge_config_owns_conversation_paths(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        {
            "judging": {
                "models": [{"name": "gpt-5"}],
                "rubrics": [{"name": "SI"}],
                "conversations": ["output/conversations"],
            }
        },
    )

    args = vera.build_parser().parse_args(["judge", "--config", str(config)])
    run_config = vera.resolve_run_config(
        args, cli_flag_names=("judge", "rubric", "conversations")
    )
    assert run_config.judging is not None
    assert run_config.judging.conversations == [
        str((vera.ROOT / "output/conversations").resolve())
    ]


@pytest.mark.asyncio
async def test_generation_delegates_to_domain_function() -> None:
    run_config = RunConfig(
        generation=GenerationConfig(
            chatbot=ModelSpec(name="chatbot", extra_params={"temperature": 0.2}),
            user=[ModelSpec(name="user", repeats=3, extra_params={"top_p": 0.8})],
            personas=["personas.tsv"],
        )
    )

    with patch(
        "generate_conversations.run_generation", new_callable=AsyncMock
    ) as run_generation:
        run_generation.return_value = ([], "output/generated")
        outputs = await vera._run_generation(run_config, sample=2, debug=False)

    assert outputs == ["output/generated"]
    assert run_generation.await_args is not None
    kwargs = run_generation.await_args.kwargs
    assert kwargs["persona_model_config"] == {"model": "user", "top_p": 0.8}
    assert kwargs["agent_model_config"] == {
        "model": "chatbot",
        "name": "chatbot",
        "temperature": 0.2,
    }
    assert kwargs["runs_per_prompt"] == 3
    assert kwargs["max_personas"] == 2
    assert kwargs["persona_files"] == ["personas.tsv"]


@pytest.mark.asyncio
async def test_generation_target_resolves_manifest_personas(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "target"
    manifest_dir.mkdir()
    manifest = manifest_dir / "rubric_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rubric_file": "rubric.tsv",
                "rubric_prompt_beginning_file": "rubric_prompt.txt",
                "question_prompt_file": "question_prompt.txt",
                "personas": ["personas.tsv"],
            }
        ),
        encoding="utf-8",
    )
    run_config = RunConfig(
        generation=GenerationConfig(
            chatbot=ModelSpec(name="chatbot"),
            user=[ModelSpec(name="user")],
        ),
        target="target",
    )

    with (
        patch.object(vera, "_target_manifests", return_value=[manifest]),
        patch(
            "generate_conversations.run_generation", new_callable=AsyncMock
        ) as run_generation,
    ):
        run_generation.return_value = ([], "output/generated")
        outputs = await vera._run_generation(run_config, sample=None, debug=False)

    assert outputs == ["output/generated"]
    assert run_generation.await_args is not None
    assert run_generation.await_args.kwargs["persona_files"] == [
        str(manifest_dir / "personas.tsv")
    ]
    assert "rubric_manifest" not in run_generation.await_args.kwargs


@pytest.mark.asyncio
async def test_generation_target_without_personas_has_actionable_error(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "rubric_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rubric_file": "rubric.tsv",
                "rubric_prompt_beginning_file": "rubric_prompt.txt",
                "question_prompt_file": "question_prompt.txt",
            }
        ),
        encoding="utf-8",
    )
    run_config = RunConfig(
        generation=GenerationConfig(
            chatbot=ModelSpec(name="chatbot"),
            user=[ModelSpec(name="user")],
        ),
        target="target",
    )

    with (
        patch.object(vera, "_target_manifests", return_value=[manifest]),
        patch(
            "generate_conversations.run_generation", new_callable=AsyncMock
        ) as run_generation,
        pytest.raises(vera.ConfigError, match="manifest .* defines no personas"),
    ):
        await vera._run_generation(run_config, sample=None, debug=False)

    run_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_judging_delegates_to_domain_function() -> None:
    run_config = RunConfig(
        judging=JudgingConfig(
            models=[ModelSpec(name="judge", repeats=2)],
            rubrics=[RubricSpec(name="data/SI/rubric_manifest.json")],
            conversations=["output/conversations"],
        )
    )
    with patch("judge.run_judging", new_callable=AsyncMock) as run_judging:
        run_judging.return_value = "output/evaluations/j_run"
        outputs = await vera._run_judging(run_config, sample=1, debug=False)

    assert outputs == ["output/evaluations/j_run"]
    assert run_judging.await_args is not None
    kwargs = run_judging.await_args.kwargs
    assert kwargs["conversation_folder"] == "output/conversations"
    assert kwargs["judge_models"] == {"judge": 1}
    assert kwargs["limit"] == 1
    assert kwargs["rubric_manifest"] == str(
        (vera.ROOT / "data/SI/rubric_manifest.json").resolve()
    )


def test_score_delegates_to_score_adapter() -> None:
    with patch.object(vera, "_run_scoring") as run_scoring:
        assert vera.main(["score", "--results", "results.csv"]) == 0

    run_scoring.assert_called_once_with("results.csv")


def test_score_adapter_delegates_to_domain_function() -> None:
    with patch("judge.score.score_results_file", return_value=0) as score_file:
        vera._run_scoring("results.csv")

    score_file.assert_called_once_with(
        "results.csv",
        personas_tsv=str(vera.ROOT / "data" / "SI" / "personas.tsv"),
    )


def test_pool_delegates_to_existing_pool_function() -> None:
    args = argparse.Namespace(evaluations=["one", "two"], print=False)
    with patch(
        "scripts.pool_vera_scores.pool_evaluation_directories"
    ) as pool_evaluation_directories:
        assert vera.cmd_pool(args) == 0

    pool_evaluation_directories.assert_called_once_with(
        ["one", "two"],
        vera.ROOT / "output",
        personas_tsv=vera.ROOT / "data" / "SI" / "personas.tsv",
    )


def test_pipeline_chains_generation_judging_and_scoring() -> None:
    with (
        patch.object(
            vera,
            "_run_generation",
            new_callable=AsyncMock,
            return_value=["output/generated"],
        ) as run_generation,
        patch.object(
            vera,
            "_run_judging",
            new_callable=AsyncMock,
            return_value=["output/evaluations/j_run"],
        ) as run_judging,
        patch.object(vera, "_run_scoring") as run_scoring,
    ):
        result = vera.main(
            [
                "pipeline",
                "-c",
                "chatbot",
                "-u",
                "user",
                "-j",
                "judge",
                "--personas",
                "data/SI/personas.tsv",
                "--rubric",
                "data/SI/rubric_manifest.json",
            ]
        )

    assert result == 0
    run_generation.assert_awaited_once()
    assert run_judging.await_args is not None
    assert run_judging.await_args.kwargs["conversations"] == ["output/generated"]
    run_scoring.assert_called_once_with("output/evaluations/j_run/results.csv")
