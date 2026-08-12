"""Unit tests for generate.py resume behavior."""

import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import generate


def _generation_kwargs(output_folder: str, **overrides: object) -> dict:
    values = {
        "persona_model_config": {"model": "mock-persona"},
        "agent_model_config": {"model": "mock-agent", "name": "mock-agent"},
        "persona_files": ["data/SI/personas.tsv"],
        "persona_extra_run_params": {},
        "agent_extra_run_params": {},
        "max_turns": 4,
        "runs_per_prompt": 1,
        "persona_names": None,
        "verbose": False,
        "output_folder": output_folder,
        "run_id": None,
        "max_concurrent": None,
        "max_total_words": None,
        "max_personas": None,
        "persona_speaks_first": True,
        "session_types": None,
        "resume": False,
        "persona_context_template_path": "data/SI/persona_context_template.txt",
    }
    values.update(overrides)
    return values


def test_main_requires_every_runtime_value() -> None:
    """The reusable generation function must not own CLI behavior defaults."""
    parameters = inspect.signature(generate.main).parameters.values()

    assert all(parameter.default is inspect.Parameter.empty for parameter in parameters)


@pytest.mark.asyncio
async def test_main_resume_uses_existing_run_folder(tmp_path: Path) -> None:
    """Resume mode should reuse provided run folder and avoid nesting."""
    run_folder = tmp_path / "p_mock_persona__a_mock_agent__t4__r1__20260331_120000"
    run_folder.mkdir(parents=True, exist_ok=True)

    with patch("generate_conversations.workflow.ConversationRunner") as mock_runner_cls:
        mock_runner = mock_runner_cls.return_value
        mock_runner.run_conversations = AsyncMock(return_value=[])

        _, output_folder = await generate.main(
            **_generation_kwargs(
                str(run_folder),
                resume=True,
                persona_model_config={"model": "mock-persona"},
            )
        )

    assert output_folder == str(run_folder)
    kwargs = mock_runner_cls.call_args.kwargs
    assert kwargs["folder_name"] == str(run_folder)
    assert kwargs["run_id"] == run_folder.name
    assert kwargs["resume"] is True


@pytest.mark.asyncio
async def test_main_resume_mismatch_raises_value_error(tmp_path: Path) -> None:
    """Resume mode should fail fast when run-folder metadata mismatches args."""
    run_folder = tmp_path / "p_mock_persona__a_mock_agent__t4__r1__20260331_120000"
    run_folder.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="persona model does not match"):
        await generate.main(
            **_generation_kwargs(
                str(run_folder),
                resume=True,
                persona_model_config={"model": "different-persona"},
            )
        )


@pytest.mark.asyncio
async def test_resolve_persona_inputs_loads_manifest_paths(
    tmp_path: Path,
) -> None:
    """Legacy callers can resolve persona inputs before calling ``main``."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "rubric_file": "rubric.tsv",
                "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
                "question_prompt_file": "question_prompt.txt",
                "personas": ["personas_custom.tsv"],
                "persona_context_template_file": "persona_context_custom.txt",
            }
        ),
        encoding="utf-8",
    )

    personas, context_template = await generate.resolve_persona_inputs(
        str(manifest_path)
    )

    assert personas == [str(tmp_path / "personas_custom.tsv")]
    assert context_template == str(tmp_path / "persona_context_custom.txt")


@pytest.mark.asyncio
async def test_resolve_persona_inputs_without_personas_raises_value_error(
    tmp_path: Path,
) -> None:
    """A manifest with no personas listed can't select a persona file."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "rubric_file": "rubric.tsv",
                "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
                "question_prompt_file": "question_prompt.txt",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no personas listed"):
        await generate.resolve_persona_inputs(str(manifest_path))
