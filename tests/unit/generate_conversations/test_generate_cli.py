"""Unit tests for generate.py resume behavior."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import generate


@pytest.mark.asyncio
async def test_main_resume_uses_existing_run_folder(tmp_path: Path) -> None:
    """Resume mode should reuse provided run folder and avoid nesting."""
    run_folder = tmp_path / "p_mock_persona__a_mock_agent__t4__r1__20260331_120000"
    run_folder.mkdir(parents=True, exist_ok=True)

    persona_model_config = {"model": "mock-persona"}
    agent_model_config = {"model": "mock-agent", "name": "mock-agent"}

    with patch("generate.ConversationRunner") as mock_runner_cls:
        mock_runner = mock_runner_cls.return_value
        mock_runner.run_conversations = AsyncMock(return_value=[])

        _, output_folder = await generate.main(
            persona_model_config=persona_model_config,
            agent_model_config=agent_model_config,
            max_turns=4,
            runs_per_prompt=1,
            output_folder=str(run_folder),
            resume=True,
            verbose=False,
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

    persona_model_config = {"model": "different-persona"}
    agent_model_config = {"model": "mock-agent", "name": "mock-agent"}

    with pytest.raises(ValueError, match="persona model does not match"):
        await generate.main(
            persona_model_config=persona_model_config,
            agent_model_config=agent_model_config,
            max_turns=4,
            runs_per_prompt=1,
            output_folder=str(run_folder),
            resume=True,
            verbose=False,
        )


@pytest.mark.asyncio
async def test_main_rubric_manifest_loads_personas_from_manifest(
    tmp_path: Path,
) -> None:
    """--rubric-manifest should select personas from the manifest, not the default."""
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

    persona_model_config = {"model": "mock-persona"}
    agent_model_config = {"model": "mock-agent", "name": "mock-agent"}

    with patch("generate.ConversationRunner") as mock_runner_cls:
        mock_runner = mock_runner_cls.return_value
        mock_runner.run_conversations = AsyncMock(return_value=[])

        await generate.main(
            persona_model_config=persona_model_config,
            agent_model_config=agent_model_config,
            max_turns=4,
            runs_per_prompt=1,
            output_folder=str(tmp_path / "out"),
            run_id="run1",
            rubric_manifest=str(manifest_path),
            verbose=False,
        )

    kwargs = mock_runner_cls.call_args.kwargs
    assert kwargs["persona_prompt_path"] == str(tmp_path / "personas_custom.tsv")
    assert kwargs["persona_context_template_path"] == str(
        tmp_path / "persona_context_custom.txt"
    )


@pytest.mark.asyncio
async def test_main_no_rubric_manifest_uses_default_personas(tmp_path: Path) -> None:
    """Omitting --rubric-manifest should keep today's fixed-default persona path."""
    persona_model_config = {"model": "mock-persona"}
    agent_model_config = {"model": "mock-agent", "name": "mock-agent"}

    with patch("generate.ConversationRunner") as mock_runner_cls:
        mock_runner = mock_runner_cls.return_value
        mock_runner.run_conversations = AsyncMock(return_value=[])

        await generate.main(
            persona_model_config=persona_model_config,
            agent_model_config=agent_model_config,
            max_turns=4,
            runs_per_prompt=1,
            output_folder=str(tmp_path / "out"),
            run_id="run1",
            verbose=False,
        )

    kwargs = mock_runner_cls.call_args.kwargs
    assert kwargs["persona_prompt_path"] == "data/personas.tsv"
    assert (
        kwargs["persona_context_template_path"] == "data/persona_context_template.txt"
    )


@pytest.mark.asyncio
async def test_main_rubric_manifest_without_personas_raises_value_error(
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
        await generate.main(
            persona_model_config={"model": "mock-persona"},
            agent_model_config={"model": "mock-agent", "name": "mock-agent"},
            max_turns=4,
            runs_per_prompt=1,
            output_folder=str(tmp_path / "out"),
            run_id="run1",
            rubric_manifest=str(manifest_path),
            verbose=False,
        )
