"""Unit tests for generate.py resume behavior."""

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
