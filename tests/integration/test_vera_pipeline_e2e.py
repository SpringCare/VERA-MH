"""End-to-end runs of ``vera pipeline``: one mocked, one live.

`tests/unit/test_vera_pipeline.py` covers resolution and stage wiring with the
stages mocked. The tests here run all three stages (generate, judge, score) for
real on one persona and 4 turns:

- The mocked test swaps every LLM for `MockLLM` at the factory, so it needs no
  network or keys and runs with the rest of the integration suite.
- The live test lets all three stages call real providers. It is kept as small
  as it can be while still exercising the parts a mock cannot: two user-side
  models from different providers (so two generation runs, two evaluations,
  and the "folders to pool" report) and two judges (so `results.csv` carries
  both). Run it with
  ``uv run pytest -m live tests/integration/test_vera_pipeline_e2e.py``.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import vera
from llm_clients import Role
from llm_clients.llm_factory import LLMFactory
from tests.mocks.mock_llm import MockLLM
from vera_cli import config as cli_config

# Cheap models the team's gateway serves. That catalog, not the provider's,
# decides what is usable here: it has no Haiku, so the Claude model is the
# cheapest it does list. Claude is a user-side model only. As a judge through
# the gateway it answers the rubric prompt in plain text instead of the JSON
# schema, which fails every question; that is a judging issue independent of
# `vera pipeline`, so both judges are OpenAI models.
#
# Pinned here, in one place, rather than read from `llm_clients.config.Config`:
# that class has no cheap-tier constants on this branch. Once #223 lands
# (`Config.DEFAULT_*` plus a live availability check), these belong on `Config`
# so a retirement is caught there instead of failing this run.
USER_MODELS = ["gpt-5.4-nano", "claude-sonnet-4-5"]
JUDGE_MODELS = ["gpt-5.4-nano", "gpt-5.6-luna"]
CHATBOT_MODEL = "gpt-5.4-nano"
TIMEOUT_SECONDS = 600
TURNS = 4


def _config(
    output: Path, *, chatbot: str, user_models: list[str], judge_models: list[str]
) -> dict:
    """A pipeline config shaped like `configs/recommended-SI.json`, shrunk."""
    return {
        "target": "SI",
        "generation": {
            "chatbot": {"name": chatbot, "repeats": 1},
            "user": [{"name": name, "repeats": 1} for name in user_models],
            "turns": TURNS,
            "output": str(output),
            "max_concurrent": 2,
            "max_total_words": None,
            "persona_speaks_first": True,
            "sessions": None,
        },
        "judging": {
            # No provider parameters: every judge model must share one set.
            "models": [{"name": name, "repeats": 1} for name in judge_models],
            "max_concurrent": 2,
            "per_judge": False,
        },
        "scoring": {"personas": None, "skip_risk_analysis": True},
    }


def _reported_folders(stdout: str) -> list[Path]:
    """The evaluation folders listed after the pipeline's "folders to pool" line."""
    lines = stdout.splitlines()
    start = next(index for index, line in enumerate(lines) if "folders to pool" in line)
    folders = []
    for line in lines[start + 1 :]:
        if not line.startswith("  "):
            break
        folders.append(Path(line.strip()))
    return folders


def _mock_llm(**kwargs) -> MockLLM:
    """Stand-in for `LLMFactory.create_llm` that returns a `MockLLM`.

    Callers pass provider settings (`model_name`, `model`, `temperature`, ...)
    that `MockLLM` does not accept, so only the identity arguments are kept.
    Persona and chatbot replies are tagged so the transcript shows who spoke;
    the judge answers every question with `MockLLM`'s schema example.
    """
    role = kwargs["role"]
    tag = {Role.PERSONA: "persona", Role.PROVIDER: "chatbot"}.get(role, "judge")
    return MockLLM(
        name=kwargs["name"],
        role=role,
        system_prompt=kwargs.get("system_prompt"),
        responses=[f"{tag} message {n}" for n in range(1, TURNS + 1)],
    )


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_pipeline_runs_end_to_end_with_mocked_llms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(cli_config.VERA_RUN_CONFIG_ENV, raising=False)

    config = tmp_path / "pipeline.json"
    config.write_text(
        json.dumps(
            _config(
                tmp_path / "output",
                chatbot="mock-bot",
                user_models=["mock-user"],
                judge_models=["mock-judge"],
            )
        ),
        encoding="utf-8",
    )

    # One patch covers all three call sites (persona, chatbot, judge): they
    # share the `LLMFactory` class object.
    with patch.object(LLMFactory, "create_llm", side_effect=_mock_llm):
        exit_code = vera.main(["pipeline", "--config", str(config), "--sample", "1"])
    assert exit_code == 0

    # With a single evaluation there is no "folders to pool" report to read.
    (run_folder,) = (tmp_path / "output").iterdir()
    (evaluation,) = (run_folder / "evaluations").iterdir()
    (transcript,) = (run_folder / "conversations").glob("*.txt")
    text = transcript.read_text(encoding="utf-8")
    # The persona speaks first and the chatbot gets the last turn: 2 + 2.
    assert text.count("persona message") == TURNS // 2
    assert text.count("chatbot message") == TURNS // 2

    with open(evaluation / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert {row["judge_model"] for row in rows} == {"mock-judge"}

    scores = json.loads((evaluation / "scores" / "scores.json").read_text())
    assert "vera_score" in scores["aggregates"]


@pytest.mark.integration
@pytest.mark.live
@pytest.mark.enable_socket
@pytest.mark.timeout(TIMEOUT_SECONDS)
def test_pipeline_runs_end_to_end_on_cheap_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    missing = [
        key for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if not os.getenv(key)
    ]
    if missing:
        pytest.skip(f"missing API keys: {missing}")
    monkeypatch.delenv(cli_config.VERA_RUN_CONFIG_ENV, raising=False)

    config = tmp_path / "pipeline.json"
    config.write_text(
        json.dumps(
            _config(
                tmp_path / "output",
                chatbot=CHATBOT_MODEL,
                user_models=USER_MODELS,
                judge_models=JUDGE_MODELS,
            )
        ),
        encoding="utf-8",
    )

    # `--sample` is invocation-only, so it may accompany `--config`; it caps
    # both generation and judging at one persona.
    exit_code = vera.main(["pipeline", "--config", str(config), "--sample", "1"])
    assert exit_code == 0

    evaluations = _reported_folders(capsys.readouterr().out)
    assert len(evaluations) == 2, "one evaluation per user-side model"

    for evaluation in evaluations:
        # <run>/evaluations/<j_*>: everything stays under the run it judged.
        run_folder = evaluation.parent.parent
        # Resolved: on macOS the temp dir is reached through /private.
        assert run_folder.resolve().is_relative_to(tmp_path.resolve())
        transcripts = list((run_folder / "conversations").glob("*.txt"))
        assert len(transcripts) == 1, f"expected one transcript in {run_folder}"

        with open(evaluation / "results.csv", newline="", encoding="utf-8") as f:
            judges = {row["judge_model"] for row in csv.DictReader(f)}
        assert judges == set(JUDGE_MODELS)

        scores = json.loads((evaluation / "scores" / "scores.json").read_text())
        assert "vera_score" in scores["aggregates"]
