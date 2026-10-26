"""Tests for the run record: `config.json` in each run folder."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import vera_mh.generate.run as generation_domain
from vera_mh import cli as vera
from vera_mh.utils.config_schema import (
    GenerationConfig,
    InvocationConfig,
    ModelSpec,
    RunConfig,
)
from vera_mh.utils.run_record import (
    HASH_FILE,
    RECORD_FILE,
    build_run_record,
    check_record_inputs,
    file_checksum,
    record_hash,
    require_run_record,
    with_start_time,
)
from vera_mh.utils.run_root import create_run_root
from vera_mh.vera_cli import config as cli_config
from vera_mh.vera_cli import generate, judge


@pytest.fixture(autouse=True)
def clear_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cli_config.VERA_RUN_CONFIG_ENV, raising=False)


def _model(name: str) -> ModelSpec:
    return ModelSpec(name=name, repeats=1, extra_params={})


def _run_config(tmp_path: Path, **generation_overrides: object) -> RunConfig:
    personas = tmp_path / "personas.tsv"
    personas.write_text("Name\tPrompt\nAlex\tHi\n", encoding="utf-8")
    template = tmp_path / "context.txt"
    template.write_text("{name}", encoding="utf-8")
    values: dict[str, object] = {
        "chatbot": _model("gpt-5"),
        "user": [_model("claude-sonnet-5"), _model("gpt-4o")],
        "personas": [str(personas)],
        "turns": 30,
        "output": str(tmp_path / "output"),
        "max_concurrent": None,
        "max_total_words": None,
        "first_speaker": "user",
        "sessions": None,
        "persona_context_template": str(template),
        "user_language": "English",
    }
    values.update(generation_overrides)
    return RunConfig(
        invocation=InvocationConfig(debug=False, sample=None, into=None),
        generation=GenerationConfig(**values),  # type: ignore[arg-type]
    )


def test_record_holds_the_resolved_config_inputs_and_provenance(
    tmp_path: Path,
) -> None:
    run_config = _run_config(tmp_path)

    record = build_run_record(run_config)

    assert record["format_version"] == 1
    assert record["generation"] == run_config.generation.to_dict()  # type: ignore[union-attr]
    # `into` names a folder to continue, so it can't belong to a folder's record.
    assert record["invocation"] == {"debug": False, "sample": None}
    personas = run_config.generation.personas[0]  # type: ignore[union-attr]
    assert record["inputs"][personas] == file_checksum(personas)
    assert set(record["provenance"]) == {"vera_version", "git_commit", "started_at"}
    assert record["provenance"]["started_at"] is None


def test_start_time_is_stamped_on_a_copy(tmp_path: Path) -> None:
    record = build_run_record(_run_config(tmp_path))

    stamped = with_start_time(record, datetime(2026, 10, 26, 14, 30, 5))

    assert stamped["provenance"]["started_at"] == "2026-10-26T14:30:05"
    assert record["provenance"]["started_at"] is None


def test_hash_ignores_provenance_and_invocation(tmp_path: Path) -> None:
    record = build_run_record(_run_config(tmp_path))
    other = with_start_time(record, datetime(2026, 1, 1))
    other["invocation"] = {"debug": True, "sample": 2}

    assert record_hash(other) == record_hash(record)


def test_hash_changes_with_settings_and_inputs(tmp_path: Path) -> None:
    run_config = _run_config(tmp_path)
    record = build_run_record(run_config)
    fewer_turns = build_run_record(_run_config(tmp_path, turns=10))

    (tmp_path / "personas.tsv").write_text("Name\tPrompt\nSam\tHi\n", encoding="utf-8")
    edited_personas = build_run_record(run_config)

    assert record_hash(fewer_turns) != record_hash(record)
    assert record_hash(edited_personas) != record_hash(record)


def test_create_run_root_writes_the_record_and_its_hash(tmp_path: Path) -> None:
    record = build_run_record(_run_config(tmp_path))

    root = create_run_root(tmp_path / "output", "run_1", record)

    assert json.loads((root / RECORD_FILE).read_text(encoding="utf-8")) == record
    assert (root / HASH_FILE).read_text(encoding="utf-8").strip() == record_hash(record)


def test_a_plain_config_has_no_inputs_to_check() -> None:
    check_record_inputs({"generation": {}})


def test_a_changed_input_stops_a_rerun(tmp_path: Path) -> None:
    record = build_run_record(_run_config(tmp_path))
    check_record_inputs(record)

    (tmp_path / "personas.tsv").write_text("edited", encoding="utf-8")

    with pytest.raises(ValueError, match="personas.tsv"):
        check_record_inputs(record)


def test_a_missing_input_stops_a_rerun(tmp_path: Path) -> None:
    record = build_run_record(_run_config(tmp_path))
    (tmp_path / "context.txt").unlink()

    with pytest.raises(ValueError, match="missing"):
        check_record_inputs(record)


def test_each_user_model_folder_records_only_its_own_model(tmp_path: Path) -> None:
    """Rerunning one folder's record repeats that folder, not its siblings."""
    with patch.object(
        generation_domain, "run_generation", new_callable=AsyncMock
    ) as run_generation:
        run_generation.return_value = ([], str(tmp_path / "run"))
        vera.main(
            ["generate", "-c", "gpt-5", "-u", "claude-sonnet-5", "gpt-4o"]
            + ["--target", "SI", "-o", str(tmp_path)]
        )

    records = [call.kwargs["run_record"] for call in run_generation.await_args_list]
    assert [record["generation"]["user"] for record in records] == [
        [{"name": "claude-sonnet-5", "repeats": 1}],
        [{"name": "gpt-4o", "repeats": 1}],
    ]


def test_continuing_a_run_writes_no_new_record(tmp_path: Path) -> None:
    run = tmp_path / "p_claude-sonnet-5__a_gpt-5__t30__r1__20260101_000000"
    run.mkdir()
    with patch.object(
        generation_domain, "run_generation", new_callable=AsyncMock
    ) as run_generation:
        run_generation.return_value = ([], str(run))
        vera.main(
            ["generate", "-c", "gpt-5", "-u", "claude-sonnet-5", "--target", "SI"]
            + ["--into", str(run)]
        )

    assert run_generation.await_args.kwargs["run_record"] is None


def test_a_record_runs_again_as_the_same_run(tmp_path: Path) -> None:
    """`vera generate --config <run>/config.json` resolves to the recorded run."""
    parser = vera.build_parser()
    (original,) = generate.resolve_configs(
        parser.parse_args(
            ["generate", "-c", "gpt-5", "-u", "claude-sonnet-5", "--target", "SI"]
        )
    )
    root = create_run_root(tmp_path, "run_1", build_run_record(original))

    (rerun,) = generate.resolve_configs(
        parser.parse_args(["generate", "--config", str(root / RECORD_FILE)])
    )

    assert rerun == original


def test_judging_passes_a_record_of_the_judging_run(tmp_path: Path) -> None:
    run = tmp_path / "p_user__a_bot__t5__r1"
    (run / "conversations").mkdir(parents=True)
    (run / "conversations" / "c0.txt").write_text("transcript", encoding="utf-8")
    with patch.object(judge, "run_judging", new_callable=AsyncMock) as run_judging:
        vera.main(
            ["judge", "-j", "gpt-4o", "--conversations", str(run), "--target", "SI"]
        )

    record = run_judging.await_args.kwargs["run_record"]
    assert record["judging"]["models"] == [{"name": "gpt-4o", "repeats": 1}]
    assert "generation" not in record
    rubric = record["judging"]["rubrics"][0]["rubric_file"]
    assert record["inputs"][rubric] == file_checksum(rubric)


@pytest.mark.asyncio
async def test_generation_writes_the_record_with_the_folder_time(
    tmp_path: Path,
) -> None:
    """The record's start time and the folder name come from one clock read."""
    run_config = _run_config(tmp_path, user=[_model("claude-sonnet-5")])
    record = build_run_record(run_config)
    runner = AsyncMock()
    runner.run_conversations.return_value = []
    with patch.object(generation_domain, "ConversationRunner", return_value=runner):
        await generation_domain.run_for_user_models(
            run_config.generation,  # type: ignore[arg-type]
            max_personas=None,
            run_record=record,
        )

    (root,) = (tmp_path / "output").iterdir()
    written = json.loads((root / RECORD_FILE).read_text(encoding="utf-8"))
    started_at = datetime.fromisoformat(written["provenance"]["started_at"])
    assert root.name.endswith(started_at.strftime("%Y%m%d_%H%M%S"))
    assert written["generation"] == record["generation"]


def test_a_new_run_without_a_record_stops() -> None:
    """Every new run folder gets a config.json, so starting one needs a record."""
    with pytest.raises(ValueError, match="needs a run record"):
        require_run_record(None)
