"""Tests for the ``vera judge`` command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import vera
from utils.config_schema import RubricFiles
from vera_cli import config as cli_config
from vera_cli import judge


@pytest.fixture(autouse=True)
def clear_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cli_config.VERA_RUN_CONFIG_ENV, raising=False)


def _generation_run(tmp_path: Path, *, conversations: int = 1) -> Path:
    """Create a folder shaped like a generation run, which judge derives output from."""
    run = tmp_path / "p_user__a_bot__t5__r1"
    transcripts = run / "conversations"
    transcripts.mkdir(parents=True)
    for index in range(conversations):
        (transcripts / f"c{index}.txt").write_text("transcript", encoding="utf-8")
    return run


def _flat_folder(tmp_path: Path) -> Path:
    """Create a legacy flat folder of transcripts, with no derivable run root."""
    folder = tmp_path / "flat"
    folder.mkdir(parents=True)
    (folder / "c.txt").write_text("transcript", encoding="utf-8")
    return folder


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _judging_config(tmp_path: Path, **overrides: object) -> dict:
    judging: dict[str, object] = {
        "models": [{"name": "gpt-4o", "repeats": 1}],
        "conversations": [str(_generation_run(tmp_path))],
        "rubrics": [
            {
                "rubric_file": "data/SI/rubric.tsv",
                "rubric_prompt_beginning_file": "data/SI/rubric_prompt_beginning.txt",
                "question_prompt_file": "data/SI/question_prompt.txt",
            }
        ],
        "output": "output",
        "max_concurrent": None,
        "per_judge": False,
    }
    judging.update(overrides)
    return {"judging": judging}


def test_judge_is_registered_and_has_help() -> None:
    parser = vera.build_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["judge", "--help"])
    assert exit_info.value.code == 0


def test_target_and_rubric_resolve_the_same_rubric(tmp_path: Path) -> None:
    """`--target` and `--rubric` differ in intent only, like generate's --personas."""
    run = _generation_run(tmp_path)
    parser = vera.build_parser()
    base = ["judge", "-j", "gpt-4o", "--conversations", str(run)]

    from_target = judge.resolve_configs(parser.parse_args([*base, "--target", "SI"]))
    from_rubric = judge.resolve_configs(parser.parse_args([*base, "--rubric", "SI"]))

    assert from_target == from_rubric


def test_output_defaults_beside_the_conversation_run(tmp_path: Path) -> None:
    run = _generation_run(tmp_path)
    args = vera.build_parser().parse_args(
        ["judge", "-j", "gpt-4o", "--conversations", str(run), "--target", "SI"]
    )

    judging = judge.resolve_configs(args)[0].judging
    assert judging is not None
    assert judging.output == str((run / "evaluations").resolve())


def test_flat_folder_requires_explicit_output(tmp_path: Path) -> None:
    """The legacy cwd-relative `evaluations/` fallback is gone; -o is required."""
    folder = _flat_folder(tmp_path)

    with pytest.raises(SystemExit) as error:
        vera.main(
            ["judge", "-j", "gpt-4o", "--conversations", str(folder), "--target", "SI"]
        )

    assert error.value.code == 2


def test_flat_folder_works_with_explicit_output(tmp_path: Path) -> None:
    folder = _flat_folder(tmp_path)
    destination = tmp_path / "evals"
    args = vera.build_parser().parse_args(
        [
            "judge",
            "-j",
            "gpt-4o",
            "--conversations",
            str(folder),
            "--target",
            "SI",
            "-o",
            str(destination),
        ]
    )

    judging = judge.resolve_configs(args)[0].judging
    assert judging is not None
    assert judging.output == str(destination.resolve())


def test_target_all_is_rejected(tmp_path: Path) -> None:
    """Deferred to Phase 4: N rubrics would share one output folder."""
    run = _generation_run(tmp_path)

    with pytest.raises(SystemExit) as error:
        vera.main(
            ["judge", "-j", "gpt-4o", "--conversations", str(run), "--target", "all"]
        )

    assert error.value.code == 2


def test_multiple_conversation_folders_are_rejected(tmp_path: Path) -> None:
    """Exactly one folder; judge separately and combine with vera pool."""
    first = _generation_run(tmp_path / "a")
    second = _generation_run(tmp_path / "b")

    with pytest.raises(SystemExit) as error:
        vera.main(
            [
                "judge",
                "-j",
                "gpt-4o",
                "--conversations",
                str(first),
                str(second),
                "--target",
                "SI",
            ]
        )

    assert error.value.code == 2


def test_judge_params_apply_to_every_model(tmp_path: Path) -> None:
    run = _generation_run(tmp_path)
    args = vera.build_parser().parse_args(
        [
            "judge",
            "-j",
            "gpt-4o:2",
            "claude:1",
            "--conversations",
            str(run),
            "--target",
            "SI",
            "--judge-params",
            "temperature=0,max_tokens=500",
        ]
    )

    judging = judge.resolve_configs(args)[0].judging
    assert judging is not None
    assert [model.extra_params for model in judging.models] == [
        {"temperature": 0, "max_tokens": 500},
        {"temperature": 0, "max_tokens": 500},
    ]


def test_repeated_model_name_is_rejected(tmp_path: Path) -> None:
    """The domain keys judge models by name, so duplicates would silently collapse."""
    run = _generation_run(tmp_path)

    with pytest.raises(SystemExit) as error:
        vera.main(
            [
                "judge",
                "-j",
                "gpt-4o",
                "gpt-4o",
                "--conversations",
                str(run),
                "--target",
                "SI",
            ]
        )

    assert error.value.code == 2


def test_per_model_judge_params_are_rejected(tmp_path: Path) -> None:
    """The domain takes one params dict, so differing per-model params must error."""
    config_data = _judging_config(
        tmp_path,
        models=[
            {"name": "gpt-4o", "repeats": 1, "temperature": 0},
            {"name": "claude", "repeats": 1, "temperature": 1},
        ],
    )

    with pytest.raises(SystemExit) as error:
        vera.main(["judge", "--config", str(_write_config(tmp_path, config_data))])

    assert error.value.code == 2


def test_judge_requires_target_or_rubric(tmp_path: Path) -> None:
    run = _generation_run(tmp_path)

    with pytest.raises(SystemExit) as error:
        vera.main(["judge", "-j", "gpt-4o", "--conversations", str(run)])

    assert error.value.code == 2


def test_judge_requires_conversations() -> None:
    with pytest.raises(SystemExit) as error:
        vera.main(["judge", "-j", "gpt-4o", "--target", "SI"])

    assert error.value.code == 2


def test_config_rejects_generation_section(tmp_path: Path) -> None:
    """`judge` rejects a generation block rather than silently ignoring it."""
    config_data = _judging_config(tmp_path)
    config_data["generation"] = {"chatbot": {"name": "x", "repeats": 1}}

    with pytest.raises(SystemExit) as error:
        vera.main(["judge", "--config", str(_write_config(tmp_path, config_data))])

    assert error.value.code == 2


def test_config_rejects_run_defining_cli_flag(tmp_path: Path) -> None:
    config = _write_config(tmp_path, _judging_config(tmp_path))

    with pytest.raises(SystemExit) as error:
        vera.main(["judge", "--config", str(config), "-j", "gpt-4o"])

    assert error.value.code == 2


def test_config_target_rejects_explicit_rubrics(tmp_path: Path) -> None:
    config_data = _judging_config(tmp_path)
    config_data["target"] = "SI"

    with pytest.raises(SystemExit) as error:
        vera.main(["judge", "--config", str(_write_config(tmp_path, config_data))])

    assert error.value.code == 2


def test_config_paths_resolve_from_repository_root(tmp_path: Path) -> None:
    config = _write_config(tmp_path, _judging_config(tmp_path))
    args = vera.build_parser().parse_args(["judge", "--config", str(config)])

    judging = judge.resolve_configs(args)[0].judging
    assert judging is not None
    assert judging.rubrics[0].rubric_file == str(
        (cli_config.ROOT / "data/SI/rubric.tsv").resolve()
    )
    assert judging.output == str((cli_config.ROOT / "output").resolve())


def test_every_rubric_file_is_resolved_not_just_the_first(tmp_path: Path) -> None:
    """`RubricFiles` is the resolved form, so all three fields must be absolute.

    Asserting only `rubric_file` (as the test above does) would pass even if the
    two prompt paths were handed through unresolved, which is exactly the shape
    the old two-step construction made possible.
    """
    config = _write_config(tmp_path, _judging_config(tmp_path))
    args = vera.build_parser().parse_args(["judge", "--config", str(config)])

    judging = judge.resolve_configs(args)[0].judging
    assert judging is not None
    rubric = judging.rubrics[0]
    assert [
        rubric.rubric_file,
        rubric.rubric_prompt_beginning_file,
        rubric.question_prompt_file,
    ] == [
        str((cli_config.ROOT / "data/SI/rubric.tsv").resolve()),
        str((cli_config.ROOT / "data/SI/rubric_prompt_beginning.txt").resolve()),
        str((cli_config.ROOT / "data/SI/question_prompt.txt").resolve()),
    ]


def test_rubric_files_cannot_be_built_straight_from_a_config_entry() -> None:
    """The resolved-by-construction invariant has no `from_dict` escape hatch.

    If one is reintroduced, raw repo-relative config strings can reach
    `RubricFiles` again and the type stops meaning what it says.
    """
    assert not hasattr(RubricFiles, "from_dict")


def test_resolved_run_omits_the_generation_section(tmp_path: Path) -> None:
    """A judging run must not emit `generation: null`.

    `to_dict` output doubles as input config, and `generate` rejects unknown
    top-level fields, so a null section would make `--print` emit something the
    other command refuses.
    """
    run = _generation_run(tmp_path)
    args = vera.build_parser().parse_args(
        ["judge", "-j", "gpt-4o", "--conversations", str(run), "--target", "SI"]
    )

    resolved = judge.resolve_configs(args)[0].to_dict()
    assert "generation" not in resolved
    assert set(resolved) == {"invocation", "judging"}


def test_print_round_trips_through_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--print` emits a config that reproduces the same resolved run."""
    run = _generation_run(tmp_path)
    parser = vera.build_parser()
    first = judge.resolve_configs(
        parser.parse_args(
            ["judge", "-j", "gpt-4o:2", "--conversations", str(run), "--target", "SI"]
        )
    )[0]

    monkeypatch.setenv(cli_config.VERA_RUN_CONFIG_ENV, json.dumps(first.to_dict()))
    replayed = judge.resolve_configs(parser.parse_args(["judge"]))

    assert replayed == [first]


def test_sample_caps_conversations_judged(tmp_path: Path) -> None:
    """`--sample` is the shared debug cap; for judge it limits conversations."""
    run = _generation_run(tmp_path, conversations=3)
    with patch.object(judge, "run_judging", new_callable=AsyncMock) as run_judging:
        vera.main(
            [
                "judge",
                "-j",
                "gpt-4o",
                "--conversations",
                str(run),
                "--target",
                "SI",
                "--sample",
                "2",
            ]
        )

    assert run_judging.await_count == 1
    assert run_judging.await_args is not None
    assert run_judging.await_args.kwargs["limit"] == 2


def test_execution_forwards_resolved_values(tmp_path: Path) -> None:
    run = _generation_run(tmp_path)
    with patch.object(judge, "run_judging", new_callable=AsyncMock) as run_judging:
        result = vera.main(
            [
                "judge",
                "-j",
                "gpt-4o:3",
                "--conversations",
                str(run),
                "--target",
                "SI",
                "--max-concurrent",
                "4",
                "--per-judge",
            ]
        )

    assert result == 0
    kwargs = run_judging.await_args.kwargs
    assert kwargs["judge_models"] == {"gpt-4o": 3}
    assert kwargs["max_concurrent"] == 4
    assert kwargs["per_judge"] is True
    assert kwargs["output_dir"] == str((run / "evaluations").resolve())
    # A parent to mint a new `j_*` run under, not an existing run to land in.
    assert kwargs["is_existing_run"] is False
    assert kwargs["resume"] is False
    assert kwargs["transcripts_dir"] == str(run / "conversations")
    assert kwargs["rubric_file"] == str(
        (cli_config.ROOT / "data/SI/rubric.tsv").resolve()
    )
