"""`persona_annotation_columns`: manifest parsing and judge CLI plumbing.

The column names are a statement of the *target*, not of judging code, so this
pins the path from a manifest entry to the resolved `JudgingConfig` judging acts
on, on both routes that reach a manifest.
"""

import json
from pathlib import Path

import pytest

from vera_cli.config import ConfigError
from vera_cli.targets import load_target


def _target(tmp_path: Path, **manifest_extra) -> Path:
    """Write a well-formed target bundle and return its manifest path."""
    target_dir = tmp_path / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "rubric.tsv",
        "rubric_prompt.txt",
        "question_prompt.txt",
        "context.txt",
    ):
        (target_dir / filename).write_text("fixture", encoding="utf-8")
    (target_dir / "personas.tsv").write_text(
        "Name\tActive Life Threat\nAAA1111\tPresent\n", encoding="utf-8"
    )
    manifest = target_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rubric_file": "rubric.tsv",
                "rubric_prompt_beginning_file": "rubric_prompt.txt",
                "question_prompt_file": "question_prompt.txt",
                "personas": ["personas.tsv"],
                "persona_context_template_file": "context.txt",
                **manifest_extra,
            }
        ),
        encoding="utf-8",
    )
    return manifest


class TestManifestParsing:
    def test_columns_are_read_from_the_manifest(self, tmp_path: Path) -> None:
        resolved = load_target(
            _target(tmp_path, persona_annotation_columns=["Active Life Threat"])
        )
        assert resolved.persona_annotation_columns == ["Active Life Threat"]

    def test_field_is_optional(self, tmp_path: Path) -> None:
        # Targets predating the field must keep loading and annotate nothing.
        assert load_target(_target(tmp_path)).persona_annotation_columns == []

    @pytest.mark.parametrize("bad", ["Active Life Threat", [""], [3], {}])
    def test_malformed_field_is_rejected(self, tmp_path: Path, bad: object) -> None:
        with pytest.raises(ConfigError, match="persona_annotation_columns"):
            load_target(_target(tmp_path, persona_annotation_columns=bad))


class TestShippedHfoTarget:
    def test_hfo_declares_active_life_threat(self) -> None:
        """The column exists in the personas file it will be joined against."""
        resolved = load_target(Path("data/HFO/manifest.json"))
        assert resolved.persona_annotation_columns == ["Active Life Threat"]

        import pandas as pd

        personas = pd.read_csv(resolved.personas[0], sep="\t", keep_default_na=False)
        assert "Active Life Threat" in personas.columns


class TestJudgeCliWiring:
    """Both manifest routes annotate, and config mode round-trips the fields."""

    @staticmethod
    def _resolved_judging(argv: list[str]) -> dict:
        import vera

        namespace = vera.build_parser().parse_args(argv)
        from vera_cli import judge as judge_command

        runs = judge_command.resolve_configs(namespace)
        assert len(runs) == 1
        return runs[0].judging.to_dict()

    def test_target_route_carries_personas_and_columns(self, tmp_path: Path) -> None:
        manifest = _target(tmp_path, persona_annotation_columns=["Active Life Threat"])
        conversations = tmp_path / "run" / "conversations"
        conversations.mkdir(parents=True)
        (conversations / "a_AAA1111_gpt-4o_run1.txt").write_text("user: hi\n")

        judging = self._resolved_judging(
            [
                "judge",
                "-j",
                "gpt-4o",
                "--conversations",
                str(conversations.parent),
                "--target",
                str(manifest),
            ]
        )
        assert judging["persona_annotation_columns"] == ["Active Life Threat"]
        assert judging["personas"] == [str(manifest.parent / "personas.tsv")]

    def test_rubric_route_annotates_identically(self, tmp_path: Path) -> None:
        """`--rubric` differs from `--target` in intent, not in resolved run.

        `test_target_and_rubric_resolve_the_same_rubric` pins that equality for
        the run as a whole; this states it for the annotation fields, so a later
        change cannot make one route annotate and the other not.
        """
        manifest = _target(tmp_path, persona_annotation_columns=["Active Life Threat"])
        conversations = tmp_path / "run" / "conversations"
        conversations.mkdir(parents=True)
        (conversations / "a_AAA1111_gpt-4o_run1.txt").write_text("user: hi\n")
        base = [
            "judge",
            "-j",
            "gpt-4o",
            "--conversations",
            str(conversations.parent),
        ]

        from_rubric = self._resolved_judging([*base, "--rubric", str(manifest)])
        from_target = self._resolved_judging([*base, "--target", str(manifest)])

        assert from_rubric["persona_annotation_columns"] == ["Active Life Threat"]
        assert from_rubric == from_target

    def test_empty_annotation_in_config_is_accepted(self, tmp_path: Path) -> None:
        """Absent and empty both mean "annotate nothing".

        `--print` emits the fields unconditionally, so a printed config for a
        target that declares no columns carries empty lists; refusing those
        would break the round trip for every pre-existing target.
        """
        import json

        import vera
        from vera_cli import config as cli_config
        from vera_cli import judge as judge_command

        manifest = _target(tmp_path)
        conversations = tmp_path / "run" / "conversations"
        conversations.mkdir(parents=True)
        (conversations / "a_AAA1111_gpt-4o_run1.txt").write_text("user: hi\n")
        parser = vera.build_parser()
        first = judge_command.resolve_configs(
            parser.parse_args(
                [
                    "judge",
                    "-j",
                    "gpt-4o",
                    "--conversations",
                    str(conversations.parent),
                    "--target",
                    str(manifest),
                ]
            )
        )[0]
        assert first.judging.persona_annotation_columns == []

        import os

        os.environ[cli_config.VERA_RUN_CONFIG_ENV] = json.dumps(first.to_dict())
        try:
            replayed = judge_command.resolve_configs(parser.parse_args(["judge"]))
        finally:
            del os.environ[cli_config.VERA_RUN_CONFIG_ENV]
        assert replayed == [first]
