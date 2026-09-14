"""Unit tests for the shared rubric bundle manifest reader."""

import json

import pytest

from utils.rubric_manifest import (
    load_manifest,
    load_manifest_persona_context_template,
    load_manifest_personas,
    load_manifest_rubric_paths,
)


def _write_bundle(tmp_path, **overrides) -> str:
    """Write a complete rubric bundle to `tmp_path` and return its manifest path.

    The three files are created for real, since `load_manifest_rubric_paths`
    checks that each field names an existing file. `overrides` replaces a
    field's value without creating a file for it, which is how the tests below
    express a typo or an unusable value.
    """
    fields = {
        "rubric_file": "rubric.tsv",
        "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
        "question_prompt_file": "question_prompt.txt",
    }
    for filename in fields.values():
        (tmp_path / filename).write_text("x", encoding="utf-8")

    fields.update(overrides)
    manifest_path = tmp_path / "rubric_manifest.json"
    manifest_path.write_text(json.dumps(fields), encoding="utf-8")
    return str(manifest_path)


@pytest.mark.unit
class TestLoadManifest:
    """Tests for load_manifest()."""

    async def test_load_manifest_success(self):
        manifest = await load_manifest("tests/fixtures/rubric_manifest_simple.json")
        assert manifest["rubric_file"] == "rubric_simple.tsv"

    async def test_load_manifest_missing_file(self):
        with pytest.raises(FileNotFoundError):
            await load_manifest("tests/fixtures/does_not_exist.json")

    async def test_load_manifest_missing_required_key(self, tmp_path):
        manifest_path = tmp_path / "incomplete_manifest.json"
        manifest_path.write_text(
            json.dumps({"rubric_file": "rubric_simple.tsv"}), encoding="utf-8"
        )

        with pytest.raises(ValueError):
            await load_manifest(str(manifest_path))


@pytest.mark.unit
class TestLoadManifestPersonas:
    """Tests for load_manifest_personas()."""

    async def test_load_manifest_personas_present(self, tmp_path):
        manifest_path = tmp_path / "manifest_with_personas.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "rubric_file": "rubric_simple.tsv",
                    "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
                    "question_prompt_file": "question_prompt.txt",
                    "personas": ["personas_a.tsv", "personas_b.tsv"],
                }
            ),
            encoding="utf-8",
        )

        personas = await load_manifest_personas(str(manifest_path))

        assert personas == [
            str(tmp_path / "personas_a.tsv"),
            str(tmp_path / "personas_b.tsv"),
        ]

    async def test_load_manifest_personas_absolute_entry_not_rejoined(self, tmp_path):
        manifest_path = tmp_path / "manifest_with_absolute_persona.json"
        absolute_personas_path = tmp_path / "elsewhere" / "personas.tsv"
        manifest_path.write_text(
            json.dumps(
                {
                    "rubric_file": "rubric_simple.tsv",
                    "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
                    "question_prompt_file": "question_prompt.txt",
                    "personas": [str(absolute_personas_path)],
                }
            ),
            encoding="utf-8",
        )

        personas = await load_manifest_personas(str(manifest_path))

        assert personas == [str(absolute_personas_path)]

    async def test_load_manifest_personas_defaults_to_empty(self):
        personas = await load_manifest_personas(
            "tests/fixtures/rubric_manifest_simple.json"
        )
        assert personas == []

    async def test_load_manifest_personas_missing_file(self):
        with pytest.raises(FileNotFoundError):
            await load_manifest_personas("tests/fixtures/does_not_exist.json")


@pytest.mark.unit
class TestLoadManifestRubricPaths:
    """Tests for load_manifest_rubric_paths()."""

    async def test_resolves_all_three_beside_manifest(self):
        paths = await load_manifest_rubric_paths(
            "tests/fixtures/rubric_manifest_simple.json"
        )

        assert paths == {
            "rubric_file": "tests/fixtures/rubric_simple.tsv",
            "rubric_prompt_beginning_file": (
                "tests/fixtures/rubric_prompt_beginning.txt"
            ),
            "question_prompt_file": "tests/fixtures/question_prompt.txt",
        }

    async def test_missing_file_names_the_offending_field(self, tmp_path):
        """A typo'd filename reports which field named it, not just the path."""
        manifest_path = _write_bundle(tmp_path, rubric_file="rubirc.tsv")

        with pytest.raises(FileNotFoundError, match="rubric_file") as error:
            await load_manifest_rubric_paths(manifest_path)

        # The manifest to edit and the value as written both appear, so the
        # message is actionable without opening the manifest.
        assert manifest_path in str(error.value)
        assert "rubirc.tsv" in str(error.value)

    async def test_empty_value_raises_rather_than_resolving_to_the_folder(
        self, tmp_path
    ):
        """`Path(dir) / "" == dir`, so an empty field must be caught up front."""
        manifest_path = _write_bundle(tmp_path, question_prompt_file="")

        with pytest.raises(ValueError, match="question_prompt_file"):
            await load_manifest_rubric_paths(manifest_path)

    async def test_non_string_value_raises_value_error(self, tmp_path):
        """A null field is a config error, not a `Path.__truediv__` TypeError."""
        manifest_path = _write_bundle(tmp_path, rubric_prompt_beginning_file=None)

        with pytest.raises(ValueError, match="rubric_prompt_beginning_file"):
            await load_manifest_rubric_paths(manifest_path)

    async def test_missing_key_still_reported_by_load_manifest(self, tmp_path):
        """Presence stays `load_manifest`'s job; no KeyError leaks out."""
        manifest_path = tmp_path / "incomplete_manifest.json"
        manifest_path.write_text(
            json.dumps({"rubric_file": "rubric.tsv"}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="question_prompt_file"):
            await load_manifest_rubric_paths(str(manifest_path))


@pytest.mark.unit
class TestLoadManifestPersonaContextTemplate:
    """Tests for load_manifest_persona_context_template()."""

    async def test_resolves_path_relative_to_manifest(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "rubric_file": "rubric.tsv",
                    "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
                    "question_prompt_file": "question_prompt.txt",
                    "persona_context_template_file": "persona_context_template.txt",
                }
            ),
            encoding="utf-8",
        )

        result = await load_manifest_persona_context_template(str(manifest_path))

        assert result == str(tmp_path / "persona_context_template.txt")

    async def test_missing_context_template_raises(self):
        with pytest.raises(ValueError, match="persona_context_template_file"):
            await load_manifest_persona_context_template(
                "tests/fixtures/rubric_manifest_simple.json"
            )
