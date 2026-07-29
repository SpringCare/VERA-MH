"""Unit tests for the shared rubric bundle manifest reader."""

import json

import pytest

from utils.rubric_manifest import load_manifest, load_manifest_personas


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
