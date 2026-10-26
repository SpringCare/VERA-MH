"""The user-side LLM's language: one prompt line, and none for English."""

from pathlib import Path

import pytest

from vera_mh.generate.utils import load_prompts_from_csv

REPO_ROOT = Path(__file__).resolve().parents[3]
PERSONAS = REPO_ROOT / "data/SI/personas.tsv"
CONTEXT = REPO_ROOT / "data/SI/persona_context_template.txt"
TEMPLATE = REPO_ROOT / "data/persona_prompt_template.txt"


def _first_prompt(**kwargs) -> str:
    rows = load_prompts_from_csv(
        prompt_path=str(PERSONAS),
        prompt_template_path=str(TEMPLATE),
        persona_context_template_path=str(CONTEXT),
        max_personas=1,
        **kwargs,
    )
    return rows[0]["prompt"]


@pytest.mark.unit
@pytest.mark.parametrize("english", ["English", "english", " English "])
def test_english_leaves_the_prompt_unchanged(english: str) -> None:
    assert _first_prompt(user_language=english) == _first_prompt()


@pytest.mark.unit
def test_other_language_adds_one_instruction_line() -> None:
    english = _first_prompt()
    spanish = _first_prompt(user_language="Spanish")

    assert spanish.startswith(english)
    assert spanish[len(english) :] == (
        "\n- Write every message in Spanish, as a native speaker would\n"
    )


@pytest.mark.unit
def test_a_regional_variant_is_passed_through_as_written() -> None:
    prompt = _first_prompt(user_language="Mexican Spanish")

    assert prompt.endswith(
        "\n- Write every message in Mexican Spanish, as a native speaker would\n"
    )
