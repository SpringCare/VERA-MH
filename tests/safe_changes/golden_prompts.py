"""Golden prompts: what each persona's user model is told.

For the first few personas of each target in data/ (SI, HFO), the prompt is
built with generation's own code: the target's persona context template filled with the
persona's row, inserted into the shared data/persona_prompt_template.txt.

The shared template is the same for every persona and is its own source of
truth, so only the persona-specific part is committed, in
tests/fixtures/golden_prompts/<target>/<persona>.txt. The test rebuilds every
prompt, checks it is exactly the shared template around that part, and
requires the part to match.

Run this only for an intended change to personas or a persona context template,
then review the diff: it shows exactly what each persona will now be told.

    uv run python -m tests.safe_changes.golden_prompts
"""

import json
import shutil
from pathlib import Path

from vera_mh.generate.utils import load_prompts_from_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests/fixtures/golden_prompts"
SHARED_TEMPLATE = REPO_ROOT / "data/persona_prompt_template.txt"
TARGETS = ("SI", "HFO")
# Every persona goes through the same code, and the personas files are
# committed anyway, so a couple per target pin the code path.
PERSONAS_PER_TARGET = 2


def shared_template_around() -> tuple[str, str]:
    """The shared template's text before and after the persona's part."""
    marker = "\0PERSONA\0"
    rendered = SHARED_TEMPLATE.read_text(encoding="utf-8").format(
        persona_context=marker
    )
    before, after = rendered.split(marker)
    return before, after


def persona_part(prompt: str) -> str:
    """The persona-specific part of a full prompt, checking the rest is the template."""
    before, after = shared_template_around()
    if not (prompt.startswith(before) and prompt.endswith(after)):
        raise AssertionError("the prompt is not the shared template around a persona")
    return prompt[len(before) : len(prompt) - len(after)]


def build_prompts(target: str) -> dict[str, str]:
    """The first personas' full prompts for one target, keyed by persona name."""
    target_dir = REPO_ROOT / "data" / target
    manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
    prompts = {}
    for personas_file in manifest["personas"]:
        rows = load_prompts_from_csv(
            prompt_path=str(target_dir / personas_file),
            prompt_template_path=str(SHARED_TEMPLATE),
            persona_context_template_path=str(
                target_dir / manifest["persona_context_template_file"]
            ),
            max_personas=PERSONAS_PER_TARGET,
        )
        prompts.update({row["Name"]: row["prompt"] for row in rows})
    return prompts


def expected_parts(target: str) -> dict[str, str]:
    """The persona-specific parts committed for one target, keyed by persona name."""
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted((GOLDEN_DIR / target).glob("*.txt"))
    }


def write_all() -> None:
    """Overwrite the committed persona parts with what generation builds now."""
    for target in TARGETS:
        folder = GOLDEN_DIR / target
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        for name, prompt in build_prompts(target).items():
            (folder / f"{name}.txt").write_text(persona_part(prompt), encoding="utf-8")


if __name__ == "__main__":
    write_all()
