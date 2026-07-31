"""Shared rubric bundle manifest reading.

A rubric bundle manifest (docs/architecture.md#rubric-bundle-manifest)
attaches a rubric and the personas it's validated for as one unit. Both
`generate.py` (personas half) and `judge/rubric_config.py` (rubric half)
read the same manifest file -- this lives in `utils/` (the leaf layer)
rather than in `judge/` so `generate.py` never has to import a `judge/`
module to read it (`generate/`/`judge/` must never import each other, per
docs/architecture.md's Layer model).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiofiles

REQUIRED_KEYS = (
    "rubric_file",
    "rubric_prompt_beginning_file",
    "question_prompt_file",
)


async def load_manifest(manifest_path: str) -> dict[str, Any]:
    """Read and validate a rubric bundle manifest JSON file.

    Args:
        manifest_path: Path to the rubric bundle manifest JSON file.

    Returns:
        The parsed manifest dict.

    Raises:
        FileNotFoundError: If the manifest doesn't exist.
        ValueError: If the manifest is missing a required rubric key.
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Rubric bundle manifest not found: {manifest_file}")

    async with aiofiles.open(manifest_file, "r", encoding="utf-8") as f:
        manifest_obj = json.loads(await f.read())

    if not isinstance(manifest_obj, dict):
        raise ValueError(
            f"Rubric bundle manifest {manifest_file} must be a JSON object (dict)"
        )

    manifest = manifest_obj

    missing_keys = [key for key in REQUIRED_KEYS if key not in manifest]
    if missing_keys:
        raise ValueError(
            f"Rubric bundle manifest {manifest_file} is missing required "
            f"key(s): {missing_keys}"
        )

    return manifest


async def load_manifest_personas(manifest_path: str) -> list[str]:
    """Read a rubric bundle manifest's `personas` list.

    Used by `generate.py --rubric-manifest` (Phase 0's generation-side
    counterpart to `judge.py --rubrics`, see docs/architecture.md's Phase 0
    migration entry) to select personas from the same manifest that
    `judge.py` loads the rubric from. `personas` is optional in the
    manifest and defaults to an empty list.

    Entries resolve relative to the manifest's own folder (never `$ROOT` or
    the caller's working directory), per docs/architecture.md#rubric-bundle-manifest
    -- the same rule `rubric_file`/etc. already follow via `RubricConfig.load()`.
    """
    manifest = await load_manifest(manifest_path)
    manifest_dir = Path(manifest_path).parent
    return [str(manifest_dir / p) for p in manifest.get("personas", [])]


async def load_manifest_persona_context_template(manifest_path: str) -> str:
    """Resolve a manifest's persona context template relative to the manifest."""
    manifest = await load_manifest(manifest_path)
    context_template = manifest.get("persona_context_template_file")
    if not context_template:
        raise ValueError(
            f"Rubric bundle manifest {manifest_path} has no "
            "persona_context_template_file"
        )

    return str(Path(manifest_path).parent / context_template)
