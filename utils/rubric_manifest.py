"""Shared rubric bundle manifest reading.

Manifest personas are optional and informational for ordinary judging. Explicit
consumers such as ``vera --target`` and legacy ``generate.py --rubric-manifest``
may resolve them into generation inputs. This helper lives in the leaf ``utils``
layer so generation never imports judging code.
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
        manifest = json.loads(await f.read())

    missing_keys = [key for key in REQUIRED_KEYS if key not in manifest]
    if missing_keys:
        raise ValueError(
            f"Rubric bundle manifest {manifest_file} is missing required "
            f"key(s): {missing_keys}"
        )

    return manifest


async def load_manifest_personas(manifest_path: str) -> list[str]:
    """Read a rubric bundle manifest's `personas` list.

    Used by ``vera --target`` and the legacy ``generate.py --rubric-manifest``
    adapter. ``personas`` is optional in the manifest and defaults to an empty
    list; each caller decides whether its invocation requires personas.

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
