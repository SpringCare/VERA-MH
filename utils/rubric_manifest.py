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


def _resolve_manifest_file(
    manifest_path: str, manifest: dict[str, Any], key: str
) -> str:
    """Resolve one manifest file field against the manifest's own folder.

    Errors name the manifest, the field, and the value as written, because that
    triple is the only thing that tells a reader where to go fix a typo. By the
    time a path reaches `RubricConfig.from_paths` that provenance is gone -- it
    holds three bare strings -- so this is the only layer that can say it.

    Returns the path in the same relative-or-absolute shape the caller passed
    in, deliberately: callers compare these against the paths they supplied.
    """
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Rubric bundle manifest {manifest_path}: field {key!r} must be a "
            f"non-empty path, got {value!r}"
        )

    resolved = Path(manifest_path).parent / value
    # `is_file` rather than `exists` so an empty or directory-valued field is
    # rejected here instead of surfacing later as a missing *folder*.
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Rubric bundle manifest {manifest_path}: field {key!r} names "
            f"{value!r}, which is not a file beside the manifest: {resolved}"
        )
    return str(resolved)


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


async def load_manifest_rubric_paths(manifest_path: str) -> dict[str, str]:
    """Resolve a manifest's three rubric files relative to the manifest.

    A manifest names its files by bare filename, and those names mean "beside
    this manifest" -- never relative to `$ROOT` or the working directory. So
    given `judge.py --rubrics data/SI/rubric_manifest.json` containing::

        {"rubric_file": "rubric.tsv",
         "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
         "question_prompt_file": "question_prompt.txt"}

    this returns::

        {"rubric_file": "data/SI/rubric.tsv",
         "rubric_prompt_beginning_file": "data/SI/rubric_prompt_beginning.txt",
         "question_prompt_file": "data/SI/question_prompt.txt"}

    The keys match `RubricConfig.from_paths`' parameters, so a caller holding a
    manifest can go straight from one to the other. Iterating `REQUIRED_KEYS`
    rather than repeating the three names keeps the returned dict from drifting
    from the set `load_manifest` validates, since it is `**`-unpacked into that
    signature.

    Validation is split by what each layer can see: `load_manifest` rejects a
    *missing* key, `_resolve_manifest_file` rejects an unusable *value* and
    names the manifest field it came from, and `RubricConfig.from_paths` keeps
    its own existence checks for callers that arrive already holding resolved
    paths.

    This is the single place the manifest-relative rule is applied for rubric
    files, so `RubricConfig.load_bundle` and callers holding a bare manifest
    path (the legacy `judge.py --rubrics` form) cannot drift apart. Callers that
    already hold resolved paths -- `vera`'s target resolution, which validates
    them itself -- do not need this at all.
    """
    manifest = await load_manifest(manifest_path)
    return {
        key: _resolve_manifest_file(manifest_path, manifest, key)
        for key in REQUIRED_KEYS
    }


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
