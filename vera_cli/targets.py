"""Discovery and validation of complete target manifests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .config import ROOT, ConfigError, existing_file, path_from_root, required

REQUIRED_FIELDS = (
    "rubric_file",
    "rubric_prompt_beginning_file",
    "question_prompt_file",
    "personas",
    "persona_context_template_file",
)


@dataclasses.dataclass(frozen=True)
class ResolvedTarget:
    manifest: str
    rubric: str
    rubric_prompt_beginning: str
    question_prompt: str
    personas: list[str]
    persona_context_template: str


def target_catalog() -> list[Path]:
    return sorted((ROOT / "data").glob("**/manifest.json"))


def resolve_target_manifest(selection: str) -> Path:
    """Resolve a target name or explicit manifest path."""
    candidate = Path(selection)
    paths = [candidate] if candidate.is_absolute() else [candidate, ROOT / candidate]
    for path in paths:
        if path.is_file():
            return path.resolve()

    matches = [
        manifest
        for manifest in target_catalog()
        if manifest.parent.name.casefold() == selection.casefold()
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise ConfigError(f"unknown target or manifest: {selection!r}")
    raise ConfigError(f"ambiguous target {selection!r}: {matches}")


def target_manifest_paths(selection: str) -> list[Path]:
    if selection.casefold() != "all":
        return [resolve_target_manifest(selection)]
    manifests = [manifest.resolve() for manifest in target_catalog()]
    if not manifests:
        raise ConfigError("--target all found no target manifests")
    return manifests


def load_target(manifest_path: Path) -> ResolvedTarget:
    """Validate and resolve one complete target manifest."""
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(
            f"could not load target manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ConfigError(f"target manifest {manifest_path} must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in value]
    if missing:
        raise ConfigError(
            f"target manifest {manifest_path} is missing required field(s): "
            f"{', '.join(missing)}"
        )

    def resolve_file(field: str, path: object) -> str:
        if not isinstance(path, str) or not path:
            raise ConfigError(f"target manifest field {field} must be a path")
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        return existing_file(str(candidate), field=field)

    persona_values = value["personas"]
    if not isinstance(persona_values, list) or not persona_values:
        raise ConfigError("target manifest field personas must be a non-empty list")
    personas = [resolve_file("personas", persona) for persona in persona_values]

    return ResolvedTarget(
        manifest=str(manifest_path.resolve()),
        rubric=resolve_file("rubric_file", value["rubric_file"]),
        rubric_prompt_beginning=resolve_file(
            "rubric_prompt_beginning_file", value["rubric_prompt_beginning_file"]
        ),
        question_prompt=resolve_file(
            "question_prompt_file", value["question_prompt_file"]
        ),
        personas=personas,
        persona_context_template=resolve_file(
            "persona_context_template_file", value["persona_context_template_file"]
        ),
    )


def resolve_generation_personas(
    config: dict[str, object], generation: dict[str, object]
) -> list[tuple[list[str], str]]:
    """Resolve target-backed or explicit persona inputs for generation."""
    target = config.get("target")
    if target is not None:
        if not isinstance(target, str) or not target:
            raise ConfigError("target must be a non-empty string")
        overlap = {"personas", "persona_context_template"}.intersection(generation)
        if overlap:
            raise ConfigError(
                "target is mutually exclusive with explicit generation fields: "
                f"{', '.join(sorted(overlap))}"
            )
        judging = config.get("judging")
        if isinstance(judging, dict) and judging.get("rubrics"):
            raise ConfigError("target is mutually exclusive with judging.rubrics")
        return [
            (resolved.personas, resolved.persona_context_template)
            for resolved in (
                load_target(manifest) for manifest in target_manifest_paths(target)
            )
        ]

    personas = required(generation, "personas", section="generation config")
    context = required(
        generation, "persona_context_template", section="generation config"
    )
    if (
        not isinstance(personas, list)
        or not personas
        or not all(isinstance(persona, str) and persona for persona in personas)
    ):
        raise ConfigError("generation.personas must be a non-empty list of paths")
    if not isinstance(context, str) or not context:
        raise ConfigError("generation.persona_context_template must be a path")
    return [
        (
            [
                existing_file(path_from_root(persona), field="generation.personas")
                for persona in personas
            ],
            existing_file(
                path_from_root(context), field="generation.persona_context_template"
            ),
        )
    ]
