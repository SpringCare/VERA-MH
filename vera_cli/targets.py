"""Discovery and validation of complete target manifests.

A *target* is a reusable evaluation bundle — rubric, personas, and all
generation and judging prompts — living in `data/<name>/` and described by its
`manifest.json`. Naming a target is shorthand for the concrete file paths inside
it; this module turns that name into those paths and fails if any are missing.

`manifest.json` is the only manifest the unified CLI reads. The legacy scripts'
`rubric_manifest.json` is not consulted here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .config import ROOT, ConfigError, existing_file, path_from_root

# A manifest must describe a *complete* target: enough for both generation and
# judging. Partial bundles are rejected at resolution time rather than failing
# midway through a run that already cost money.
REQUIRED_FIELDS = (
    "rubric_file",
    "rubric_prompt_beginning_file",
    "question_prompt_file",
    "personas",
    "persona_context_template_file",
)


@dataclasses.dataclass(frozen=True)
class ResolvedTarget:
    """One validated target with every manifest path made absolute.

    `generate` uses only `personas` and `persona_context_template`; the rubric
    and prompt fields are validated here because a target is defined as
    complete, and are consumed once `vera judge` exists.
    """

    manifest: str
    rubric: str
    rubric_prompt_beginning: str
    question_prompt: str
    personas: list[str]
    persona_context_template: str


def target_catalog() -> list[Path]:
    """List every target manifest under `data/`, sorted for stable ordering.

    Backs both name lookup and `--target all`.
    """
    return sorted((ROOT / "data").glob("**/manifest.json"))


def resolve_target_manifest(selection: str) -> Path:
    """Resolve one target name or explicit manifest path to a manifest file.

    A path is tried first (as given, then relative to the repository root); if
    it is not a file, `selection` is matched case-insensitively against target
    directory names under `data/`. An ambiguous name is an error rather than a
    silent first-match.
    """
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
    """Resolve a target selection to manifests, expanding the `all` keyword.

    Returns a single-element list for a named target or path, and every
    discovered manifest for `all`. This is the one place a selection becomes
    plural, and therefore the only reason `generate` can produce more than one
    run configuration.
    """
    if selection.casefold() != "all":
        return [resolve_target_manifest(selection)]
    manifests = [manifest.resolve() for manifest in target_catalog()]
    if not manifests:
        raise ConfigError("--target all found no target manifests")
    return manifests


def load_target(manifest_path: Path) -> ResolvedTarget:
    """Validate one target manifest and resolve its paths against its own folder.

    Manifest paths are relative to the directory containing the manifest, so a
    target bundle can be copied or moved without editing it. Every referenced
    file must exist.
    """
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


def targets_from_config(
    *,
    config: dict[str, object],
    section: dict[str, object],
    explicit_fields: tuple[str, ...],
    section_name: str,
) -> list[ResolvedTarget] | None:
    """Resolve a config's top-level `target`, or report that it has none.

    Returns one `ResolvedTarget` per selected target, or `None` when the config
    states its components explicitly instead, in which case the caller reads
    those fields itself.

    This owns the two rules every command shares, so they cannot drift:

    - `target: "all"` selects every discovered target, and is the only input that
      yields more than one. Anything else yields exactly one.
    - A target and the explicit fields it would supply are mutually exclusive. A
      target already determines those values, so naming them too is a
      contradiction rather than an override.

    Projecting a `ResolvedTarget` onto the fields a command needs is left to the
    caller: `generate` takes personas and the context template, `judge` takes the
    rubric and its prompts. Those differ per command; the rules above do not.
    """
    target = config.get("target")
    if target is None:
        return None
    if not isinstance(target, str) or not target:
        raise ConfigError("target must be a non-empty string")
    overlap = set(explicit_fields).intersection(section)
    if overlap:
        raise ConfigError(
            f"target is mutually exclusive with explicit {section_name} fields: "
            f"{', '.join(sorted(overlap))}"
        )
    return [load_target(manifest) for manifest in target_manifest_paths(target)]


def config_dir(value: object, *, field: str) -> str:
    """Resolve one explicit config *directory* against the repository root.

    Sibling of `config_path`/`config_paths`, which verify files. A conversations
    folder is a directory, so it needs its own existence check — but it obeys the
    same rule, so it lives beside them rather than in the one command that
    happens to be the only caller today.
    """
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a path")
    resolved = Path(path_from_root(value))
    if not resolved.is_dir():
        raise ConfigError(f"{field} does not exist or is not a directory: {resolved}")
    return str(resolved)


def config_path(value: object, *, field: str) -> str:
    """Resolve one explicit config path against the repository root, verifying it.

    Shared by every command's explicit-component branch, so "a config path is
    repo-relative and must exist" is stated once.
    """
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a path")
    return existing_file(path_from_root(value), field=field)


def config_paths(value: object, *, field: str) -> list[str]:
    """Resolve a non-empty list of explicit config paths, verifying each."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ConfigError(f"{field} must be a non-empty list of paths")
    return [config_path(item, field=field) for item in value]
