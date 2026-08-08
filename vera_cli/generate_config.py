"""Resolve ``vera generate`` inputs into canonical run configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils.config_schema import GenerationConfig, ModelSpec, RunConfig

from .config import ConfigError, existing_file, load_config, path_from_root
from .generate_arguments import cli_value, explicit_fields, parse_sessions
from .targets import load_target, resolve_target_manifest, target_manifest_paths

BEHAVIOR_FIELDS = (
    "turns",
    "output",
    "max_concurrent",
    "max_total_words",
    "persona_speaks_first",
    "sessions",
)


def _required(data: dict[str, Any], field: str) -> Any:
    if field not in data:
        raise ConfigError(f"generation config is missing required field: {field}")
    return data[field]


def _model(token: str) -> ModelSpec:
    try:
        return ModelSpec.from_shorthand(token)
    except ValueError as error:
        raise ConfigError(str(error)) from error


def _models(tokens: list[str]) -> list[ModelSpec]:
    return [_model(token) for token in tokens]


def _run_config(
    *,
    chatbot: ModelSpec,
    users: list[ModelSpec],
    personas: list[str],
    context_template: str,
    behavior: dict[str, Any],
) -> RunConfig:
    try:
        generation = GenerationConfig(
            chatbot=chatbot,
            user=users,
            personas=personas,
            turns=behavior["turns"],
            output=behavior["output"],
            max_concurrent=behavior["max_concurrent"],
            max_total_words=behavior["max_total_words"],
            persona_speaks_first=behavior["persona_speaks_first"],
            sessions=behavior["sessions"],
            persona_context_template=context_template,
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid generation config: {error}") from error
    return RunConfig(generation=generation)


def _from_cli(args: argparse.Namespace) -> list[RunConfig]:
    chatbot = getattr(args, "chatbot", None)
    users = getattr(args, "user", None)
    target = getattr(args, "target", None)
    personas = getattr(args, "personas", None)
    if not chatbot:
        raise ConfigError("generate requires -c/--chatbot")
    if not users:
        raise ConfigError("generate requires at least one -u/--user model")
    if not target and not personas:
        raise ConfigError("generate requires --target or --personas")

    selection = target or personas
    assert selection is not None
    manifests = (
        target_manifest_paths(selection)
        if target
        else [resolve_target_manifest(selection)]
    )
    behavior = {
        "turns": cli_value(args, "turns"),
        "output": str(Path(cli_value(args, "output")).resolve()),
        "max_concurrent": cli_value(args, "max_concurrent"),
        "max_total_words": cli_value(args, "max_total_words"),
        "persona_speaks_first": not cli_value(args, "provider_speaks_first"),
        "sessions": parse_sessions(cli_value(args, "sessions")),
    }
    return [
        _run_config(
            chatbot=_model(chatbot),
            users=_models(users),
            personas=resolved.personas,
            context_template=resolved.persona_context_template,
            behavior=behavior,
        )
        for resolved in (load_target(manifest) for manifest in manifests)
    ]


def _config_models(generation: dict[str, Any]) -> tuple[ModelSpec, list[ModelSpec]]:
    chatbot = _required(generation, "chatbot")
    users = _required(generation, "user")
    if not isinstance(chatbot, dict):
        raise ConfigError("generation.chatbot must be an object")
    if not isinstance(users, list) or not all(isinstance(user, dict) for user in users):
        raise ConfigError("generation.user must be a list of objects")
    try:
        return ModelSpec.from_dict(chatbot), [
            ModelSpec.from_dict(user) for user in users
        ]
    except ValueError as error:
        raise ConfigError(f"invalid generation model: {error}") from error


def _target_inputs(
    config: dict[str, Any], generation: dict[str, Any]
) -> list[tuple[list[str], str]]:
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

    personas = _required(generation, "personas")
    context = _required(generation, "persona_context_template")
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


def _from_json(config: dict[str, Any]) -> list[RunConfig]:
    controls = sorted({"sample", "debug", "print"}.intersection(config))
    if controls:
        raise ConfigError(
            "debug/execution controls belong on the command line, not in config: "
            f"{', '.join(controls)}"
        )
    unknown = set(config).difference({"generation", "judging", "target"})
    if unknown:
        raise ConfigError(
            f"unknown top-level config field(s): {', '.join(sorted(unknown))}"
        )

    value = config.get("generation")
    if not isinstance(value, dict):
        raise ConfigError("generate requires a generation config object")
    generation = dict(value)
    chatbot, users = _config_models(generation)
    behavior = {field: _required(generation, field) for field in BEHAVIOR_FIELDS}
    if behavior["sessions"] is not None and (
        not isinstance(behavior["sessions"], list)
        or not all(isinstance(session, str) for session in behavior["sessions"])
    ):
        raise ConfigError("generation.sessions must be null or a list of strings")
    if not isinstance(behavior["output"], str):
        raise ConfigError("generation.output must be a path string")
    behavior["output"] = path_from_root(behavior["output"])

    return [
        _run_config(
            chatbot=chatbot,
            users=users,
            personas=personas,
            context_template=context,
            behavior=behavior,
        )
        for personas, context in _target_inputs(config, generation)
    ]


def resolve_generate_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Resolve either config JSON or CLI flags, never a mixture."""
    config = load_config(getattr(args, "config", None))
    supplied = explicit_fields(args)
    if config is not None and supplied:
        flags = ", ".join(f"--{field.replace('_', '-')}" for field in supplied)
        raise ConfigError(
            f"config input cannot be combined with run-defining CLI flags: {flags}"
        )
    return _from_json(config) if config is not None else _from_cli(args)
