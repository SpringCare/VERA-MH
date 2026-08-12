"""The ``vera generate`` command: flags, resolution, and workflow call."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from generate import main as generate_conversations
from utils.config_schema import GenerationConfig, InvocationConfig, ModelSpec, RunConfig
from utils.debug import set_debug

from .config import (
    ConfigError,
    model_from_config,
    models_from_config,
    path_from_root,
    print_resolved_config,
    render_invocation,
    required,
    resolve_input,
)
from .targets import (
    load_target,
    resolve_generation_personas,
    resolve_target_manifest,
    target_manifest_paths,
)

DEFAULTS: dict[str, Any] = {
    "turns": 30,
    "output": "output",
    "max_concurrent": None,
    "max_total_words": None,
    "provider_speaks_first": False,
    "sessions": None,
}

RUN_FIELDS = (
    "chatbot",
    "user",
    "target",
    "personas",
    "turns",
    "output",
    "max_concurrent",
    "max_total_words",
    "provider_speaks_first",
    "sessions",
)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``generate`` with the root parser."""
    parser = subparsers.add_parser("generate", help="Simulate conversations")
    parser.add_argument(
        "-c",
        "--chatbot",
        default=argparse.SUPPRESS,
        help="Chatbot model under test (required with CLI-defined runs)",
    )
    parser.add_argument(
        "-u",
        "--user",
        nargs="+",
        metavar="<model>[:<repeats>]",
        default=argparse.SUPPRESS,
        help="User-side model(s) and full persona-set repeats",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--target",
        default=argparse.SUPPRESS,
        help="Complete target name or manifest path; use 'all' for every target",
    )
    target.add_argument(
        "--personas",
        default=argparse.SUPPRESS,
        help="Target name or manifest path whose personas and prompt should be used",
    )
    parser.add_argument(
        "-t",
        "--turns",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Maximum conversation turns (default: {DEFAULTS['turns']})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=argparse.SUPPRESS,
        help=f"Parent directory for generation runs (default: {DEFAULTS['output']})",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum concurrent conversations (default: unlimited)",
    )
    parser.add_argument(
        "--max-total-words",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum response words per conversation (default: unlimited)",
    )
    parser.add_argument(
        "--provider-speaks-first",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Have the chatbot speak first (default: persona speaks first)",
    )
    parser.add_argument(
        "--sessions",
        default=argparse.SUPPRESS,
        help="Comma-separated session types to run in order",
    )
    parser.add_argument("--config", help="JSON path or '-' for stdin")
    parser.add_argument(
        "--sample",
        type=int,
        default=argparse.SUPPRESS,
        help="Debug-only cap on personas loaded per file",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable debug logging",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print the resolved invocation without executing it",
    )
    parser.set_defaults(handler=run)


def _value(args: argparse.Namespace, field: str) -> Any:
    return getattr(args, field, DEFAULTS[field])


def _sessions(value: str | None) -> list[str] | None:
    if value is None:
        return None
    sessions = [item.strip() for item in value.split(",") if item.strip()]
    if not sessions:
        raise ConfigError("--sessions must contain at least one non-empty name")
    return sessions


def _run_config(
    invocation: InvocationConfig,
    chatbot: ModelSpec,
    users: list[ModelSpec],
    personas: list[str],
    context: str,
    **behavior: Any,
) -> RunConfig:
    return RunConfig(
        invocation=invocation,
        generation=GenerationConfig(
            chatbot=chatbot,
            user=users,
            personas=personas,
            persona_context_template=context,
            **behavior,
        ),
    )


def _from_cli(
    args: argparse.Namespace, invocation: InvocationConfig
) -> list[RunConfig]:
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
    assert isinstance(selection, str)
    manifests = (
        target_manifest_paths(selection)
        if target
        else [resolve_target_manifest(selection)]
    )
    resolved_targets = [load_target(manifest) for manifest in manifests]
    behavior = {
        "turns": _value(args, "turns"),
        "output": str(Path(_value(args, "output")).resolve()),
        "max_concurrent": _value(args, "max_concurrent"),
        "max_total_words": _value(args, "max_total_words"),
        "persona_speaks_first": not _value(args, "provider_speaks_first"),
        "sessions": _sessions(_value(args, "sessions")),
    }
    return [
        _run_config(
            invocation,
            ModelSpec.from_shorthand(chatbot),
            [ModelSpec.from_shorthand(user) for user in users],
            resolved.personas,
            resolved.persona_context_template,
            **behavior,
        )
        for resolved in resolved_targets
    ]


def _from_config(
    config: dict[str, Any], invocation: InvocationConfig
) -> list[RunConfig]:
    value = config.get("generation")
    if not isinstance(value, dict):
        raise ConfigError("generate requires a generation config object")
    generation = dict(value)
    chatbot = model_from_config(
        required(generation, "chatbot", section="generation config"),
        field="generation.chatbot",
    )
    users = models_from_config(
        required(generation, "user", section="generation config"),
        field="generation.user",
    )
    behavior = {
        field: required(generation, field, section="generation config")
        for field in (
            "turns",
            "output",
            "max_concurrent",
            "max_total_words",
            "persona_speaks_first",
            "sessions",
        )
    }
    if not isinstance(behavior["output"], str):
        raise ConfigError("generation.output must be a path string")
    behavior["output"] = path_from_root(behavior["output"])

    return [
        _run_config(invocation, chatbot, users, personas, context, **behavior)
        for personas, context in resolve_generation_personas(config, generation)
    ]


def resolve_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Resolve either config JSON or CLI flags, never a mixture."""
    try:
        config, invocation = resolve_input(
            args,
            run_fields=RUN_FIELDS,
            allowed_fields={"generation", "judging", "target"},
        )
        return (
            _from_config(config, invocation)
            if config is not None
            else _from_cli(args, invocation)
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid generation config: {error}") from error


def _model_config(model: ModelSpec, *, chatbot: bool) -> dict[str, Any]:
    config = {**model.extra_params, "model": model.name}
    if chatbot:
        config["name"] = model.name
    return config


async def _execute(run_configs: list[RunConfig]) -> None:
    for run_config in run_configs:
        generation = run_config.generation
        for user_model in generation.user:
            await generate_conversations(
                persona_model_config=_model_config(user_model, chatbot=False),
                agent_model_config=_model_config(generation.chatbot, chatbot=True),
                persona_files=list(generation.personas),
                persona_extra_run_params=dict(user_model.extra_params),
                agent_extra_run_params=dict(generation.chatbot.extra_params),
                max_turns=generation.turns,
                runs_per_prompt=user_model.repeats,
                persona_names=None,
                verbose=True,
                output_folder=generation.output,
                run_id=None,
                max_concurrent=generation.max_concurrent,
                max_total_words=generation.max_total_words,
                max_personas=run_config.invocation.sample,
                persona_speaks_first=generation.persona_speaks_first,
                session_types=generation.sessions,
                resume=False,
                persona_context_template_path=generation.persona_context_template,
            )


def run(args: argparse.Namespace) -> int:
    """Resolve and execute one or more generation configurations."""
    run_configs = resolve_configs(args)
    if args.print_only:
        for run_config in run_configs:
            print(render_invocation(run_config, command="generate"))
        return 0

    if any(config.invocation.debug for config in run_configs):
        set_debug(True)
    for run_config in run_configs:
        print_resolved_config(run_config)
    asyncio.run(_execute(run_configs))
    return 0
