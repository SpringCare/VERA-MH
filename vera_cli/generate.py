"""The ``vera generate`` command: flags, resolution, and the workflow call.

Read this module top-down. It is the reference implementation of the command
contract described in `vera_cli/README.md`:

1. `register` declares the flags and attaches `run` as the subparser's handler.
2. `run` is the entry point `vera.py` dispatches to.
3. `resolve_configs` picks one input form and produces canonical `RunConfig`s.
4. `_execute` hands each resolved run to the generation domain.

Nothing below step 3 reads an `argparse.Namespace`, and nothing above it touches
the generation domain.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from generate import run_for_user_models
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
    generation_persona_sets,
    load_target,
    resolve_target_manifest,
    target_manifest_paths,
)

# CLI behavior defaults. They live here, beside the flag definitions, rather than
# in the parser or the schema: the parser uses `argparse.SUPPRESS` so that flag
# *presence* stays detectable (see `register`), which means defaults cannot be
# parser defaults and are instead applied during resolution by `_value`.
#
# These defaults are CLI-only. A config-driven run states every behavior field
# explicitly, so no run silently inherits a value that is not written down
# somewhere the user controls.
DEFAULTS: dict[str, Any] = {
    "turns": 30,
    "output": "output",
    "max_concurrent": None,
    "max_total_words": None,
    "provider_speaks_first": False,
    "sessions": None,
}

# Flags that do not define the run: they change how one invocation executes or
# is presented, so they are the only ones allowed alongside `--config`.
#
# This is the only flag classification written down. Every other flag is
# run-defining by subtraction (`resolve_input` derives it from the parsed
# namespace), so a newly added flag is subject to the config-or-flags rule
# automatically rather than needing to be listed somewhere second.
INVOCATION_ONLY_FLAGS = frozenset({"config", "sample", "debug", "print_only"})

# Top-level config keys `generate` accepts. `judging` is deliberately absent:
# until `vera judge` exists there is nothing to do with it, and accepting a key
# this command ignores is worse than rejecting it.
ALLOWED_CONFIG_FIELDS = {"generation", "target"}


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``generate`` with the root parser.

    Every run-defining flag below sets ``default=argparse.SUPPRESS`` so the flag
    is absent from the parsed namespace unless the user actually passed it.
    That presence check is what lets `resolve_input` tell "the user chose this
    value" from "nobody chose anything", which in turn is what makes the
    config-or-flags rule enforceable and what keeps defaults out of the parser.
    `None` cannot serve as that sentinel because `None` is a meaningful value
    for `--max-concurrent`, `--max-total-words`, and `--sessions`.

    The invocation-only flags in `INVOCATION_ONLY_FLAGS` are exempt from this
    convention where the code reads them unconditionally: `--config` and
    `--print` need a real default so `args.config` and `args.print_only` always
    exist.
    """
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
        help=(
            "Complete target name or manifest path; use 'all' to run every "
            "target, which is the only input that produces more than one run"
        ),
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

    # `run` is this module's handler, defined immediately below. `vera.py`
    # dispatches to whatever a subparser records here.
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Resolve the requested run(s) and execute them.

    Resolution happens up front and completely, so an invalid target, missing
    persona file, or contradictory input fails before any model is called.
    """
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


def resolve_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Resolve either config JSON or CLI flags into canonical runs.

    Returns one `RunConfig` per selected target; only `--target all` (or
    `target: "all"` in config) selects more than one. Multiple `-u` models do
    *not* multiply this list — a `RunConfig` carries every user model, and the
    generation domain expands them into runs.
    """
    try:
        config, invocation = resolve_input(
            args,
            invocation_only_flags=INVOCATION_ONLY_FLAGS,
            allowed_config_fields=ALLOWED_CONFIG_FIELDS,
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


def _from_cli(
    args: argparse.Namespace, invocation: InvocationConfig
) -> list[RunConfig]:
    """Resolve CLI flags into canonical runs, applying CLI defaults.

    `--target` and `--personas` differ only in intent: `--target` names a whole
    bundle (and is what a later `vera pipeline` will reuse for judging too),
    while `--personas` names the generation component explicitly. Both resolve
    through the same manifest, so they produce identical generation inputs.
    """
    chatbot: str | None = getattr(args, "chatbot", None)
    users: list[str] | None = getattr(args, "user", None)
    target: str | None = getattr(args, "target", None)
    personas: str | None = getattr(args, "personas", None)

    # None of these can be enforced by the parser. `required=True` is impossible
    # because a config file may supply the same values instead, and the flags use
    # `SUPPRESS` so absence is indistinguishable from a default. The
    # target/personas group enforces "not both" but cannot require one.
    if not chatbot:
        raise ConfigError("generate requires -c/--chatbot")
    if not users:
        raise ConfigError("generate requires at least one -u/--user model")

    # Only `--target` honors the `all` keyword; `--personas` names one bundle.
    if target:
        manifests = target_manifest_paths(target)
    elif personas:
        manifests = [resolve_target_manifest(personas)]
    else:
        raise ConfigError("generate requires --target or --personas")

    resolved_targets = [load_target(manifest) for manifest in manifests]
    return [
        _run_config(
            invocation,
            chatbot=ModelSpec.from_shorthand(chatbot),
            users=[ModelSpec.from_shorthand(user) for user in users],
            personas=resolved.personas,
            persona_context_template=resolved.persona_context_template,
            turns=_value(args, "turns"),
            # CLI paths resolve against the working directory, unlike config
            # paths, which resolve against the repository root.
            output=str(Path(_value(args, "output")).resolve()),
            max_concurrent=_value(args, "max_concurrent"),
            max_total_words=_value(args, "max_total_words"),
            persona_speaks_first=not _value(args, "provider_speaks_first"),
            sessions=_sessions(_value(args, "sessions")),
        )
        for resolved in resolved_targets
    ]


def _from_config(
    config: dict[str, Any], invocation: InvocationConfig
) -> list[RunConfig]:
    """Resolve a config object into canonical runs.

    Every behavior field is required rather than defaulted: a stored config is
    meant to be a complete, reproducible description of a run, so a value the
    file does not state is an error instead of something this code fills in.
    """
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
        _run_config(
            invocation,
            chatbot=chatbot,
            users=users,
            personas=personas,
            persona_context_template=context,
            **behavior,
        )
        for personas, context in generation_persona_sets(config, generation)
    ]


def _value(args: argparse.Namespace, field: str) -> Any:
    """Read a run-defining flag, falling back to its CLI default.

    Necessary because `register` suppresses parser defaults so flag presence
    stays detectable.
    """
    return getattr(args, field, DEFAULTS[field])


def _sessions(value: str | None) -> list[str] | None:
    """Split the comma-separated `--sessions` value into an ordered list."""
    if value is None:
        return None
    sessions = [item.strip() for item in value.split(",") if item.strip()]
    if not sessions:
        raise ConfigError("--sessions must contain at least one non-empty name")
    return sessions


def _run_config(
    invocation: InvocationConfig,
    *,
    chatbot: ModelSpec,
    users: list[ModelSpec],
    personas: list[str],
    persona_context_template: str,
    turns: int,
    output: str,
    max_concurrent: int | None,
    max_total_words: int | None,
    persona_speaks_first: bool,
    sessions: list[str] | None,
) -> RunConfig:
    """Assemble and validate one canonical `RunConfig`.

    Behavior fields are named rather than forwarded as opaque keywords so this
    signature is a readable statement of what a run consists of, and so the
    `RunConfig`/`GenerationConfig` nesting is written once for both input paths.
    Type enforcement itself happens at runtime in `GenerationConfig.__post_init__`
    — the config path reaches here as `dict[str, Any]`, so a static checker
    cannot vouch for it.
    """
    return RunConfig(
        invocation=invocation,
        generation=GenerationConfig(
            chatbot=chatbot,
            user=users,
            personas=personas,
            persona_context_template=persona_context_template,
            turns=turns,
            output=output,
            max_concurrent=max_concurrent,
            max_total_words=max_total_words,
            persona_speaks_first=persona_speaks_first,
            sessions=sessions,
        ),
    )


async def _execute(run_configs: list[RunConfig]) -> None:
    """Hand each resolved run to the generation domain, one target at a time.

    Expanding a run's user models into individual generations belongs to the
    domain, not the CLI, so this passes the resolved `GenerationConfig` straight
    through to `generate.run_for_user_models` — a stopgap wrapper, see its
    docstring. Targets stay sequential here: `--target all` runs share chatbot,
    user models, turns, and repeats, and run folder names carry only
    second-granularity timestamps, so concurrent starts would collide.
    """
    for run_config in run_configs:
        await run_for_user_models(
            run_config.generation, max_personas=run_config.invocation.sample
        )
