#!/usr/bin/env python3
"""VERA-MH unified CLI orchestrator.

The CLI resolves either command-line flags or JSON config into one ``RunConfig``
and delegates to parser-independent domain functions. It deliberately contains
orchestration only; domain behavior stays in the existing modules.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from utils.config_schema import (
    GenerationConfig,
    JudgingConfig,
    ModelSpec,
    RubricSpec,
    RunConfig,
)

PROG = "vera"
ROOT = Path(__file__).resolve().parent
VERA_RUN_CONFIG_ENV = "VERA_RUN_CONFIG"


FLAG_SPECS: dict[str, dict[str, Any]] = {
    "chatbot": {
        "flags": ("-c", "--chatbot"),
        "kwargs": {
            "metavar": "<model>",
            "help": "Chatbot (provider/agent) LLM under test. No default.",
        },
    },
    "user": {
        "flags": ("-u", "--user"),
        "kwargs": {
            "nargs": "+",
            "metavar": "<model>[:<repeats>]",
            "help": "User-side LLM(s), e.g. `-u gpt:1 sonnet:2`.",
        },
    },
    "judge": {
        "flags": ("-j", "--judge"),
        "kwargs": {
            "nargs": "+",
            "metavar": "<model>[:<repeats>]",
            "help": "Judge LLM(s), e.g. `-j claude:1 gpt:2`.",
        },
    },
    "personas": {
        "flags": ("--personas",),
        "kwargs": {
            "nargs": "+",
            "metavar": "<file>",
            "help": "Persona file(s). Mutually exclusive with --target.",
        },
    },
    "target": {
        "flags": ("--target",),
        "kwargs": {
            "metavar": "<name>",
            "help": (
                "Select personas and rubric(s) from a named rubric bundle. "
                "Use `--target all` for every discovered bundle."
            ),
        },
    },
    "rubric": {
        "flags": ("--rubric",),
        "kwargs": {
            "nargs": "+",
            "metavar": "<manifest>",
            "help": "Rubric bundle manifest path(s).",
        },
    },
    "conversations": {
        "flags": ("--conversations",),
        "kwargs": {
            "nargs": "+",
            "metavar": "<folder>",
            "help": "Existing transcript folder(s) to judge.",
        },
    },
    "evaluations": {
        "flags": ("--evaluations",),
        "kwargs": {
            "nargs": "+",
            "metavar": "<folder>",
            "help": "Evaluation folder(s) to pool together.",
        },
    },
    "results": {
        "flags": ("-r", "--results"),
        "kwargs": {
            "metavar": "<results.csv>",
            "help": "Path to an existing results.csv to score.",
        },
    },
    "config": {
        "flags": ("--config",),
        "kwargs": {
            "metavar": "<path>",
            "help": (
                "JSON config file (or `-` for stdin). Cannot be combined with "
                f"run-defining CLI flags or {VERA_RUN_CONFIG_ENV}."
            ),
        },
    },
    "sample": {
        "flags": ("--sample",),
        "kwargs": {
            "type": int,
            "metavar": "N",
            "help": "Debug-only cap for personas, rubrics, and judges.",
        },
    },
    "debug": {
        "flags": ("-d", "--debug"),
        "kwargs": {
            "action": "store_true",
            "help": "Enable debug logging for a CLI-defined run.",
        },
    },
    "print": {
        "flags": ("--print",),
        "kwargs": {
            "action": "store_true",
            "help": "Print the resolved config and exit without running.",
        },
    },
}


class ConfigError(ValueError):
    """Raised for invalid or ambiguous CLI/config input."""


def add_flags(
    parser: argparse.ArgumentParser | argparse._MutuallyExclusiveGroup, *names: str
) -> None:
    """Attach centrally-defined flags to a parser."""
    for name in names:
        spec = FLAG_SPECS[name]
        parser.add_argument(*spec["flags"], **spec["kwargs"])


def parse_model_list(tokens: Optional[list[str]]) -> list[ModelSpec]:
    return [ModelSpec.from_shorthand(token) for token in tokens or []]


def parse_single_model(token: Optional[str]) -> Optional[ModelSpec]:
    return ModelSpec.from_shorthand(token) if token is not None else None


def _load_config_json(args: argparse.Namespace) -> Optional[dict[str, Any]]:
    config_arg = getattr(args, "config", None)
    env_config = os.environ.get(VERA_RUN_CONFIG_ENV)

    if config_arg and env_config:
        raise ConfigError(f"--config and {VERA_RUN_CONFIG_ENV} are mutually exclusive")
    try:
        if config_arg:
            if config_arg == "-":
                return json.loads(sys.stdin.read())
            with open(config_arg, encoding="utf-8") as config_file:
                return json.load(config_file)
        if env_config:
            return json.loads(env_config)
    except (json.JSONDecodeError, OSError) as error:
        raise ConfigError(f"could not load config: {error}") from error
    return None


def _cli_flags_given(args: argparse.Namespace, names: tuple[str, ...]) -> list[str]:
    return [
        name for name in names if getattr(args, name, None) not in (None, False, [])
    ]


def _root_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate.resolve())


def _resolve_config_paths(run_config: RunConfig) -> RunConfig:
    """Apply the architecture's $ROOT rule to config-sourced path fields."""
    if run_config.generation:
        run_config.generation.personas = [
            _root_path(path) for path in run_config.generation.personas
        ]
    if run_config.judging:
        run_config.judging.conversations = [
            _root_path(path) for path in run_config.judging.conversations
        ]
        for rubric in run_config.judging.rubrics:
            if rubric.name.lower() != "all" and (
                "/" in rubric.name or rubric.name.endswith(".json")
            ):
                rubric.name = _root_path(rubric.name)
    return run_config


def resolve_run_config(
    args: argparse.Namespace, cli_flag_names: tuple[str, ...]
) -> RunConfig:
    """Resolve exactly one run-definition source into a canonical config.

    ``--sample`` is AD-17's sole named exception and may accompany config.
    ``--debug`` and ``--print`` remain CLI-only controls.
    """
    config_json = _load_config_json(args)
    given_cli_flags = _cli_flags_given(args, (*cli_flag_names, "debug", "print"))
    config_sourced = config_json is not None

    if config_sourced and given_cli_flags:
        rendered = ", ".join(f"--{name.replace('_', '-')}" for name in given_cli_flags)
        raise ConfigError(
            f"config input cannot be combined with run-defining CLI flags: {rendered}"
        )
    if config_sourced:
        debug_fields = sorted({"sample", "debug", "print"}.intersection(config_json))
        if debug_fields:
            raise ConfigError(
                "debug/execution controls belong on the command line, not in config: "
                f"{', '.join(debug_fields)}"
            )

    try:
        run_config = (
            RunConfig.from_dict(config_json)
            if config_sourced
            else _run_config_from_cli(args)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"invalid run config: {error}") from error

    setattr(args, "_config_sourced", config_sourced)
    return _resolve_config_paths(run_config) if config_sourced else run_config


def _run_config_from_cli(args: argparse.Namespace) -> RunConfig:
    chatbot = parse_single_model(getattr(args, "chatbot", None))
    users = parse_model_list(getattr(args, "user", None))
    personas = list(getattr(args, "personas", None) or [])
    generation = None
    if chatbot or users or personas:
        generation = GenerationConfig(chatbot=chatbot, user=users, personas=personas)

    judge_models = parse_model_list(getattr(args, "judge", None))
    rubrics = [RubricSpec(name=path) for path in getattr(args, "rubric", None) or []]
    conversations = list(getattr(args, "conversations", None) or [])
    judging = None
    if judge_models or rubrics or conversations:
        judging = JudgingConfig(
            models=judge_models,
            rubrics=rubrics,
            conversations=conversations,
        )

    return RunConfig(
        generation=generation,
        judging=judging,
        target=getattr(args, "target", None),
    )


def print_resolved_config(run_config: RunConfig) -> None:
    print(json.dumps(run_config.to_dict(), indent=2))


def _validate_sample(sample: Optional[int]) -> None:
    if sample is not None and sample < 1:
        raise ConfigError("--sample must be at least 1")


def _manifest_catalog() -> list[Path]:
    return sorted((ROOT / "data").glob("**/rubric_manifest.json"))


def _resolve_named_manifest(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate.resolve()

    matches = [
        path
        for path in _manifest_catalog()
        if path.parent.name.casefold() == name.casefold()
        or path.stem.casefold() == name.casefold()
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise ConfigError(f"unknown rubric target or manifest: {name!r}")
    raise ConfigError(f"ambiguous rubric target {name!r}: {matches}")


def _target_manifests(target: Optional[str]) -> list[Path]:
    if target is None:
        return []
    if target.casefold() == "all":
        manifests = [path.resolve() for path in _manifest_catalog()]
        if not manifests:
            raise ConfigError("--target all found no rubric bundle manifests")
        return manifests
    return [_resolve_named_manifest(target)]


def _rubric_manifest(rubric: RubricSpec) -> str:
    return str(_resolve_named_manifest(rubric.name))


def _enable_debug(enabled: bool) -> None:
    if enabled:
        from utils.debug import set_debug

        set_debug(True)


def _model_config(spec: ModelSpec, *, chatbot: bool = False) -> dict[str, Any]:
    config = {"model": spec.name, **spec.extra_params}
    if chatbot:
        config["name"] = spec.name
    return config


async def _run_generation(
    run_config: RunConfig, *, sample: Optional[int], debug: bool
) -> list[str]:
    from generate_conversations import run_generation
    from utils.rubric_manifest import (
        load_manifest_persona_context_template,
        load_manifest_personas,
    )

    generation = run_config.generation
    if generation is None or generation.chatbot is None:
        raise ConfigError("generation configuration is missing")
    if generation.chatbot.repeats != 1:
        raise ConfigError("generation.chatbot repeats must be 1")

    _enable_debug(debug)
    persona_sources: list[list[str]] = []
    persona_context_templates: list[Optional[str]] = []
    manifests = _target_manifests(run_config.target)
    if manifests:
        for manifest in manifests:
            manifest_personas = await load_manifest_personas(str(manifest))
            if not manifest_personas:
                raise ConfigError(
                    f"Target {run_config.target!r} cannot generate conversations "
                    f"because its manifest {manifest} defines no personas. Add "
                    "personas to the manifest, or select personas and rubrics "
                    "independently."
                )
            persona_sources.append(manifest_personas)
            try:
                persona_context_templates.append(
                    await load_manifest_persona_context_template(str(manifest))
                )
            except ValueError:
                persona_context_templates.append(None)
    else:
        persona_sources.extend([persona] for persona in generation.personas)
        persona_context_templates.extend(None for _ in generation.personas)
    if sample is not None:
        persona_sources = persona_sources[:sample]
        persona_context_templates = persona_context_templates[:sample]

    output_folders: list[str] = []
    for user in generation.user:
        for persona_files, context_template in zip(
            persona_sources, persona_context_templates
        ):
            extra_kwargs: dict[str, Any] = {}
            if context_template is not None:
                extra_kwargs["persona_context_template_path"] = context_template
            _, output_folder = await run_generation(
                persona_model_config=_model_config(user),
                agent_model_config=_model_config(generation.chatbot, chatbot=True),
                persona_files=persona_files,
                persona_extra_run_params=dict(user.extra_params),
                agent_extra_run_params=dict(generation.chatbot.extra_params),
                runs_per_prompt=user.repeats,
                max_personas=sample,
                output_folder="output",
                **extra_kwargs,
            )
            output_folders.append(output_folder)
    return output_folders


def _group_models_by_params(
    models: list[ModelSpec],
) -> list[tuple[list[ModelSpec], dict[str, Any]]]:
    grouped: dict[str, tuple[list[ModelSpec], dict[str, Any]]] = {}
    for model in models:
        key = json.dumps(model.extra_params, sort_keys=True, default=str)
        if key not in grouped:
            grouped[key] = ([], dict(model.extra_params))
        grouped[key][0].append(model)
    return list(grouped.values())


async def _run_judging(
    run_config: RunConfig,
    *,
    sample: Optional[int],
    debug: bool,
    conversations: Optional[list[str]] = None,
) -> list[str]:
    judging = run_config.judging
    if judging is None:
        raise ConfigError("judging configuration is missing")

    _enable_debug(debug)
    from judge import run_judging

    conversation_folders = list(conversations or judging.conversations)
    rubrics = list(judging.rubrics)
    if run_config.target:
        rubrics = [
            RubricSpec(name=str(path)) for path in _target_manifests(run_config.target)
        ]
    if sample is not None:
        rubrics = rubrics[:sample]

    outputs: list[str] = []
    for conversation_folder in conversation_folders:
        for rubric in rubrics:
            models = list(rubric.models or judging.models)
            if sample is not None:
                models = models[:sample]
            for grouped_models, extra_params in _group_models_by_params(models):
                judge_models: dict[str, int] = {}
                for model in grouped_models:
                    repeats = min(model.repeats, sample) if sample else model.repeats
                    judge_models[model.name] = repeats
                output = await run_judging(
                    conversation_folder=conversation_folder,
                    rubric_manifest=_rubric_manifest(rubric),
                    judge_models=judge_models,
                    judge_model_extra_params=extra_params,
                    limit=sample,
                    max_concurrent=None,
                    per_judge=False,
                    verbose_workers=False,
                    debug=debug,
                )
                if output:
                    outputs.append(output)
    return outputs


def _run_scoring(results_csv: str) -> None:
    from judge.score import score_results_file

    status = score_results_file(
        results_csv,
        personas_tsv=str(ROOT / "data" / "SI" / "personas.tsv"),
    )
    if status:
        raise ConfigError(f"could not score results file: {results_csv}")


def _validate_generation(run_config: RunConfig) -> None:
    generation = run_config.generation
    if generation is None or generation.chatbot is None:
        raise ConfigError(
            "generate requires a chatbot (-c/--chatbot or generation.chatbot)"
        )
    if not generation.user:
        raise ConfigError(
            "generate requires at least one user model (-u/--user or generation.user)"
        )
    if not generation.personas and not run_config.target:
        raise ConfigError("generate requires --personas or --target")


def _models_for_rubric(judging: JudgingConfig, rubric: RubricSpec) -> list[ModelSpec]:
    return rubric.models or judging.models


def _validate_judging(
    run_config: RunConfig, *, require_conversations: bool = True
) -> None:
    judging = run_config.judging
    if judging is None:
        raise ConfigError("judge requires a judging configuration")
    rubrics = judging.rubrics or [
        RubricSpec(name=str(path)) for path in _target_manifests(run_config.target)
    ]
    if not rubrics:
        raise ConfigError("judge requires --rubric, judging.rubrics, or --target")
    if any(not _models_for_rubric(judging, rubric) for rubric in rubrics):
        raise ConfigError(
            "judge requires judge models (-j/--judge, judging.models, or rubric models)"
        )
    if require_conversations and not judging.conversations:
        raise ConfigError(
            "judge requires conversations from --conversations or judging.conversations"
        )


def cmd_generate(args: argparse.Namespace) -> int:
    run_config = resolve_run_config(
        args, cli_flag_names=("chatbot", "user", "personas", "target")
    )
    _validate_sample(args.sample)
    _validate_generation(run_config)
    print_resolved_config(run_config)
    if args.print:
        return 0
    asyncio.run(_run_generation(run_config, sample=args.sample, debug=args.debug))
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    run_config = resolve_run_config(
        args, cli_flag_names=("judge", "rubric", "conversations")
    )
    _validate_sample(args.sample)
    _validate_judging(run_config)
    print_resolved_config(run_config)
    if args.print:
        return 0
    asyncio.run(_run_judging(run_config, sample=args.sample, debug=args.debug))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    if not args.results:
        raise ConfigError("score requires -r/--results")
    if args.print:
        print(f"vera score -r {args.results}")
        return 0
    _run_scoring(args.results)
    return 0


def cmd_pool(args: argparse.Namespace) -> int:
    if not args.evaluations:
        raise ConfigError("pool requires --evaluations")
    if args.print:
        print(f"vera pool --evaluations {' '.join(args.evaluations)}")
        return 0

    from scripts.pool_vera_scores import pool_evaluation_directories

    pool_evaluation_directories(
        args.evaluations,
        ROOT / "output",
        personas_tsv=ROOT / "data" / "SI" / "personas.tsv",
    )
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    run_config = resolve_run_config(
        args,
        cli_flag_names=("chatbot", "user", "judge", "personas", "target", "rubric"),
    )
    _validate_sample(args.sample)
    _validate_generation(run_config)
    _validate_judging(run_config, require_conversations=False)
    print_resolved_config(run_config)
    if args.print:
        return 0

    async def run_pipeline() -> list[str]:
        generated = await _run_generation(
            run_config, sample=args.sample, debug=args.debug
        )
        return await _run_judging(
            run_config,
            sample=args.sample,
            debug=args.debug,
            conversations=generated,
        )

    evaluation_folders = asyncio.run(run_pipeline())
    for folder in evaluation_folders:
        _run_scoring(str(Path(folder) / "results.csv"))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    if not args.config:
        raise ConfigError("resume requires --config <run's own config.json>")
    if args.print:
        print(f"vera resume --config {args.config}")
        return 0
    raise ConfigError(
        "resume execution is deferred until the config checksum/state contract "
        "described in docs/architecture.md is implemented"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG, description="VERA-MH unified CLI orchestrator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_subcommand(
        name: str, help_text: str, handler: Callable[[argparse.Namespace], int]
    ) -> argparse.ArgumentParser:
        subcommand = subparsers.add_parser(name, help=help_text)
        subcommand.set_defaults(handler=handler)
        return subcommand

    generate = add_subcommand("generate", "Simulate conversations.", cmd_generate)
    add_flags(
        generate,
        "chatbot",
        "user",
        "personas",
        "target",
        "config",
        "sample",
        "debug",
        "print",
    )

    judge = add_subcommand(
        "judge", "Evaluate existing transcripts against a rubric.", cmd_judge
    )
    add_flags(
        judge,
        "judge",
        "rubric",
        "conversations",
        "config",
        "sample",
        "debug",
        "print",
    )

    score = add_subcommand(
        "score", "Aggregate results.csv into scores and visualizations.", cmd_score
    )
    add_flags(score, "results", "print")

    pool = add_subcommand(
        "pool",
        "Concatenate evaluation folders into one pooled result.",
        cmd_pool,
    )
    add_flags(pool, "evaluations", "print")

    pipeline = add_subcommand(
        "pipeline",
        "Full generate -> judge -> score workflow for one chatbot.",
        cmd_pipeline,
    )
    add_flags(
        pipeline,
        "chatbot",
        "user",
        "judge",
        "personas",
        "target",
        "rubric",
        "config",
        "sample",
        "debug",
        "print",
    )

    resume = add_subcommand(
        "resume",
        "Resume an incomplete run from config.json + state.json.",
        cmd_resume,
    )
    add_flags(resume, "config", "debug", "print")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ConfigError as error:
        parser.error(str(error))
        return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
