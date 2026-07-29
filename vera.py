#!/usr/bin/env python3
"""VERA-MH unified CLI orchestrator.

Per docs/architecture.md's "CLI surface" section, `vera.py` is the single
root-level orchestrator: subcommands parse arguments and delegate to domain
runners; they contain no business logic themselves.

Subcommands: generate, judge, score, pool, pipeline, resume -- see
docs/vera-cli-use-cases.md for the full CLI/config design this implements.

This is Phase 1 of the migration (docs/architecture.md#migration-from-current-layout):
argument parsing and config resolution land now; wiring into the existing
`generate`/`judge` engines is tracked separately and stubbed here for now.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Optional

from utils.config_schema import (
    GenerationConfig,
    JudgingConfig,
    ModelSpec,
    RubricSpec,
    RunConfig,
)

PROG = "vera"

VERA_RUN_CONFIG_ENV = "VERA_RUN_CONFIG"


# ---------------------------------------------------------------------------
# Centralized flag registry.
#
# Every CLI flag is defined exactly once here and referenced by name from
# whichever subcommand(s) need it, so `-c`/`-u`/`-j`/`--config`/etc. can never
# drift into slightly-different definitions across subcommands.
# ---------------------------------------------------------------------------

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
            "help": "User-side LLM(s), e.g. `-u gpt:1 sonnet:2`. `repeats` default 1.",
        },
    },
    "judge": {
        "flags": ("-j", "--judge"),
        "kwargs": {
            "nargs": "+",
            "metavar": "<model>[:<repeats>]",
            "help": "Judge LLM(s), e.g. `-j claude:1 gpt:2`. `repeats` defaults to 1.",
        },
    },
    "personas": {
        "flags": ("--personas",),
        "kwargs": {
            "nargs": "+",
            "metavar": "<file>",
            "help": "Persona file(s) for generation. Mutually exclusive with --target.",
        },
    },
    "target": {
        "flags": ("--target",),
        "kwargs": {
            "metavar": "<name>",
            "help": (
                "Resolve <name> to a rubric bundle manifest and set both "
                "generation personas and the judging rubric from it in one shot. "
                "Use `--target all` to run every known evaluator."
            ),
        },
    },
    "rubric": {
        "flags": ("--rubric",),
        "kwargs": {
            "nargs": "+",
            "metavar": "<manifest>",
            "help": "Rubric bundle manifest path(s) (see architecture.md).",
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
                "JSON config file (or `-` for stdin). Mutually exclusive with "
                f"CLI model/persona/rubric flags and the {VERA_RUN_CONFIG_ENV} env var."
            ),
        },
    },
    "sample": {
        "flags": ("--sample",),
        "kwargs": {
            "type": int,
            "metavar": "N",
            "help": "Smoke-test override: cap personas/rubrics/judges to N.",
        },
    },
    "print": {
        "flags": ("--print",),
        "kwargs": {
            "action": "store_true",
            "help": "Print the resolved flag-string and exit, without running.",
        },
    },
}


def add_flags(
    parser: argparse.ArgumentParser | argparse._MutuallyExclusiveGroup, *names: str
) -> None:
    """Attach flags from FLAG_SPECS to a parser or argument group by name."""
    for name in names:
        spec = FLAG_SPECS[name]
        parser.add_argument(*spec["flags"], **spec["kwargs"])


# ---------------------------------------------------------------------------
# Shared shorthand parsing.
# ---------------------------------------------------------------------------


def parse_model_list(tokens: Optional[list[str]]) -> list[ModelSpec]:
    if not tokens:
        return []
    return [ModelSpec.from_shorthand(t) for t in tokens]


def parse_single_model(token: Optional[str]) -> Optional[ModelSpec]:
    if token is None:
        return None
    return ModelSpec.from_shorthand(token)


# ---------------------------------------------------------------------------
# Config resolution: CLI flags and --config both funnel into one RunConfig.
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when CLI flags and --config/env conflict, or a field is missing."""


def _load_config_json(args: argparse.Namespace) -> Optional[dict[str, Any]]:
    """Load raw config JSON from --config or VERA_RUN_CONFIG, if given."""
    config_arg = getattr(args, "config", None)
    env_config = os.environ.get(VERA_RUN_CONFIG_ENV)

    if config_arg and env_config:
        raise ConfigError(f"--config and {VERA_RUN_CONFIG_ENV} are mutually exclusive")

    if config_arg:
        if config_arg == "-":
            return json.loads(sys.stdin.read())
        with open(config_arg) as f:
            return json.load(f)

    if env_config:
        return json.loads(env_config)

    return None


def _cli_flags_given(args: argparse.Namespace, names: tuple[str, ...]) -> list[str]:
    return [n for n in names if getattr(args, n, None)]


def resolve_run_config(
    args: argparse.Namespace, cli_flag_names: tuple[str, ...]
) -> RunConfig:
    """Resolve a subcommand's parsed args into one canonical RunConfig.

    CLI flags and --config/VERA_RUN_CONFIG are strictly either/or -- never
    combined for the same run (docs/vera-cli-use-cases.md#config-mechanism).
    """
    config_json = _load_config_json(args)
    given_cli_flags = _cli_flags_given(args, cli_flag_names)

    if config_json is not None:
        if given_cli_flags:
            raise ConfigError(
                "--config/"
                f"{VERA_RUN_CONFIG_ENV} cannot be combined with CLI flags: "
                f"{', '.join(given_cli_flags)}"
            )
        return RunConfig.from_dict(config_json)

    return _run_config_from_cli(args)


def _run_config_from_cli(args: argparse.Namespace) -> RunConfig:
    generation: Optional[GenerationConfig] = None
    chatbot = parse_single_model(getattr(args, "chatbot", None))
    user = parse_model_list(getattr(args, "user", None))
    personas = getattr(args, "personas", None) or []
    if chatbot is not None or user or personas:
        generation = GenerationConfig(chatbot=chatbot, user=user, personas=personas)

    judging: Optional[JudgingConfig] = None
    judge_models = parse_model_list(getattr(args, "judge", None))
    rubric_paths = getattr(args, "rubric", None) or []
    rubrics = [RubricSpec(name=r) for r in rubric_paths]
    if judge_models or rubrics:
        judging = JudgingConfig(models=judge_models, rubrics=rubrics)

    return RunConfig(
        generation=generation,
        judging=judging,
        target=getattr(args, "target", None),
        sample=getattr(args, "sample", None),
    )


def print_resolved_config(run_config: RunConfig) -> None:
    print(json.dumps(run_config.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Subcommand handlers.
#
# Phase 1 only replaces the front end (docs/architecture.md's migration
# table) -- these delegate to the existing generate/judge engines in a
# follow-up change. For now they resolve + print the config and stop.
# ---------------------------------------------------------------------------


def _not_yet_wired(command: str) -> None:
    print(
        f"vera {command}: argument parsing and config resolution only for now -- "
        "wiring into the existing generate/judge engine lands in a follow-up "
        "change (docs/architecture.md, migration Phase 1).",
        file=sys.stderr,
    )


def cmd_generate(args: argparse.Namespace) -> int:
    run_config = resolve_run_config(
        args, cli_flag_names=("chatbot", "user", "personas", "target")
    )
    if run_config.generation is None or run_config.generation.chatbot is None:
        raise ConfigError(
            "generate requires a chatbot (-c/--chatbot or generation.chatbot)"
        )
    if not run_config.generation.user:
        raise ConfigError(
            "generate requires at least one user model (-u/--user or generation.user)"
        )
    if not run_config.generation.personas and not run_config.target:
        raise ConfigError("generate requires --personas or --target")
    print_resolved_config(run_config)
    if args.print:
        return 0
    _not_yet_wired("generate")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    run_config = resolve_run_config(args, cli_flag_names=("judge", "rubric"))
    if run_config.judging is None or not run_config.judging.models:
        raise ConfigError(
            "judge requires at least one judge model (-j/--judge or judging.models)"
        )
    if not run_config.judging.rubrics:
        raise ConfigError("judge requires --rubric or judging.rubrics")
    if not args.conversations:
        raise ConfigError("judge requires --conversations")
    print_resolved_config(run_config)
    if args.print:
        return 0
    _not_yet_wired("judge")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    if not args.results:
        raise ConfigError("score requires -r/--results")
    if args.print:
        print(f"vera score -r {args.results}")
        return 0
    _not_yet_wired("score")
    return 0


def cmd_pool(args: argparse.Namespace) -> int:
    if not args.evaluations:
        raise ConfigError("pool requires --evaluations")
    if args.print:
        print(f"vera pool --evaluations {' '.join(args.evaluations)}")
        return 0
    _not_yet_wired("pool")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    run_config = resolve_run_config(
        args,
        cli_flag_names=("chatbot", "user", "judge", "personas", "target", "rubric"),
    )
    if run_config.target is None:
        if run_config.generation is None or run_config.generation.chatbot is None:
            raise ConfigError(
                "pipeline requires a chatbot (-c/--chatbot or generation.chatbot)"
            )
        if not run_config.generation.user:
            raise ConfigError(
                "pipeline requires at least one user model (-u/--user or "
                "generation.user)"
            )
        if run_config.judging is None or not run_config.judging.models:
            raise ConfigError(
                "pipeline requires at least one judge model (-j/--judge or "
                "judging.models)"
            )
    print_resolved_config(run_config)
    if args.print:
        return 0
    _not_yet_wired("pipeline")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    if not args.config:
        raise ConfigError("resume requires --config <run's own config.json>")
    if args.print:
        print(f"vera resume --config {args.config}")
        return 0
    _not_yet_wired("resume")
    return 0


# ---------------------------------------------------------------------------
# Parser construction.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG, description="VERA-MH unified CLI orchestrator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_subcommand(
        name: str, help_text: str, handler: Callable[[argparse.Namespace], int]
    ) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        return sub

    generate = add_subcommand("generate", "Simulate conversations.", cmd_generate)
    add_flags(
        generate, "chatbot", "user", "personas", "target", "config", "sample", "print"
    )

    judge = add_subcommand(
        "judge", "Evaluate existing transcripts against a rubric.", cmd_judge
    )
    add_flags(judge, "judge", "rubric", "conversations", "config", "sample", "print")

    score = add_subcommand(
        "score", "Aggregate results.csv into scores and visualizations.", cmd_score
    )
    add_flags(score, "results", "print")

    pool = add_subcommand(
        "pool",
        "Concatenate multiple evaluation folders into one pooled result.",
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
        "print",
    )

    resume = add_subcommand(
        "resume",
        "Resume an incomplete run from its config.json + state.json.",
        cmd_resume,
    )
    add_flags(resume, "config", "print")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ConfigError as e:
        parser.error(str(e))
        return 2  # pragma: no cover - argparse.error() exits before this


if __name__ == "__main__":
    sys.exit(main())
