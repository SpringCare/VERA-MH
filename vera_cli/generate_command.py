"""Thin command adapter for ``vera generate``."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from generate import main as generate_conversations
from utils.config_schema import ModelSpec, RunConfig
from utils.debug import set_debug

from .config import ConfigError, print_resolved_config, render_invocation
from .generate_config import resolve_generate_configs


def _model_config(model: ModelSpec, *, chatbot: bool) -> dict[str, Any]:
    config = {**model.extra_params, "model": model.name}
    if chatbot:
        config["name"] = model.name
    return config


async def _run_configs(run_configs: list[RunConfig], *, sample: int | None) -> None:
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
                max_personas=sample,
                persona_speaks_first=generation.persona_speaks_first,
                session_types=generation.sessions,
                resume=False,
                persona_context_template_path=generation.persona_context_template,
            )


def run(args: argparse.Namespace) -> int:
    """Resolve and execute one or more generation configurations."""
    if args.sample is not None and args.sample < 1:
        raise ConfigError("--sample must be at least 1")
    run_configs = resolve_generate_configs(args)

    if args.print_only:
        for run_config in run_configs:
            print(
                render_invocation(
                    run_config,
                    command="generate",
                    sample=args.sample,
                    debug=args.debug,
                )
            )
        return 0

    if args.debug:
        set_debug(True)
    for run_config in run_configs:
        print_resolved_config(run_config)
    try:
        asyncio.run(_run_configs(run_configs, sample=args.sample))
    except ValueError as error:
        raise ConfigError(str(error)) from error
    return 0
