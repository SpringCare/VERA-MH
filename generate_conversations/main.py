"""Main application entry point for one resolved generation invocation.

This module owns run-level setup and validation, then delegates conversation
execution to ``ConversationRunner``. CLI parsing, defaults, and target/config
resolution belong outside the generation domain.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.config_schema import GenerationConfig, ModelSpec
from utils.naming import (
    build_generation_run_folder_name,
    model_token_for_run_folder,
    parse_generation_run_folder_name,
)

from .runner import ConversationRunner


async def run_generation(
    *,
    persona_model_config: Dict[str, Any],
    agent_model_config: Dict[str, Any],
    persona_files: List[str],
    persona_extra_run_params: Dict[str, Any],
    agent_extra_run_params: Dict[str, Any],
    max_turns: int,
    runs_per_prompt: int,
    persona_names: Optional[List[str]],
    verbose: bool,
    output_folder: str,
    run_id: Optional[str],
    max_concurrent: Optional[int],
    max_total_words: Optional[int],
    max_personas: Optional[int],
    persona_speaks_first: bool,
    session_types: Optional[List[str]],
    resume: bool,
    persona_context_template_path: str,
) -> tuple[List[Dict[str, Any]], str]:
    """Generate conversations from already-resolved runtime values."""
    if verbose:
        print("🔄 Generating conversations with the following parameters:")
        print(f"  - Persona model: {persona_model_config}")
        print(f"  - Agent model: {agent_model_config}")
        print(f"  - Persona extra run params: {persona_extra_run_params}")
        print(f"  - Agent extra run params: {agent_extra_run_params}")
        print(f"  - Max turns: {max_turns}")
        print(f"  - Runs per prompt: {runs_per_prompt}")
        print(f"  - Persona names: {persona_names}")
        print(f"  - Output folder: {output_folder}")
        print(f"  - Run ID: {run_id}")
        print(f"  - Max concurrent: {max_concurrent}")
        print(f"  - Max total words: {max_total_words}")
        print(f"  - Max personas: {max_personas}")
        print(f"  - Persona speaks first: {persona_speaks_first}")
        print(f"  - Resume: {resume}")

    if not persona_files:
        raise ValueError("generation requires at least one persona file")
    if len(persona_files) > 1:
        print(
            f"Warning: multiple persona files passed ({persona_files}); "
            "multi-persona-file support is not yet implemented, using only "
            f"the first: {persona_files[0]}",
            file=sys.stderr,
        )
    persona_prompt_path = persona_files[0]

    if resume:
        if not os.path.isdir(output_folder):
            raise ValueError(
                "Resume mode requires --output to point to an existing run folder."
            )
        run_folder_name = os.path.basename(os.path.normpath(output_folder))
        run_meta = parse_generation_run_folder_name(run_folder_name)
        expected_persona = model_token_for_run_folder(persona_model_config["model"])
        expected_agent = model_token_for_run_folder(agent_model_config["model"])

        if run_meta["persona"] != expected_persona:
            raise ValueError(
                "Resume folder persona model does not match current --user-agent. "
                f"Expected p_{expected_persona}, got p_{run_meta['persona']}."
            )
        if run_meta["agent"] != expected_agent:
            raise ValueError(
                "Resume folder provider model does not match current --provider-agent. "
                f"Expected a_{expected_agent}, got a_{run_meta['agent']}."
            )
        if run_meta["turns"] != max_turns:
            raise ValueError(
                "Resume folder max turns does not match current --turns. "
                f"Expected t{max_turns}, got t{run_meta['turns']}."
            )
        if run_meta["runs"] != runs_per_prompt:
            raise ValueError(
                "Resume folder runs-per-prompt does not match current --runs. "
                f"Expected r{runs_per_prompt}, got r{run_meta['runs']}."
            )
        if run_id is None:
            run_id = run_folder_name
        elif run_id != run_folder_name:
            raise ValueError(
                "Resume mode requires --run-id to match the run folder name when set."
            )
    elif run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = build_generation_run_folder_name(
            persona_model_config["model"],
            agent_model_config["model"],
            max_turns,
            runs_per_prompt,
            timestamp,
        )
        output_folder = f"{output_folder}/{run_id}"
        os.makedirs(output_folder, exist_ok=True)

    runner = ConversationRunner(
        persona_model_config=persona_model_config,
        agent_model_config=agent_model_config,
        max_turns=max_turns,
        runs_per_prompt=runs_per_prompt,
        folder_name=output_folder,
        run_id=run_id,
        max_concurrent=max_concurrent,
        max_total_words=max_total_words,
        max_personas=max_personas,
        persona_speaks_first=persona_speaks_first,
        session_types=session_types,
        resume=resume,
        persona_prompt_path=persona_prompt_path,
        persona_context_template_path=persona_context_template_path,
    )
    results = await runner.run_conversations(persona_names=persona_names)

    if verbose:
        skipped = sum(1 for result in results if result.get("skipped"))
        message = (
            f"✅ Generated {len(results) - skipped} conversations → {output_folder}/"
        )
        if skipped:
            message += f" ({skipped} skipped)"
        print(message)

    return results, output_folder


# Moved here from the root `generate.py` when that script went to `legacy/`
# (v2.0.1): `vera generate` calls this, and `vera_cli` must not import from
# `legacy/`. It used to call the legacy `main()`, a pass-through with the same
# keyword arguments, so it now calls `run_generation` directly.
async def run_for_user_models(
    generation: GenerationConfig,
    *,
    max_personas: Optional[int],
    into: Optional[str] = None,
) -> List[str]:
    """Run one generation per user-side model, sequentially.

    STOPGAP. This exists so the CLI hands the generation domain a resolved
    `GenerationConfig` and lets the domain decide how a multi-model request
    expands into runs, instead of the CLI looping over models and flattening
    each one into the legacy dict shape itself.

    Deliberately thin: sequential, with no shared run identity across models.
    Sequential is not incidental — `max_concurrent` is applied *within* one
    generation, so running models concurrently would silently multiply the
    caller's concurrency cap against the same provider.

    The real fix is for the generation domain to accept these types natively,
    at which point `_legacy_model_config` below and this wrapper both
    disappear.

    Returns the run folder each user model wrote to, in the order the models
    were given. `run_generation` has always returned its output folder and
    this wrapper used to drop it; `vera pipeline` needs it, because chaining
    generation into judging means knowing where the transcripts landed
    without re-deriving it from the filesystem or scraping it out of a log.
    `vera generate` ignores the return.
    """
    run_folders: List[str] = []
    for user in generation.user:
        _, run_folder = await run_generation(
            persona_model_config=_legacy_model_config(user),
            # Only the agent side may carry `name`; see `_legacy_model_config`.
            agent_model_config=_legacy_model_config(generation.chatbot)
            | {"name": generation.chatbot.name},
            persona_files=list(generation.personas),
            persona_extra_run_params=dict(user.extra_params),
            agent_extra_run_params=dict(generation.chatbot.extra_params),
            max_turns=generation.turns,
            runs_per_prompt=user.repeats,
            persona_names=None,
            verbose=True,
            # `into` names an existing run folder to continue; without it,
            # `output` is a parent to mint a new run under.
            output_folder=into or generation.output,
            run_id=None,
            max_concurrent=generation.max_concurrent,
            max_total_words=generation.max_total_words,
            max_personas=max_personas,
            persona_speaks_first=generation.persona_speaks_first,
            session_types=generation.sessions,
            resume=into is not None,
            persona_context_template_path=generation.persona_context_template,
        )
        run_folders.append(run_folder)
    return run_folders


def _legacy_model_config(model: ModelSpec) -> Dict[str, Any]:
    """Flatten a `ModelSpec` into the dict shape `run_generation` expects.

    STOPGAP translation shim, deleted once the generation domain takes
    `ModelSpec` directly. `repeats` is dropped because it is passed separately
    as `runs_per_prompt`.

    Callers add `name` for the agent side only. That asymmetry is not this
    function's to decide, and it is not about output naming — run folders are
    built from `model` (`utils/naming.py:model_token_for_run_folder`). `name` is
    the provider's display name, defaulted to `"Provider"` in
    `generate_conversations/runner.py`. It is legal on the agent config only
    because the runner filters reserved keys out of that one before splatting
    it into `LLMFactory.create_llm`, while splatting the persona config raw —
    so a `name` key there would collide. Fix belongs in the runner.
    """
    return {**model.extra_params, "model": model.name}
