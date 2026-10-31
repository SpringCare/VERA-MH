"""Main application entry point for one resolved generation invocation.

This module owns run-level setup and validation, then delegates conversation
execution to ``ConversationRunner``. CLI parsing, defaults, and target/config
resolution belong outside the generation domain.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, List, Optional

from vera_mh.utils.config_schema import GenerationConfig, ModelSpec
from vera_mh.utils.naming import (
    build_generation_run_folder_name,
    model_token_for_run_folder,
    parse_generation_run_folder_name,
)
from vera_mh.utils.role import Role
from vera_mh.utils.run_record import require_run_record, with_start_time
from vera_mh.utils.run_root import create_run_root, run_started_at

from .runner import ConversationRunner


async def run_generation(
    generation: GenerationConfig,
    *,
    max_personas: Optional[int],
    into: Optional[str] = None,
    run_record: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate conversations for each user-side model in turn.

    Returns the run folder each user model wrote to, in the order the models
    were given; `vera pipeline` judges them next.

    Sequential on purpose: `max_concurrent` caps the conversations *within* one
    user model's run, so running models concurrently would multiply that cap
    against the same provider.

    `into` names an existing run folder to continue; without it, a new run
    folder is created under `generation.output`. `run_record` describes the
    whole request; each new run folder gets a copy naming only its own user
    model, written as its `config.json` (see `utils/run_record.py`), so
    rerunning one folder's `config.json` repeats that folder and not its
    siblings. A continued run already has one.
    """
    if not generation.personas:
        raise ValueError("generation requires at least one persona file")
    if len(generation.personas) > 1:
        print(
            f"Warning: multiple persona files passed ({list(generation.personas)}); "
            "multi-persona-file support is not yet implemented, using only "
            f"the first: {generation.personas[0]}",
            file=sys.stderr,
        )

    run_folders: List[str] = []
    for user in generation.user:
        print("🔄 Generating conversations with the following parameters:")
        print(f"  - User model: {user.name} {dict(user.extra_params)}")
        print(f"  - Chatbot model: {generation.chatbot.name}")
        print(f"  - Max turns: {generation.turns}")
        print(f"  - Repeats: {user.repeats}")
        print(f"  - Max concurrent: {generation.max_concurrent}")
        print(f"  - Max total words: {generation.max_total_words}")
        print(f"  - Max personas: {max_personas}")
        print(f"  - First speaker: {generation.first_speaker}")
        print(f"  - User language: {generation.user_language}")
        print(f"  - Continuing: {into or 'no'}")

        if into is not None:
            _check_run_folder_matches(into, generation, user)
            run_folder = into
        else:
            run_folder = _create_run_folder(
                generation, user, _record_for_user(run_record, user)
            )

        runner = ConversationRunner(
            # The runner still takes each model as a dict of its settings, with
            # the model under "model"; it goes with the job loop. Only the
            # chatbot's carries `name`: the runner splats the user's dict
            # straight into `LLMFactory.create_llm`, where a `name` key would
            # collide with the persona name it passes.
            persona_model_config={**user.extra_params, "model": user.name},
            agent_model_config={
                **generation.chatbot.extra_params,
                "model": generation.chatbot.name,
                "name": generation.chatbot.name,
            },
            max_turns=generation.turns,
            runs_per_prompt=user.repeats,
            folder_name=run_folder,
            max_concurrent=generation.max_concurrent,
            max_total_words=generation.max_total_words,
            max_personas=max_personas,
            first_speaker=Role(generation.first_speaker),
            session_types=generation.sessions,
            resume=into is not None,
            persona_prompt_path=generation.personas[0],
            persona_context_template_path=generation.persona_context_template,
            user_language=generation.user_language,
        )
        results = await runner.run_conversations()

        skipped = sum(1 for result in results if result.get("skipped"))
        message = f"✅ Generated {len(results) - skipped} conversations → {run_folder}/"
        if skipped:
            message += f" ({skipped} skipped)"
        print(message)
        run_folders.append(run_folder)
    return run_folders


def _check_run_folder_matches(
    run_folder: str, generation: GenerationConfig, user: ModelSpec
) -> None:
    """Refuse to continue a run folder made with other models, turns, or repeats."""
    if not os.path.isdir(run_folder):
        raise ValueError(
            "Resume mode requires --output to point to an existing run folder."
        )
    run_meta = parse_generation_run_folder_name(
        os.path.basename(os.path.normpath(run_folder))
    )
    expected_persona = model_token_for_run_folder(user.name)
    expected_agent = model_token_for_run_folder(generation.chatbot.name)

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
    if run_meta["turns"] != generation.turns:
        raise ValueError(
            "Resume folder max turns does not match current --turns. "
            f"Expected t{generation.turns}, got t{run_meta['turns']}."
        )
    if run_meta["runs"] != user.repeats:
        raise ValueError(
            "Resume folder runs-per-prompt does not match current --runs. "
            f"Expected r{user.repeats}, got r{run_meta['runs']}."
        )


def _create_run_folder(
    generation: GenerationConfig,
    user: ModelSpec,
    run_record: Optional[Dict[str, Any]],
) -> str:
    """Create a new run folder for one user model, with its `config.json`."""
    started_at = run_started_at()
    name = build_generation_run_folder_name(
        user.name,
        generation.chatbot.name,
        generation.turns,
        user.repeats,
        started_at.strftime("%Y%m%d_%H%M%S"),
    )
    record = with_start_time(require_run_record(run_record), started_at)
    return str(create_run_root(generation.output, name, record))


def _record_for_user(
    run_record: Optional[Dict[str, Any]], user: ModelSpec
) -> Optional[Dict[str, Any]]:
    """The run record for the folder generated by one user model."""
    if run_record is None:
        return None
    record = copy.deepcopy(run_record)
    record["generation"]["user"] = [user.to_dict()]
    return record
