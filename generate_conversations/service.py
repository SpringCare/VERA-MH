"""Application service for conversation generation.

This module owns generation orchestration independently of any command-line
parser. CLI entry points are adapters around :func:`run_generation`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.naming import (
    build_generation_run_folder_name,
    model_token_for_run_folder,
    parse_generation_run_folder_name,
)

from .runner import ConversationRunner


async def run_generation(
    persona_model_config: Dict[str, Any],
    agent_model_config: Dict[str, Any],
    persona_files: List[str],
    persona_extra_run_params: Optional[Dict[str, Any]] = None,
    agent_extra_run_params: Optional[Dict[str, Any]] = None,
    max_turns: int = 3,
    runs_per_prompt: int = 2,
    persona_names: Optional[List[str]] = None,
    verbose: bool = True,
    output_folder: Optional[str] = None,
    run_id: Optional[str] = None,
    max_concurrent: Optional[int] = None,
    max_total_words: Optional[int] = None,
    max_personas: Optional[int] = None,
    persona_speaks_first: bool = True,
    session_types: Optional[List[str]] = None,
    resume: bool = False,
    persona_context_template_path: str = "data/SI/persona_context_template.txt",
) -> tuple[List[Dict[str, Any]], str]:
    """Generate conversations from already-resolved persona file paths."""
    persona_extra_run_params = persona_extra_run_params or {}
    agent_extra_run_params = agent_extra_run_params or {}

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

    if output_folder is None:
        output_folder = "output"

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
        skipped_n = sum(1 for result in results if result.get("skipped"))
        ok_n = len(results) - skipped_n
        message = f"✅ Generated {ok_n} conversations → {output_folder}/"
        if skipped_n:
            message += f" ({skipped_n} skipped)"
        print(message)

    return results, output_folder
