#!/usr/bin/env python3

import argparse
import asyncio
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from generate_conversations import ConversationRunner
from llm_clients.llm_interface import DEFAULT_START_PROMPT
from utils.debug import set_debug
from utils.naming import (
    build_generation_run_folder_name,
    model_token_for_run_folder,
    parse_generation_run_folder_name,
)
from utils.utils import parse_key_value_list


async def main(
    persona_model_config: Dict[str, Any],
    agent_model_config: Dict[str, Any],
    persona_extra_run_params: Dict[str, Any] = {},
    agent_extra_run_params: Dict[str, Any] = {},
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
) -> tuple[List[Dict[str, Any]], str]:
    """
    Generate conversations and return results.

    Args:
        # TODO: should the extra config be separated?
        persona_model_config: Configuration dictionary for the persona model
        agent_model_config: Configuration dictionary for the agent model
        persona_extra_run_params: Extra parameters for the persona model
        agent_extra_run_params: Extra parameters for the agent model
        max_turns: Maximum turns per conversation
        runs_per_prompt: Number of runs per prompt
        persona_names: List of persona names to use. If None, uses all personas.
        verbose: Whether to print status messages
        output_folder: Parent directory for new runs (default ``output/``), or the
            existing ``p_*`` run folder when ``resume`` is True.
        max_total_words: Optional maximum total words across all responses
        max_concurrent: Maximum number of concurrent conversations. If None, runs all
            conversations concurrently.
        max_personas: Optional maximum number of personas to load from CSV. If None,
            loads all personas.
        persona_speaks_first: If True (default), persona speaks first; else provider
            speaks first. max_turns is adjusted so the provider always speaks last.

    Returns:
        List of conversation results

    Raises:
        ValueError: Configuration error
        Exception: Other errors
    """
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

    # Generate default folder name if not provided
    if output_folder is None:
        output_folder = "output"

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
        # TODO: do we want to give a message if the folder already exists?
        os.makedirs(output_folder, exist_ok=True)

    # Configuration
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
    )

    # Run conversations
    results = await runner.run_conversations(persona_names=persona_names)

    if verbose:
        skipped_n = sum(1 for r in results if r.get("skipped"))
        ok_n = len(results) - skipped_n
        msg = f"✅ Generated {ok_n} conversations → {output_folder}/"
        if skipped_n:
            msg += f" ({skipped_n} skipped)"
        print(msg)

    return results, output_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM conversations")

    parser.add_argument(
        "--user-agent",
        "-u",
        help=(
            "Model for the user-agent. Examples: claude-sonnet-4-5-20250929, "
            "gemini-1.5-pro, llama3:8b"
        ),
        required=True,
    )
    parser.add_argument(
        "--user-agent-extra-params",
        "-uep",
        help=(
            "Extra parameters for the user-agent. "
            "Examples: temperature=0.7, max_tokens=1000"
        ),
        type=parse_key_value_list,
        default={},
    )

    parser.add_argument(
        "--provider-agent",
        "-p",
        help=(
            "Model for the provider-agent. Examples: claude-sonnet-4-5-20250929, "
            "gemini-1.5-pro, llama3:8b"
        ),
        required=True,
    )

    parser.add_argument(
        "--provider-agent-extra-params",
        "-pep",
        help=(
            "Extra parameters for the provider-agent. "
            "Examples: temperature=0.7, max_tokens=1000"
        ),
        default={},
        type=parse_key_value_list,
    )

    parser.add_argument(
        "--runs",
        "-r",
        help="Number of runs per prompt",
        default=1,
        type=int,
        required=True,
    )

    parser.add_argument(
        "--turns",
        "-t",
        help="Number of turns per conversation",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max-total-words",
        "-w",
        help="Optional maximum total words across all responses in a conversation",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--run-id",
        "-i",
        help=(
            "Run ID for the conversations for this run. "
            "If not provided, a default will be generated."
        ),
        default=None,
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Parent directory where a new p_*__a_*__t*__r*__* run folder is created "
            "(default: output). With --resume, must be the existing run folder path."
        ),
    )

    parser.add_argument(
        "--resume",
        help=(
            "Resume a previous run from an existing run folder. "
            "Skips transcripts that already exist for persona/run pairs."
        ),
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--max-concurrent",
        "-c",
        help=(
            "Maximum number of concurrent conversations. "
            "Default is None (run all conversations concurrently)."
        ),
        default=None,
        type=int,
    )

    parser.add_argument(
        "--max-personas",
        "-mp",
        help="Maximum number of personas to use. Limits personas loaded from CSV.",
        default=None,
        type=int,
    )

    parser.add_argument(
        "-psf",
        "--provider-speaks-first",
        help="Provider agent speaks first; max_turns will be adjusted "
        "so provider has last turn. Default: persona speaks first.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-pfm",
        "--provider-first-message",
        help="Static first message from provider (no LLM call for first turn).",
        default=None,
    )

    parser.add_argument(
        "-psp",
        "--provider-start-prompt",
        help="Prompt sent to provider LLM when starting conversation (first turn).",
        default=DEFAULT_START_PROMPT,
    )

    parser.add_argument(
        "-usm",
        "--user-first-message",
        help="Static first message from user-agent (no LLM call for first turn).",
        default=None,
    )

    parser.add_argument(
        "-usp",
        "--user-start-prompt",
        help="Prompt sent to user-agent LLM when starting conversation (first turn).",
        default=DEFAULT_START_PROMPT,
    )

    def parse_sessions_arg(s: str) -> List[str]:
        sessions = [t.strip() for t in s.split(",") if t.strip()]
        if not sessions:
            raise argparse.ArgumentTypeError(
                "--sessions must contain at least one non-empty session name"
            )
        return sessions

    parser.add_argument(
        "--sessions",
        help=("Comma-separated sequence of session types to run in order "),
        type=parse_sessions_arg,
        default=None,
    )

    parser.add_argument(
        "--debug",
        "-d",
        help="Enable debug logging for conversation generation",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # Set debug mode if flag is provided
    if args.debug:
        set_debug(True)

    persona_model_config = {
        "model": args.user_agent,
        **args.user_agent_extra_params,
    }
    if args.user_first_message is not None:
        persona_model_config["first_message"] = args.user_first_message
    persona_model_config["start_prompt"] = args.user_start_prompt

    agent_model_config = {
        "model": args.provider_agent,
        # TODO: does provider need a name?
        # persona "name" (e.g., "Avery") is set later when creating conversations
        "name": args.provider_agent,
        **args.provider_agent_extra_params,
    }
    if args.provider_first_message is not None:
        agent_model_config["first_message"] = args.provider_first_message
    agent_model_config["start_prompt"] = args.provider_start_prompt

    # TODO: Do the run id here, so that it can be printed when starting
    results, output_folder = asyncio.run(
        main(
            persona_model_config=persona_model_config,
            agent_model_config=agent_model_config,
            max_turns=args.turns,
            runs_per_prompt=args.runs,
            persona_extra_run_params={
                k: v
                for k, v in persona_model_config.items()
                if k
                not in [
                    "model",
                    "model_name",
                    "name",
                    "temperature",
                    "max_tokens",
                    "top_p",
                ]
            },
            agent_extra_run_params={
                k: v
                for k, v in agent_model_config.items()
                if k
                not in [
                    "model",
                    "model_name",
                    "name",
                    "temperature",
                    "max_tokens",
                    "top_p",
                ]
            },
            output_folder=args.output or "output",
            max_concurrent=args.max_concurrent,
            max_total_words=args.max_total_words,
            max_personas=args.max_personas,
            persona_speaks_first=not args.provider_speaks_first,
            session_types=args.sessions,
            resume=args.resume,
        )
    )
    if results and all(r.get("skipped") for r in results):
        sys.exit(1)
