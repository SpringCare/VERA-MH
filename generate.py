#!/usr/bin/env python3

import argparse
import asyncio
import sys
from typing import Any, Dict, List, Optional

from generate_conversations import run_generation
from llm_clients.llm_interface import DEFAULT_START_PROMPT
from utils.debug import set_debug
from utils.rubric_manifest import (
    load_manifest_persona_context_template,
    load_manifest_personas,
)
from utils.utils import parse_key_value_list


async def main(
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
    """Generate conversations from fully resolved inputs."""
    return await run_generation(
        persona_model_config=persona_model_config,
        agent_model_config=agent_model_config,
        persona_files=persona_files,
        persona_extra_run_params=persona_extra_run_params,
        agent_extra_run_params=agent_extra_run_params,
        max_turns=max_turns,
        runs_per_prompt=runs_per_prompt,
        persona_names=persona_names,
        verbose=verbose,
        output_folder=output_folder,
        run_id=run_id,
        max_concurrent=max_concurrent,
        max_total_words=max_total_words,
        max_personas=max_personas,
        persona_speaks_first=persona_speaks_first,
        session_types=session_types,
        resume=resume,
        persona_context_template_path=persona_context_template_path,
    )


async def resolve_persona_inputs(manifest: str) -> tuple[List[str], str]:
    """Resolve generation inputs for legacy callers of ``generate.py``."""
    persona_files = await load_manifest_personas(manifest)
    if not persona_files:
        raise ValueError(f"Rubric bundle manifest {manifest} has no personas listed")
    context_template = await load_manifest_persona_context_template(manifest)
    return persona_files, context_template


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
        default="output",
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

    parser.add_argument(
        "--rubric-manifest",
        help=(
            "Rubric bundle manifest to load personas from (see "
            "docs/architecture.md#target-manifest). Defaults to the SI bundle."
        ),
        default="data/SI/rubric_manifest.json",
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

    persona_files, context_template = asyncio.run(
        resolve_persona_inputs(args.rubric_manifest)
    )

    results, output_folder = asyncio.run(
        main(
            persona_model_config=persona_model_config,
            agent_model_config=agent_model_config,
            persona_files=persona_files,
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
            persona_names=None,
            verbose=True,
            output_folder=args.output,
            run_id=args.run_id,
            max_concurrent=args.max_concurrent,
            max_total_words=args.max_total_words,
            max_personas=args.max_personas,
            persona_speaks_first=not args.provider_speaks_first,
            session_types=args.sessions,
            resume=args.resume,
            persona_context_template_path=context_template,
        )
    )
    if results and all(r.get("skipped") for r in results):
        sys.exit(1)
