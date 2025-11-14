#!/usr/bin/env python3

import asyncio
from typing import List, Dict, Any, Optional
from generate_conversations import ConversationRunner
from datetime import datetime
import os
import argparse

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
    folder_name: Optional[str] = None,
    run_id: Optional[str] = None,
    max_concurrent: Optional[int] = None,
) -> List[Dict[str, Any]]:
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
        folder_name: Custom folder name for saving conversations. If None, uses default format.
        max_concurrent: Maximum number of concurrent conversations. If None, runs all conversations concurrently.
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
        print(f"  - Folder name: {folder_name}")
        print(f"  - Run ID: {run_id}")
        print(f"  - Max concurrent: {max_concurrent}")
    
    # Generate default folder name if not provided
    if folder_name is None:
        folder_name = "conversations"
    
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        persona_info = persona_model_config["model"].replace("-", "_").replace(".", "_")
        agent_info = agent_model_config["model"].replace("-", "_").replace(".", "_")
        if persona_extra_run_params:
            persona_info += f"_{persona_extra_run_params}"
            
        if agent_extra_run_params:
            agent_info += f"_{agent_extra_run_params}"
        
        run_id = f'p_{persona_info}__a_{agent_info}__t{max_turns}__r{runs_per_prompt}__{timestamp}'
        folder_name = f"{folder_name}/{run_id}"
        # TODO: do we want to give a message if the folder already exists?
        os.makedirs(folder_name, exist_ok=True)
    
    # Configuration
    runner = ConversationRunner(
        persona_model_config=persona_model_config,
        agent_model_config=agent_model_config,
        max_turns=max_turns,
        runs_per_prompt=runs_per_prompt,
        folder_name=folder_name,
        run_id=run_id,
        max_concurrent=max_concurrent,
    )
    
    # Run conversations
    results = await runner.run_conversations(persona_names=persona_names)

    if verbose:
        print(f"✅ Generated {len(results)} conversations → {folder_name}/")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM conversations")

    parser.add_argument(
        "--user-agent",
        "-u",
        help="Model for the user-agent. Examples: claude-3-5-sonnet-20241022, gemini-1.5-pro, llama3:8b"
    )
    parser.add_argument(
        "--user-agent-extra-params",
        "-uep",
        help="Extra parameters for the user-agent. Examples: temperature=0.7, max_tokens=1000",
        type=parse_key_value_list,
        default={},
    )

    parser.add_argument(
        "--provider-agent",
        "-p",
        help="Model for the provider-agent. Examples: claude-3-5-sonnet-20241022, gemini-1.5-pro, llama3:8b"
    )

    parser.add_argument(
        "--provider-agent-extra-params",
        "-pep",
        help="Extra parameters for the provider-agent. Examples: temperature=0.7, max_tokens=1000",
        default={},
        type=parse_key_value_list,
    )

    parser.add_argument(
        "--runs",
        "-r",
        help="Number of runs per prompt",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--turns",
        "-t",
        help="Number of turns per conversation",
        default=30,
        type=int,
    )

    parser.add_argument(
        "--run-id",
        "-i",
        help="Run ID for the conversations for this run. If not provided, a default will be generated.",
        default=None,
    )

    parser.add_argument(
        "--folder-name",
        "-f",
        help="Folder name containng the conversations for this run. Default is 'conversations'.",
        default='conversations',
    )

    parser.add_argument(
        "--max-concurrent",
        "-c",
        help="Maximum number of concurrent conversations. Default is None (run all conversations concurrently).",
        default=None,
        type=int,
    )

    args = parser.parse_args()

    persona_model_config = {
        "model": args.user_agent,
        **args.user_agent_extra_params,
    }

    agent_model_config = {
        "model": args.provider_agent,
        # TODO: why does agent need a name, but not persona?
        "name": args.provider_agent,
        **args.provider_agent_extra_params,
    }

    # TODO: Do the run id here, so that it can be printed when starting
    # Note: we are discarding the results, because they are saved to file
    _ = asyncio.run(main(
        persona_model_config=persona_model_config,
        agent_model_config=agent_model_config,
        max_turns=args.turns, 
        runs_per_prompt=args.runs,
        persona_extra_run_params={k: v for k, v in persona_model_config.items() if k not in ["model", "model_name", "name", "temperature", "max_tokens"]},
        agent_extra_run_params={k: v for k, v in agent_model_config.items() if k not in ["model", "model_name", "name", "temperature", "max_tokens"]},
        folder_name=args.folder_name,  
        max_concurrent=args.max_concurrent,
    ))
    