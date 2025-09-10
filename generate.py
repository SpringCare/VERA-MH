#!/usr/bin/env python3

import asyncio
from re import DEBUG
from socket import timeout
from typing import List, Dict, Any
from generate_conversations import ConversationRunner
from datetime import datetime
import os

async def generate_conversations(
    persona_model_config: Dict[str, Any],
    agent_model_config: Dict[str, Any],
    max_turns: int = 3,
    runs_per_prompt: int = 2,
    persona_names: List[str] = None,
    verbose: bool = True,
    folder_name: str = None,
    extra_run_params: Dict[str, Any] = {},
) -> List[Dict[str, Any]]:
    """
    Generate conversations and return results.
    
    Args:
        persona_model_config: Configuration dictionary for the persona model
        agent_model_config: Configuration dictionary for the agent model
        max_turns: Maximum turns per conversation
        runs_per_prompt: Number of runs per prompt
        persona_names: List of persona names to use
        verbose: Whether to print status messages
        folder_name: Custom folder name for saving conversations. If None, uses default format.
        
    Returns:
        List of conversation results
        
    Raises:
        ValueError: Configuration error
        Exception: Other errors
    """
    if verbose:
        print("🔄 Generating conversations...")
    
    # Generate default folder name if not provided
    if folder_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        persona_meta = persona_model_config["model"].replace("-", "_").replace(".", "_")
        agent_meta = agent_model_config["model"].replace("-", "_").replace(".", "_")
        run_id = f'p_{persona_meta}__a_{agent_meta}_{timestamp}_t{max_turns}_r{runs_per_prompt}_{extra_run_params}'
        folder_name = f"conversations/{run_id}"
        os.makedirs(folder_name, exist_ok=True)
    
    # Configuration
    runner = ConversationRunner(
        persona_model_config=persona_model_config,
        agent_model_config=agent_model_config,
        max_turns=max_turns,
        runs_per_prompt=runs_per_prompt,
        folder_name=folder_name,
        run_id=run_id,
    )
    
    # Run conversations
    results = await runner.run_conversations(persona_names=persona_names)

    if verbose:
        print(f"✅ Generated {len(results)} conversations → conversations/{folder_name}/")
    
    return results

async def main(persona_model_config: Dict[str, Any], agent_model_config: Dict[str, Any], max_turns: int, runs_per_prompt: int, folder_name: str = None, extra_run_params: Dict[str, Any] = {}):
    """Main function to run LLM conversation simulations."""
    return await generate_conversations(
        persona_model_config=persona_model_config, 
        agent_model_config=agent_model_config,
        max_turns=max_turns, 
        runs_per_prompt=runs_per_prompt,
        folder_name=folder_name,
        extra_run_params=extra_run_params,
    )

if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        max_turns = 3
        runs_per_prompt = 3
    else:   
        max_turns = 30
        runs_per_prompt = 5
    
    # Persona model configuration
    persona_model_config = {
        # "model": "claude-sonnet-4-20250514",
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 1000, 
        "timeout":1000, # shoudl be seconds
        "max_completion_tokens":5000,
    }
    
    # Agent model configuration
    agent_model_config = {
        "model": "gpt-4o",
        "name": "GPT-4o",
       
        "prompt_name": "",  # This should match a prompt config file
        # "name": "Claude Sonnet",  # Display name for the LLM
        # "model": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # Optional: specify custom folder name
    # folder_name = "custom_experiment_name"
    
    # note: we are discarding the results, becuase they get saved to file
    _ = asyncio.run(main(
        persona_model_config=persona_model_config,
        agent_model_config=agent_model_config,
        max_turns=max_turns, 
        runs_per_prompt=runs_per_prompt,
        extra_run_params={k: v for k, v in persona_model_config.items() if k not in ["model", "temperature", "max_tokens"]},
        folder_name=None,  # Will use default format
    ))
    