#!/usr/bin/env python3

import asyncio
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
        folder_name = f"conversations/p_{persona_meta}__a_{agent_meta}_{timestamp}_t{max_turns}_r{runs_per_prompt}"
        os.makedirs(folder_name, exist_ok=True)
    
    # Configuration
    runner = ConversationRunner(
        persona_model_config=persona_model_config,
        agent_model_config=agent_model_config,
        max_turns=max_turns,
        runs_per_prompt=runs_per_prompt,
        folder_name=folder_name,
    )
    
    # Run conversations
    results = await runner.run_conversations(persona_names=persona_names)

    if verbose:
        print(f"✅ Generated {len(results)} conversations → conversations/{folder_name}/")
    
    return results

async def main(persona_model_config: Dict[str, Any], agent_model_config: Dict[str, Any], max_turns: int, runs_per_prompt: int, folder_name: str = None):
    """Main function to run LLM conversation simulations."""
    return await generate_conversations(
        persona_model_config=persona_model_config, 
        agent_model_config=agent_model_config,
        max_turns=max_turns, 
        runs_per_prompt=runs_per_prompt,
        folder_name=folder_name,
    )

if __name__ == "__main__":
    max_turns = 5
    runs_per_prompt = 1
    
    # Persona model configuration
    persona_model_config = {
        # "model": "claude-sonnet-4-20250514",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # Agent model configuration
    agent_model_config = {
        "model": "gpt-4o-mini",
        "name": "GPT-4o-mini",
        # "model": "claude-sonnet-4-20250514",
        "prompt_name": "",  # This should match a prompt config file
        # "name": "Claude Sonnet",  # Display name for the LLM
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # Optional: specify custom folder name
    # folder_name = "custom_experiment_name"
    
    exit_code = asyncio.run(main(
        persona_model_config=persona_model_config,
        agent_model_config=agent_model_config,
        max_turns=max_turns, 
        runs_per_prompt=runs_per_prompt,
        folder_name=None,  # Will use default format
    ))
    exit(exit_code or 0)