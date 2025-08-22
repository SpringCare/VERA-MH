#!/usr/bin/env python3

import asyncio
from typing import List, Dict, Any
from generate_conversations import ConversationRunner
from datetime import datetime

async def generate_conversations(
    persona_model_config: Dict[str, Any],
    llm2_model_config: Dict[str, Any],
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
        llm2_model_config: Configuration dictionary for the LLM2 model
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
        llm1_meta = persona_model_config["model"].replace("-", "_").replace(".", "_")
        llm2_meta = llm2_model_config["model"].replace("-", "_").replace(".", "_")
        folder_name = f"p_{llm1_meta}__a_{llm2_meta}_{timestamp}"
    
    # Configuration
    runner = ConversationRunner(
        persona_model_config=persona_model_config,
        llm2_model_config=llm2_model_config,
        max_turns=max_turns,
        runs_per_prompt=runs_per_prompt,
        folder_name=folder_name,
    )
    
    # Run conversations
    results = await runner.run_conversations(persona_names=persona_names)

    if verbose:
        print(f"✅ Generated {len(results)} conversations → conversations/{folder_name}/")
    
    return results

async def main(persona_model_config: Dict[str, Any], llm2_model_config: Dict[str, Any], max_turns: int, runs_per_prompt: int, folder_name: str = None):
    """Main function to run LLM conversation simulations."""
    return await generate_conversations(
        persona_model_config=persona_model_config, 
        llm2_model_config=llm2_model_config,
        max_turns=max_turns, 
        runs_per_prompt=runs_per_prompt,
        folder_name=folder_name,
    )

if __name__ == "__main__":
    max_turns = 30
    runs_per_prompt = 5
    
    # Persona model configuration
    persona_model_config = {
        "model": "gpt-5",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # LLM2 model configuration
    llm2_model_config = {
        "model": "claude-sonnet-4-20250514",
        "prompt_name": "claude_philosopher",  # This should match a prompt config file
        "name": "Claude Philosopher",  # Display name for the LLM
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # Optional: specify custom folder name
    # folder_name = "custom_experiment_name"
    
    exit_code = asyncio.run(main(
        persona_model_config=persona_model_config,
        llm2_model_config=llm2_model_config,
        max_turns=max_turns, 
        runs_per_prompt=runs_per_prompt,
        folder_name=None,  # Will use default format
    ))
    exit(exit_code or 0)