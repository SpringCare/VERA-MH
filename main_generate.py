#!/usr/bin/env python3

import asyncio
from typing import List, Dict, Any
from generate_conversations import ConversationRunner

async def generate_conversations(
    persona_model: str = "gpt-4",
    llm2_prompt: str = "",  #TODO: remove this
    max_turns: int = 3,
    runs_per_prompt: int = 2,
    persona_names: List[str] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate conversations and return results.
    
    Args:
        llm1_model: Model for LLM1
        llm2_prompt: Prompt name for LLM2  
        max_turns: Maximum turns per conversation
        runs_per_prompt: Number of runs per prompt
        prompts: List of prompts to use for LLM1
        verbose: Whether to print status messages
        
    Returns:
        List of conversation results
        
    Raises:
        ValueError: Configuration error
        Exception: Other errors
    """
    if verbose:
        print("🔄 Generating conversations...")
    
    # Configuration
    runner = ConversationRunner(
        llm1_model=persona_model,
        llm2_prompt=llm2_prompt,
        max_turns=max_turns,
        runs_per_prompt=runs_per_prompt,
        
    )
    
    # Run conversations
    results = await runner.run_conversations(persona_names=persona_names)

    if verbose:
        print(f"✅ Generated {len(results)} conversations → conversations/")
    
    return results

async def main(persona_model, max_turns, runs_per_prompt):
    """Main function to run LLM conversation simulations."""

    _ = await generate_conversations(persona_model=persona_model, max_turns=max_turns, runs_per_prompt=runs_per_prompt)
    # print("💡 To judge these conversations, run: python main_judge.py -f conversations/")
    return 0

if __name__ == "__main__":
    try:
        max_turns = 30
        runs_per_prompt = 5
        persona_model = 'gpt-4'
        exit_code = asyncio.run(main(persona_model=persona_model,max_turns=max_turns, runs_per_prompt=runs_per_prompt))
        exit(exit_code or 0)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        exit(1)