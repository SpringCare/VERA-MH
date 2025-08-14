#!/usr/bin/env python3

import asyncio
from llm_factory import LLMFactory
from conversation_simulator import ConversationSimulator
from utils.prompt_loader import load_prompt_config
from utils.logging_utils import (
    setup_conversation_logger, 
    log_conversation_start, 
    log_conversation_turn, 
    log_conversation_end, 
    log_error, 
    cleanup_logger
)
from datetime import datetime
import time

def generate_conversation_filename(llm1_model: str, llm1_prompt: str, llm2_name: str) -> str:
    """Generate a descriptive filename for conversation logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds for uniqueness
    model_short = llm1_model.replace("claude-3-", "c3-").replace("gpt-", "g")  # Shorten model names
    return f"conversation_{model_short}_{llm1_prompt}_vs_{llm2_name}_{timestamp}"

async def run_single_conversation(llm1_config: dict, llm2, llm2_prompt: str, max_turns: int, conversation_id: int, run_number: int) -> dict:
    """Run a single conversation asynchronously."""
    
    model_name = llm1_config["model"]
    prompt_name = llm1_config["prompt"]
    
    # Generate filename base for both conversation and log files (include run number)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    model_short = model_name.replace("claude-3-", "c3-").replace("gpt-", "g")
    filename_base = f"conversation_{model_short}_{prompt_name}_vs_{llm2_prompt}_{timestamp}_run{run_number}"
    
    # Setup logging
    logger = setup_conversation_logger(filename_base)
    
    start_time = time.time()
    
    try:
        # Load LLM1 prompt configuration
        prompt_config = load_prompt_config(prompt_name)
        
        # Create LLM1 instance with specified model and prompt
        llm1 = LLMFactory.create_llm(
            model_name=model_name,
            name=f"{model_name.split('-')[0].title()} {prompt_name.title()}",
            system_prompt=prompt_config["system_prompt"]
        )
        
        # Use initial message from LLM1 prompt
        initial_message = prompt_config["initial_message"] or "Hello! Let's start a conversation."
        
        # Log conversation start
        log_conversation_start(
            logger=logger,
            llm1_model=model_name,
            llm1_prompt=prompt_name,
            llm2_name=llm2.get_name(),
            llm2_model=getattr(llm2, 'model_name', 'unknown'),
            initial_message=initial_message,
            max_turns=max_turns
        )
        
        # Create conversation simulator
        simulator = ConversationSimulator(llm1, llm2)
        
        # Run the conversation
        conversation = await simulator.start_conversation(initial_message, max_turns)
        
        # Log each conversation turn
        for i, turn in enumerate(conversation, 1):
            log_conversation_turn(
                logger=logger,
                turn_number=i,
                speaker=turn.get("speaker", "Unknown"),
                input_message=turn.get("input", ""),
                response=turn.get("response", ""),
                early_termination=turn.get("early_termination", False)
            )
        
        end_time = time.time()
        conversation_time = end_time - start_time
        
        # Check if conversation ended early
        early_termination = any(turn.get("early_termination", False) for turn in conversation)
        
        # Log conversation end
        log_conversation_end(
            logger=logger,
            total_turns=len(conversation),
            early_termination=early_termination,
            total_time=conversation_time
        )
        
        # Save conversation file
        simulator.save_conversation(f"{filename_base}.txt", 'conversations')
        print(f'done {llm1_config}, {run_number}')

        return {
            "id": conversation_id,
            "llm1_model": model_name,
            "llm1_prompt": prompt_name,
            "run_number": run_number,
            "turns": len(conversation),
            "filename": f"{filename_base}.txt",
            "log_file": f"{filename_base}.log",
            "duration": conversation_time,
            "early_termination": early_termination,
            "conversation": conversation
        }
        
    except Exception as e:
        log_error(logger, f"Error in conversation {conversation_id}", e)
        raise
    
    finally:
        # Clean up logger to prevent memory leaks
        cleanup_logger(logger)

async def main():
    """Main function to run multiple LLM conversation simulations concurrently."""
    
    try:
        # Configuration
        llm1_model = "gpt-4"  # Single line: set the model for LLM1
        llm1_prompts = ["assistant", "creative"]  # Loop over various assistants
        llm2_prompt = "philosopher"    # Fixed LLM2 prompt
        max_turns = 3
        runs_per_prompt = 2  # Number of times to run each prompt
        
        # Load LLM2 configuration (fixed, shared across all conversations)
        config2 = load_prompt_config(llm2_prompt)
        llm2 = LLMFactory.create_llm(
            model_name=config2["model"],
            name="Claude Philosopher",
            system_prompt=config2["system_prompt"]
        )
        
        # Create tasks for all conversations (each prompt run 3 times)
        tasks = []
        conversation_id = 1
        
        for prompt in llm1_prompts:        
            for run in range(1, runs_per_prompt + 1):
                print(f"Running prompt: {prompt}, run {run}")
                tasks.append(
                    run_single_conversation(
                        {"model": llm1_model, "prompt": prompt}, 
                        llm2, 
                        llm2_prompt, 
                        max_turns, 
                        conversation_id,
                        run
                    )
                )
                conversation_id += 1
        
        start_time = datetime.now()
        
        # Run all conversations concurrently
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Summary
        print(f"\nCompleted {len(results)} conversations in {total_time:.2f} seconds")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please make sure you have set your ANTHROPIC_API_KEY in the .env file")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())