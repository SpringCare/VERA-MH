#!/usr/bin/env python3

import asyncio
from llm_clients import LLMFactory
from .conversation_simulator import ConversationSimulator
from utils.prompt_loader import load_prompt_config
from .utils import load_prompts_from_csv
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
from typing import List, Dict, Any, Optional

class ConversationRunner:
    """Handles running LLM conversations with logging and file management."""
    
    def __init__(
        self, 
        llm1_model: str, 
        llm2_prompt: str, 
        max_turns: int = 6, 
        runs_per_prompt: int = 3,
    ):
        self.llm1_model = llm1_model
        self.llm2_prompt = llm2_prompt
        self.max_turns = max_turns
        self.runs_per_prompt = runs_per_prompt
        
    # def _extract_persona_name(self, prompt_text: str) -> str:
    #     """Extract persona name from the prompt text."""
    #     import re
    #     # Look for pattern "Your name is [Name],"
    #     match = re.search(r'Your name is ([^,]+),', prompt_text)
    #     if match:
    #         return match.group(1).strip()
    #     return "Unknown"
    
    async def run_single_conversation(
        self, 
        llm1_config: dict, 
        llm2, 
        max_turns: int, 
        conversation_id: int, 
        run_number: int
    ) -> Dict[str, Any]:
        """Run a single conversation asynchronously."""
        
        model_name = llm1_config["model"]
        system_prompt = llm1_config["prompt"]  # This is now the full persona prompt
        
        # Extract persona name from the prompt
        persona_name = llm1_config["name"]

        # Generate filename base using persona name, model, and run number
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        model_short = model_name.replace("claude-3-", "c3-").replace("gpt-", "g")
        persona_safe = persona_name.replace(" ", "_").replace(".", "")
        filename_base = f"conversation_{persona_safe}_{model_short}_run{run_number}_{timestamp}"
        
        # Setup logging
        logger = setup_conversation_logger(filename_base)
        
        start_time = time.time()
        
        try:
            # Create LLM1 instance with the persona prompt directly
            print(system_prompt)
            llm1 = LLMFactory.create_llm(
                model_name=model_name,
                name=f"{model_name.split('-')[0].title()} {persona_name}",
                system_prompt=system_prompt
            )
            
            # Use a standard greeting as initial message
            # initial_message = "Hello! I'd like to talk to someone about how I've been feeling."
            
            # Log conversation start
            log_conversation_start(
                logger=logger,
                llm1_model=model_name,
                llm1_prompt=persona_name,
                llm2_name=llm2.get_name(),
                llm2_model=getattr(llm2, 'model_name', 'unknown'),
                initial_message="initial_message",
                max_turns=max_turns
            )
            
            # Create conversation simulator
            simulator = ConversationSimulator(llm1, llm2)
            
            # Run the conversation
            conversation = await simulator.start_conversation(max_turns)
            
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
            conversation_file = f"conversations/{filename_base}.txt"
            simulator.save_conversation(f"{filename_base}.txt", 'conversations')
            
            result = {
                "id": conversation_id,
                "llm1_model": model_name,
                "llm1_prompt": persona_name,
                "run_number": run_number,
                "turns": len(conversation),
                "filename": f"{filename_base}.txt",
                "log_file": f"{filename_base}.log",
                "duration": conversation_time,
                "early_termination": early_termination,
                "conversation": conversation
            }
            
            print(f'done {llm1_config}, {run_number}')
    
            return result
            
        except Exception as e:
            log_error(logger, f"Error in conversation {conversation_id}", e)
            raise
        
        finally:
            # Clean up logger to prevent memory leaks
            cleanup_logger(logger)
    
    async def run_conversations(self, persona_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run multiple conversations concurrently."""
        
        # Load prompts from CSV based on persona names
        # those are already filtered
        loaded = load_prompts_from_csv(persona_names)
        persona_names, llm1_prompts = list(loaded.keys()), list(loaded.values())

        # Load LLM2 configuration (fixed, shared across all conversations)
        config2 = load_prompt_config(self.llm2_prompt)
        llm2 = LLMFactory.create_llm(
            model_name=config2["model"],
            name="Claude Philosopher",
            system_prompt=config2["system_prompt"]
        )
        
        # Create tasks for all conversations (each prompt run multiple times)
        tasks = []
        conversation_id = 1
        
        for persona in persona_names:        
            for run in range(1, self.runs_per_prompt + 1):
                print(f"Running prompt: {persona}, run {run}")
                tasks.append(
                    self.run_single_conversation(
                        {"model": self.llm1_model, "prompt": loaded[persona], "name": persona},
                        llm2, 
                        self.max_turns, 
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
        
        print(f"\nCompleted {len(results)} conversations in {total_time:.2f} seconds")
        
        return results