#!/usr/bin/env python3

import asyncio
import os
import uuid
from llm_clients import LLMFactory
from .conversation_simulator import ConversationSimulator
from .utils import load_prompts_from_csv
from utils.logging_utils import (
    setup_conversation_logger, 
    log_conversation_start, 
    log_conversation_turn, 
    log_conversation_end, 
    cleanup_logger
)
from datetime import datetime
import time
from typing import List, Dict, Any, Optional

class ConversationRunner:
    """Handles running LLM conversations with logging and file management."""
    
    def __init__(
        self,
        persona_model_config: Dict[str, Any],
        agent_model_config: Dict[str, Any],
        run_id: str,
        max_turns: int = 6,
        runs_per_prompt: int = 3,
        folder_name: str = "conversations",
        max_total_words: Optional[int] = None,
    ):
        self.persona_model_config = persona_model_config
        self.agent_model_config = agent_model_config
        self.max_turns = max_turns
        self.runs_per_prompt = runs_per_prompt
        self.folder_name = folder_name
        self.run_id = run_id
        self.max_total_words = max_total_words

        self.AGENT_SYSTEM_PROMPT = self.agent_model_config.get("system_prompt", "You are a helpful AI assistant.")
    
    async def run_single_conversation(
        self, 
        persona_config: dict, 
        agent, 
        max_turns: int, 
        conversation_id: int, 
        run_number: int,
        **kargs: dict
    ) -> Dict[str, Any]:
        """Run a single conversation asynchronously."""
        model_name = persona_config["model"]
        system_prompt = persona_config["prompt"]  # This is now the full persona prompt
        persona_name = persona_config["name"]

        # Generate filename base using persona name, model, and run number
        tag = uuid.uuid4().hex[:6]
        # TODO: consider removing timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # TODO: should this be inside the LLM class?
        model_short = model_name.replace("claude-3-", "c3-").replace("gpt-", "g").replace("claude-sonnet-4-", "cs4-")
        persona_safe = persona_name.replace(" ", "_").replace(".", "")
        filename_base = f"{tag}_{persona_safe}_{model_short}_run{run_number}"
        os.makedirs(f"{self.folder_name}", exist_ok=True)
    

        # Setup logging
        logger = setup_conversation_logger(filename_base, run_id=self.run_id)
        start_time = time.time()
        
        # Create LLM1 instance with the persona prompt and configuration
        persona = LLMFactory.create_llm(
            model_name=model_name,
            name=f"{model_short} {persona_name}",
            system_prompt=system_prompt,
            **self.persona_model_config
        )

        # Log conversation start
        log_conversation_start(
            logger=logger,
            llm1_model_str=model_name,
            llm1_prompt=persona_name,
            llm2_name=agent.get_name(),
            llm2_model_str=getattr(agent, 'model_name', 'unknown'),
            initial_message="",
            max_turns=max_turns,
            llm1_model=persona,
            llm2_model=agent,
        )
        
        # Create conversation simulator and run conversation
        simulator = ConversationSimulator(persona, agent)
        # Run the conversation - let first speaker start naturally with None
        conversation = await simulator.start_conversation(initial_message=None, max_turns=max_turns, max_total_words=self.max_total_words)
            
        
        # Log each conversation turn
        for i, turn in enumerate(conversation, 1):
            log_conversation_turn(
                logger=logger,
                turn_number=i,
                speaker=turn.get("speaker", "Unknown"),
                input_message=turn.get("input", ""),
                response=turn.get("response", ""),
                early_termination=turn.get("early_termination", False),
                logging=turn.get("logging", {})
            )
        
        # Calculate timing and check early termination
        end_time = time.time()
        conversation_time = end_time - start_time
        early_termination = any(turn.get("early_termination", False) for turn in conversation)
        
        # Log conversation end
        log_conversation_end(
            logger=logger,
            total_turns=len(conversation),
            early_termination=early_termination,
            total_time=conversation_time
        )
        
        # Save conversation file
        simulator.save_conversation(f"{filename_base}.txt", self.folder_name)
        
        result = {
            "id": conversation_id,
            "llm1_model": model_name,
            "llm1_prompt": persona_name,
            "run_number": run_number,
            "turns": len(conversation),
            "filename": f"{self.folder_name}/{filename_base}.txt",
            "log_file": f"{self.folder_name}/{filename_base}.log",
            "duration": conversation_time,
            "early_termination": early_termination,
            "conversation": conversation
        }
        

        cleanup_logger(logger)
        return result
    
    async def run_conversations(self, persona_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run multiple conversations concurrently."""
        # Load prompts from CSV based on persona names
        personas = load_prompts_from_csv(persona_names)
        
        # Load agent configuration (fixed, shared across all conversations)
        agent = LLMFactory.create_llm(
            model_name=self.agent_model_config["model"],
            name=self.agent_model_config.pop("name"),
            system_prompt=self.AGENT_SYSTEM_PROMPT,
            **self.agent_model_config
        )
        
        # Create tasks for all conversations (each prompt run multiple times)
        tasks = []
        conversation_id = 1
        
        for persona in personas:

            for run in range(1, self.runs_per_prompt + 1):
                
                tasks.append(
                    # TODO: should we pass the persona object here?
                    self.run_single_conversation(
                        {"model": self.persona_model_config["model"], "prompt": persona["prompt"], "name": persona["Name"], "run": run},
                        agent, 
                        self.max_turns, 
                        conversation_id,
                        run
                    )
                )
                conversation_id += 1
        
        # Run all conversations concurrently
        start_time = datetime.now()
        results = await asyncio.gather(*tasks)
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"\nCompleted {len(results)} conversations in {total_time:.2f} seconds")
        return results