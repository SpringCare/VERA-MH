"""Utilities for conversation logging."""

import logging
import os
from typing import Optional
from llm_clients import LLMInterface

def setup_conversation_logger(log_filename: str, run_id: str, log_folder: str = "logging", level=logging.INFO) -> logging.Logger:
    """
    Set up a logger for a specific conversation.
    
    Args:
        log_filename: Name of the log file (without extension)
        log_folder: Directory to save log files
        
    Returns:
        Configured logger instance
    """
    # Ensure log folder exists
    os.makedirs(log_folder, exist_ok=True)
    # Create unique logger name to avoid conflicts
    logger_name = f"{log_filename}"
    
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set logging level
    logger.setLevel(level)
    
    # Create file handler
    log_file_path = os.path.join(log_folder, run_id, f"{log_filename}.log")
    os.makedirs(os.path.join(log_folder, run_id), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger

# TODO: This should print all the llm1 and 2 settings
def log_conversation_start(logger: logging.Logger, llm1_model_str: str, llm1_prompt: str, 
                          llm2_name: str, llm2_model_str: str, initial_message: str, max_turns: int, 
                          llm1_model: LLMInterface, llm2_model: LLMInterface, logging: dict = {}):
    """Log conversation initialization details."""

    logger.info("=" * 60)
    logger.info("CONVERSATION STARTED")
    logger.info("=" * 60)
    logger.info("LLM1 Configuration:")
    logger.info(f"  - Model: {llm1_model_str}")
    logger.info(f"  - Prompt: {llm1_prompt}")
    logger.info(f"Configuration, temperature: {llm1_model.temperature}")
    logger.info(f"Configuration, max_tokens: {llm1_model.max_tokens}")
    logger.info("LLM2 Configuration:")
    logger.info(f"  - Name: {llm2_name}")
    logger.info(f"  - Model: {llm2_model_str}")
    logger.info(f"Configuration, temperature: {llm2_model.temperature}")
    logger.info(f"Configuration, max_tokens: {llm2_model.max_tokens}")
    logger.info("Conversation Settings:")
    # logger.info(f"  - Initial Message: {initial_message}")
    logger.info(f"  - Max Turns: {max_turns}")
    logger.info("=" * 60)

def log_conversation_turn(logger: logging.Logger, turn_number: int, speaker: str, 
                         input_message: str, response: str, early_termination: bool = False, logging: dict = {}):
    """Log individual conversation turn."""
    logger.info(f"TURN {turn_number} - {speaker}")
    logger.info(f"Input: {input_message}")
    logger.info(f"Response: {response}")
    logger.info(f"response_id: {logging.get('response_id')}")
    logger.info(f"logging: {logging}")
    if early_termination:
        logger.warning(f"EARLY TERMINATION detected by {speaker}")
    logger.info("-" * 40)
    

def log_conversation_end(logger: logging.Logger, total_turns: int, early_termination: bool, 
                        total_time: Optional[float] = None):
    """Log conversation completion details."""
    logger.info("=" * 60)
    logger.info("CONVERSATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total Turns: {total_turns}")
    logger.info(f"Early Termination: {early_termination}")
    if total_time:
        logger.info(f"Duration: {total_time:.2f} seconds")
    logger.info("=" * 60)

def log_error(logger: logging.Logger, error_message: str, exception: Exception = None):
    """Log error information."""
    logger.error(f"ERROR: {error_message}")
    if exception:
        logger.error(f"Exception: {str(exception)}")
        logger.error(f"Exception Type: {type(exception).__name__}")

def cleanup_logger(logger: logging.Logger):
    """Clean up logger handlers to prevent memory leaks."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)