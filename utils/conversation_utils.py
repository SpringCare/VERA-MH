"""Utilities for conversation management and file operations."""

from datetime import datetime
from typing import List, Dict, Any

def generate_conversation_filename(prefix: str = "conversation") -> str:
    """
    Generate a timestamped filename for conversation logs.
    
    Args:
        prefix: Prefix for the filename
        
    Returns:
        Formatted filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.txt"

def save_conversation_to_file(conversation_history: List[Dict[str, Any]], filename: str, folder: str, llm1_name: str = None) -> None:
    """
    Save conversation history to a text file.
    
    Args:
        conversation_history: List of conversation turns
        filename: Output filename
        folder: Output folder
        llm1_name: Name of LLM1 to identify it as 'user'
    """
    summary = format_conversation_summary(conversation_history, llm1_name)
    with open(f"{folder}/{filename}", 'w', encoding='utf-8') as f:
        f.write(summary)

def format_conversation_summary(conversation_history: List[Dict[str, Any]], llm1_name: str = None) -> str:
    """
    Format conversation history into a readable summary.
    
    Args:
        conversation_history: List of conversation turns
        llm1_name: Name of LLM1 to identify it as 'user'
        
    Returns:
        Formatted conversation summary
    """
    if not conversation_history:
        return "No conversation recorded."
    
    summary = ""
    
    for i, turn in enumerate(conversation_history):
        speaker_name = turn['speaker']
        
        # Determine if this is LLM1 (user) or LLM2 (agent)
        if llm1_name and llm1_name in speaker_name:
            prefix = "user:"
        else:
            prefix = "agent:"
        
        # Add the response with proper formatting
        summary += f"{prefix} {turn['response']}\n"
        
        # Check for early termination flag
        if turn.get('early_termination', False):
            summary += f"[CONVERSATION ENDED - {speaker_name} signaled termination]\n"
        
        # Add empty line between speakers, but not after the last turn
        if i < len(conversation_history) - 1:
            summary += "\n"
    
    return summary