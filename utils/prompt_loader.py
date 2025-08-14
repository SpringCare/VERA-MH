"""Utilities for loading and managing prompts."""

from typing import Dict, Any
from .model_config_loader import get_model_for_prompt

def load_meta_prompt(prompt_name: str, prompts_dir: str = "prompts") -> str:
    """
    Load meta prompt from file in prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        prompts_dir: Directory containing prompt files
        
    Returns:
        The prompt content as a string
    """
    file_path = f"{prompts_dir}/{prompt_name}.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file '{file_path}' not found. Using default prompt.")
        return "You are a helpful AI assistant. Keep your responses concise and informative."
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return "You are a helpful AI assistant. Keep your responses concise and informative."

def load_prompt_config(prompt_name: str, prompts_dir: str = "prompts") -> Dict[str, Any]:
    """
    Load prompt configuration including model (from separate config), system prompt and initial message.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        prompts_dir: Directory containing prompt files
        
    Returns:
        Dictionary containing 'model', 'system_prompt' and 'initial_message'
    """
    file_path = f"{prompts_dir}/{prompt_name}.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Get model from separate configuration
        model = get_model_for_prompt(prompt_name)
        
        # Parse prompt content (system prompt and initial message only)
        if "---INITIAL_MESSAGE---" in content:
            parts = content.split("---INITIAL_MESSAGE---")
            system_prompt = parts[0].strip()
            initial_message = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Legacy format - just system prompt
            system_prompt = content
            initial_message = ""
        
        return {
            "model": model,
            "system_prompt": system_prompt,
            "initial_message": initial_message
        }
    except FileNotFoundError:
        print(f"Warning: Prompt file '{file_path}' not found. Using default config.")
        model = get_model_for_prompt(prompt_name)  # Still get model from config
        return {
            "model": model,
            "system_prompt": "You are a helpful AI assistant. Keep your responses concise and informative.",
            "initial_message": "Hello! How can I help you today?"
        }
    except Exception as e:
        print(f"Error loading prompt config: {e}")
        model = get_model_for_prompt(prompt_name)  # Still get model from config
        return {
            "model": model,
            "system_prompt": "You are a helpful AI assistant. Keep your responses concise and informative.",
            "initial_message": "Hello! How can I help you today?"
        }