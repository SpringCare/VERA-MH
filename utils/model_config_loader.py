"""Utilities for loading model configurations."""

import json
from typing import Any, Dict


def load_model_config(config_file: str = "model_config.json") -> Dict[str, Any]:
    """
    Load model configuration from JSON file.

    Args:
        config_file: Path to the model configuration JSON file

    Returns:
        Dictionary containing model configuration
    """
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Model config file '{config_file}' not found. Using defaults.")
        return {"prompt_models": {}, "default_model": "claude-3-5-sonnet-20241022"}
    except Exception as e:
        print(f"Error loading model config: {e}")
        return {"prompt_models": {}, "default_model": "claude-3-5-sonnet-20241022"}


def get_model_for_prompt(
    prompt_name: str, config_file: str = "model_config.json"
) -> str:
    """
    Get the model name for a specific prompt.

    Args:
        prompt_name: Name of the prompt
        config_file: Path to the model configuration JSON file

    Returns:
        Model name string
    """
    config = load_model_config(config_file)
    return config["prompt_models"].get(prompt_name, config["default_model"])
