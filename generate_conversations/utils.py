#!/usr/bin/env python3
"""Utility functions for conversation generation."""

import csv
from pathlib import Path
from typing import List, Optional

# TODO: hardcoded names
def load_prompts_from_csv(name_list: Optional[List[str]] = None, 
                          prompt_path = "data/personas.csv",
                          prompt_template_path = "data/persona_prompt_template.txt"
                          ) -> List[dict[str, str]]:
    """Load prompts from personas.csv file and return them as a list.
    
    Args:
        name_list: Optional list of names to filter by. If None, returns all prompts.
        prompt_path: Path to the CSV file containing persona data
        prompt_template_path: Path to the template file for formatting prompts
    """

    csv_path = Path(prompt_path)
    template_path = Path(prompt_template_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompts CSV file not found: {csv_path}")
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Read template once outside the loop for efficiency
    with open(template_path, 'r', encoding='utf-8') as template_file:
        template = template_file.read()
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by name list if provided
            if name_list is not None and row["Name"] not in name_list:
                continue
            
            # Format the template with row data
            try:
                prompt = template.format(**row)
                row['prompt'] = prompt
                data.append(row)
            except KeyError as e:
                print(f"Warning: Missing key {e} in row for {row.get('Name', 'Unknown')}")
                continue

    return data