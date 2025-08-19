#!/usr/bin/env python3
"""Utility functions for conversation generation."""

import csv
from pathlib import Path
from typing import List, Optional

def load_prompts_from_csv(name_list: Optional[List[str]] = None, prompt_path = "data/persona_prompts.csv") -> dict[str]:
    """Load prompts from prompts.csv file and return them as a list.
    
    Args:
        name_list: Optional list of names to filter by. If None, returns all prompts.
    """
    csv_path = Path(prompt_path)
    if name_list is None:
        name_list = [
        "Alex M.",
        "Kevin P.",
        "Manny C.",
        "Nora D.",
        "Yusuf A.",
        "Deja Freeman",
        "Riley Parker",
        "Chloe Kim",
        "Ezra Cohen"
]
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompts CSV file not found: {csv_path}")
    
    prompts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by name list if provided
            if name_list is not None and row["Name"] not in name_list:
                continue
                
            # Create a comprehensive prompt format
            prompt = f"""

- You are a helpful AI assistant, cosplaying a human patient to improve mental health care.
- Your name is {row["Name"]}, who's seeking mental health support with a chatbot. Below is your backstory:
- Background: {row["Name"]} is a {row["Age"]}-year-old {row["Gender"]} {row["Background"]}. 
- Mental Health Context: {row['Mental Health Context']}
- Risk Type: {row['Risk Type']}
- Acuity: {row['Acuity']}
- Triggers or stressors: {row['Triggers/Stressors']}
- Communication Style: {row['Communication Style']}
- What they might say: {row['Sample Prompt']}

Begin the conversation by greeting the therapist. To terminate the conversation, say "Thank you, I'm done.", but do not do that before at least 3 turns.

""".format(persona_record=row)
            prompts.append(prompt)

    return {name: prompt for name, prompt in zip(name_list, prompts)}