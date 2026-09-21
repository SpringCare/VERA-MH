#!/usr/bin/env python3
"""Utility functions for conversation generation."""

import csv
from pathlib import Path
from typing import List, Optional


# TODO: hardcoded names
def load_prompts_from_csv(
    name_list: Optional[List[str]] = None,
    prompt_path="data/personas.tsv",
    prompt_template_path="data/persona_prompt_template.txt",
    max_personas: Optional[int] = None,
) -> List[dict[str, str]]:
    """Load prompts from personas.csv file and return them as a list.

    Args:
        name_list: Optional list of names to filter by. If None, returns all prompts.
        prompt_path: Path to the CSV file containing persona data
        prompt_template_path: Path to the template file for formatting prompts
        max_personas: Optional maximum number of personas to load
    """

    csv_path = Path(prompt_path)
    template_path = Path(prompt_template_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Prompts CSV file not found: {csv_path}")

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    if max_personas is not None and max_personas <= 0:
        raise ValueError("max_personas must be > 0")

    # Read template once outside the loop for efficiency
    with open(template_path, "r", encoding="utf-8") as template_file:
        template = template_file.read()

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Stop if we've reached max_personas
            if max_personas is not None and len(data) >= max_personas:
                break

            # Filter by name list if provided
            if name_list is not None and row["Name"] not in name_list:
                continue

            # Format the template with row data
            try:
                prompt = template.format(**row)
                row["prompt"] = prompt
                data.append(row)
            except KeyError as e:
                print(
                    f"Warning: Missing key {e} in row for {row.get('Name', 'Unknown')}"
                )
                continue

    return data
