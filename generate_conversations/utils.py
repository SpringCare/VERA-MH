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
    multiple_responses: bool = False,
) -> List[dict[str, str]]:
    """Load prompts from personas.csv file and return them as a list.

    Args:
        name_list: Optional list of names to filter by. If None, returns all prompts.
        prompt_path: Path to the CSV file containing persona data
        prompt_template_path: Path to the template file for formatting prompts
        multiple_responses: If True, include instructions for generating
            multiple responses
    """

    csv_path = Path(prompt_path)
    template_path = Path(prompt_template_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Prompts CSV file not found: {csv_path}")

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # Read template once outside the loop for efficiency
    with open(template_path, "r", encoding="utf-8") as template_file:
        template = template_file.read()

    # Remove multiple response instructions if not needed
    if not multiple_responses:
        lines = template.split("\n")
        filtered_lines = []
        skip_next = False
        for line in lines:
            # Skip the three lines about multiple responses
            if "When asked to provide multiple responses" in line:
                skip_next = 2  # Skip this line and the next 2
                continue
            if skip_next > 0:
                skip_next -= 1
                continue
            filtered_lines.append(line)
        template = "\n".join(filtered_lines)

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
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
