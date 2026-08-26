#!/usr/bin/env python3
"""Utility functions for conversation generation."""

import csv
from pathlib import Path
from string import Formatter
from typing import List, Optional


# TODO: hardcoded names
def load_prompts_from_csv(
    name_list: Optional[List[str]] = None,
    prompt_path="data/SI/personas.tsv",
    prompt_template_path="data/persona_prompt_template.txt",
    *,
    persona_context_template_path: str,
    max_personas: Optional[int] = None,
) -> List[dict[str, str]]:
    """Load prompts from personas.csv file and return them as a list.

    Args:
        name_list: Optional list of names to filter by. If None, returns all prompts.
        prompt_path: Path to the CSV file containing persona data
        prompt_template_path: Path to the template file for formatting prompts
        persona_context_template_path: Required schema-specific context template.
        max_personas: Optional maximum number of personas to load
    """

    csv_path = Path(prompt_path)
    template_path = Path(prompt_template_path)
    context_template_path = Path(persona_context_template_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Prompts CSV file not found: {csv_path}")

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    if not context_template_path.exists():
        raise FileNotFoundError(
            f"Persona context template file not found: {context_template_path}"
        )

    if max_personas is not None and max_personas <= 0:
        raise ValueError("max_personas must be > 0")

    # Read template once outside the loop for efficiency
    with open(template_path, "r", encoding="utf-8") as template_file:
        template = template_file.read()

    with open(context_template_path, "r", encoding="utf-8") as template_file:
        context_template = template_file.read()

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])

        context_fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(context_template)
            if field_name is not None
        }
        missing_fields = context_fields - fieldnames
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Persona context template requires columns not found in "
                f"{csv_path}: {missing}"
            )

        prompt_fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name is not None
        }
        missing_prompt_fields = (prompt_fields - {"persona_context"}) - fieldnames
        if missing_prompt_fields:
            missing = ", ".join(sorted(missing_prompt_fields))
            raise ValueError(
                f"Persona prompt template requires columns not found in "
                f"{csv_path}: {missing}"
            )

        for row in reader:
            # Stop if we've reached max_personas
            if max_personas is not None and len(data) >= max_personas:
                break

            # Filter by name list if provided
            if name_list is not None and row["Name"] not in name_list:
                continue

            persona_context = context_template.format(**row)
            row["prompt"] = template.format(persona_context=persona_context)
            data.append(row)

    return data
