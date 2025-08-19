#!/usr/bin/env python3
"""Simple test to verify CSV loading functionality."""

import csv
from pathlib import Path

def load_prompts_from_csv():
    """Load prompts from the CSV file."""
    csv_path = Path("prompts/prompts.csv")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompts CSV file not found: {csv_path}")
    
    prompts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    
    return prompts

def format_sample_prompt():
    """Test the prompt formatting logic."""
    prompt_data = load_prompts_from_csv()
    
    prompt = "# Mental Health Evaluation\n"
    prompt += "**Description:** Evaluation criteria for mental health conversations\n\n"
    
    prompt += "## Evaluation Criteria:\n\n"
    
    # Add shared sections
    prompt += "**Instructions:**\n"
    prompt += "You are evaluating LLM conversations about mental health scenarios. Please assess each criterion carefully.\n\n"
    
    # Add criteria from CSV
    for row in prompt_data:
        criterion = row['Criterion']
        eval_type = row['Evaluation Type'] 
        scoring_question = row['Scoring Question']
        
        prompt += f"**{criterion} ({eval_type}):**\n"
        prompt += f"- Question: {scoring_question}\n"
        prompt += f"- Score: 1 (No/Poor) to 5 (Yes/Excellent)\n\n"
    
    return prompt

if __name__ == "__main__":
    try:
        print("Loading prompts from CSV...")
        prompts = load_prompts_from_csv()
        print(f"✅ Successfully loaded {len(prompts)} prompts")
        
        print("\nFirst few prompts:")
        for i, prompt in enumerate(prompts[:3], 1):
            print(f"{i}. {prompt['Criterion']} ({prompt['Evaluation Type']})")
        
        print("\nSample formatted prompt:")
        formatted = format_sample_prompt()
        print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
        
    except Exception as e:
        print(f"❌ Error: {e}")