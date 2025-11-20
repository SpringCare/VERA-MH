"""
Judge Runner - High-level functions for batch conversation evaluation.
Contains the main logic extracted from main_judge.py to reduce code duplication.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from .llm_judge import LLMJudge
from .utils import load_rubric_structure
import pandas as pd

# In case this needs to be synced in the meta prompt for the judge
EVALUATION_SEPARATOR = ":"

# Load dimensions from rubric
DIMENSIONS, _ = load_rubric_structure("data/rubric.tsv")

async def batch_evaluate_with_individual_judges(
    conversation_file_paths: List[str],
    rubrics: List[str],
    judge_model: str,
    output_folder: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple conversations, creating a new LLMJudge instance for each conversation.
    
    Args:
        conversation_file_paths: List of conversation file paths
        rubrics: List of rubric names to use
        judge_model: Model to use for judging
        output_folder: Folder to save evaluation results
        limit: Optional limit on number of conversations to evaluate
        
    Returns:
        List of evaluation results
    """
    # Apply limit if specified
    if limit is not None:
        conversation_file_paths = conversation_file_paths[:limit]
    
    results = []
    total_files = len(conversation_file_paths)

    
    for i, conversation_file in enumerate(conversation_file_paths, 1):
        print(f"📄 ({i}/{total_files}) {Path(conversation_file).name}")
      
        
        # Create a new LLMJudge instance for this conversation
        judge = LLMJudge(judge_model=judge_model)
        
        evaluation = await judge.evaluate_conversation_question_flow(
            conversation_file, 
            output_folder=output_folder,
            auto_save=True,
            verbose=True
        )
        # NOTE: We can't guarantee that the evalation always has the same format, so we need to enforce it
        # TODO: maybe move this cleaning to the utils?

            # evaluation shape: {dimension: {"score": str, "reasoning": str, "yes_question_id": str, "yes_reasoning": str}}
        try:
            evaluation_dict = {}
            for dimension, values in evaluation.items():
                # Add score for this dimension
                evaluation_dict[dimension] = values['score']
                # Add yes question ID for this dimension (empty if no Yes answer)
                evaluation_dict[f"{dimension}_yes_question_id"] = values.get('yes_question_id', '')
                # Add yes reasoning for this dimension (empty if no Yes answer)
                evaluation_dict[f"{dimension}_yes_reasoning"] = values.get('yes_reasoning', '')
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            print("the folloing dict is malformed")
            print(evaluation)
            evaluation_dict = {}


        
        results.append({"filename": Path(conversation_file).name, **evaluation_dict, "run_id": Path(conversation_file).parent.name})
    return results


async def judge_conversations(
    judge_model: str,
    conversation_folder: str,
    rubrics: List[str] = ["rubric.csv"],
    output_root: str = "evaluations",
    limit: Optional[int] = None,
    verbose: bool = True,
    output_folder: Optional[str] = None,
    save_aggregated_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Judge conversations in a folder and return results.
    
    Args:
        conversation_folder: Folder containing conversation files
        rubrics: List of rubric names to use
        judge_model: Model to use for judging
        output_folder: Output folder for evaluation results
        limit: Optional limit on number of files to process
        verbose: Whether to print status messages
        
    Returns:
        List of evaluation results
        
    Raises:
        ValueError: Configuration error
        Exception: Other errors
    """
    if output_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        output_folder = f"{output_root}/j_{judge_model}_{timestamp}__{Path(conversation_folder).name}"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Check folder exists
    folder_path = Path(conversation_folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {conversation_folder}")
    
    # Find conversation files
    conversation_files = list(folder_path.glob("*.txt"))
    if not conversation_files:
        raise FileNotFoundError(f"No .txt files found in: {conversation_folder}")
    
    total_found = len(conversation_files)
    
    if limit:
        conversation_files = conversation_files[:limit]
        if verbose:
            print(f"🔍 Found {total_found} files, judging {limit} (debug mode)")
    else:
        if verbose:
            print(f"🔍 Found {total_found} files to judge")
    
    # Convert to strings
    conversation_file_paths = [str(f) for f in conversation_files]

   
    # Run batch evaluation with individual judges
    results = await batch_evaluate_with_individual_judges(
        conversation_file_paths,
        rubrics,
        judge_model,
        output_folder,
        limit=limit
    )
    print(pd.DataFrame(results, columns=["filename"] + DIMENSIONS))
    if save_aggregated_results:
        columns = ["filename", "run_id"] + list(results[0].keys())
        pd.DataFrame(results, columns=columns).to_csv(f"{output_folder}/results.csv", index=False)
    if verbose:
        print(f"✅ Completed {len(results)} evaluations → {output_folder}/")
    
    return results


async def judge_single_conversation(
    judge: LLMJudge, 
    conversation_file: str, 
    rubrics: List[str], 
    output_folder: str
) -> Optional[Dict[str, Any]]:
    """
    Judge a single conversation file.
    
    Args:
        judge: LLMJudge instance
        conversation_file: Path to conversation file
        rubrics: List of rubric names to use
        output_folder: Output folder for results
        
    Returns:
        Evaluation results or None if failed
    """
    if not Path(conversation_file).exists():
        print(f"❌ File not found: {conversation_file}")
        return None
    
    print(f"📄 Judging: {Path(conversation_file).name}")
    
    result = await judge.evaluate_conversation(
        conversation_file, 
        output_folder=output_folder,
        auto_save=True
    )

    print(f"🟢 Done: {Path(conversation_file).name}") 
    return result
        