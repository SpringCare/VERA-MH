#!/usr/bin/env python3
"""
Judge Runner - High-level functions for batch conversation evaluation.
Contains the main logic extracted from main_judge.py to reduce code duplication.
"""

import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from .llm_judge import LLMJudge


async def judge_conversations(
    conversation_folder: str = "conversations",
    rubrics: List[str] = None,
    judge_model: str = "gpt-4",
    output_folder: str = "evaluations",
    limit: Optional[int] = None,
    verbose: bool = True
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
    if rubrics is None:
        rubrics = ["helpfulness", "safety", "communication"]
    
    if verbose:
        print(f"🎯 LLM Judge | Model: {judge_model} | Rubrics: {', '.join(rubrics)}")
    
    # Initialize judge
    judge = LLMJudge(judge_model=judge_model)
    
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
    
    # Run batch evaluation
    results = await judge.batch_evaluate(
        conversation_file_paths,
        rubrics,
        output_folder=output_folder,
        limit=limit
    )
    
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
    
    try:
        result = await judge.evaluate_conversation(
            conversation_file, 
            rubrics, 
            output_folder=output_folder,
            auto_save=True
        )
        
        print(f"✅ Done")
        print_evaluation_summary(result)
        return result
        
    except Exception as e:
        print(f"❌ Failed to judge conversation: {e}")
        return None


async def judge_conversation_folder(
    judge: LLMJudge, 
    folder: str, 
    rubrics: List[str], 
    output_folder: str, 
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Judge all conversations in a folder.
    
    Args:
        judge: LLMJudge instance
        folder: Folder containing conversation files
        rubrics: List of rubric names to use
        output_folder: Output folder for results
        limit: Optional limit on number of files
        
    Returns:
        List of evaluation results
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder}")
        return []
    
    # Find all conversation files
    conversation_files = list(folder_path.glob("*.txt"))
    if not conversation_files:
        print(f"❌ No .txt files found in: {folder}")
        return []
    
    total_found = len(conversation_files)
    
    if limit:
        conversation_files = conversation_files[:limit]
        print(f"🔍 Found {total_found} files, judging {limit} (debug mode)")
    else:
        print(f"🔍 Found {total_found} files to judge")
    
    # Convert to strings
    conversation_file_paths = [str(f) for f in conversation_files]
    
    try:
        results = await judge.batch_evaluate(
            conversation_file_paths,
            rubrics,
            output_folder=output_folder,
            limit=limit
        )
        
        print(f"✅ Completed {len(results)} evaluations → {output_folder}/")
        return results
            
    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}")
        return []


def print_evaluation_summary(result: Dict[str, Any]) -> None:
    """
    Print a brief summary of evaluation results.
    
    Args:
        result: Evaluation results dictionary
    """
    if "evaluations" not in result:
        return
        
    for rubric_id, evaluation in result["evaluations"].items():
        if "error" in evaluation:
            print(f"  ❌ {evaluation.get('rubric_name', rubric_id)}")
            continue
            
        if "scores" not in evaluation:
            continue
        
        scores = evaluation['scores']
        if scores:
            avg_score = sum(dim['score'] for dim in scores.values() if 'score' in dim) / len(scores)
            print(f"  📊 {evaluation.get('rubric_name', rubric_id)}: {avg_score:.1f}/5")