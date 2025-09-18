#!/usr/bin/env python3
"""
Script to evaluate conversations using the LLM Judge system.
"""

import asyncio
import argparse
from pathlib import Path
from .llm_judge import LLMJudge

async def evaluate_single_conversation(
    conversation_file: str, 
    judge_model: str 
):
    """Evaluate a single conversation and print results."""
    
    judge = LLMJudge(judge_model=judge_model)
    
    try:
        print(f"Evaluating: {conversation_file}")
        
        _ = await judge.evaluate_conversation(conversation_file)
        
        # Print brief results
        # print_evaluation_results(evaluation)
        
    except Exception as e:
        print(f"Error: {e}")

# def print_evaluation_results(evaluation: dict):
#     """Print evaluation results in a readable format."""
    
#     for rubric_id, result in evaluation["evaluations"].items():
#         if "error" in result:
#             print(f"❌ {result.get('rubric_name', rubric_id)}: Error")
#             continue
        
#         if "scores" in result:
#             scores = result["scores"]
#             avg_score = sum(dim['score'] for dim in scores.values() if 'score' in dim) / len(scores)
#             print(f"📊 {result.get('rubric_name', rubric_id)}: {avg_score:.1f}/5")

async def batch_evaluate_conversations(
    conversation_folder: str, 
    rubric_files: list, 
    judge_model: str,
    limit: int = None
):
    """Evaluate all conversations in a folder."""
    
    conversation_path = Path(conversation_folder)
    if not conversation_path.exists():
        print(f"Conversation folder not found: {conversation_folder}")
        return
    
    # Find all conversation files
    conversation_files = list(conversation_path.glob("*.txt"))
    if not conversation_files:
        print(f"No conversation files found in: {conversation_folder}")
        return
    
    total_found = len(conversation_files)
    files_to_process = min(limit or total_found, total_found)
    print(f"Found {total_found} files, processing {files_to_process}")
    
    judge = LLMJudge(judge_model=judge_model)
    
    # Convert Path objects to strings
    conversation_file_paths = [str(f) for f in conversation_files]
    
    try:
        results = await judge.batch_evaluate(
            conversation_file_paths, 
            rubric_files,
            output_folder="evaluations",
            limit=limit
        )
        
        print(f"✅ Completed {len(results)} evaluations → evaluations/")
        
        # Create summary report
        # create_summary_report(results, "evaluations/summary_report.json")
        
    except Exception as e:
        print(f"Error in batch evaluation: {e}")

# def create_summary_report(evaluations: list, output_file: str):
#     """Create a summary report of all evaluations."""
    
#     summary = {
#         "total_conversations": len(evaluations),
#         "judge_model": evaluations[0]["judge_model"] if evaluations else "unknown",
#         "rubric_summaries": {},
#         "conversation_summaries": []
#     }
    
#     # Aggregate scores by rubric and dimension
#     rubric_scores = {}
    
#     for evaluation in evaluations:
#         conv_summary = {
#             "conversation_file": evaluation["conversation_file"],
#             "rubric_scores": {}
#         }
        
#         for rubric_id, result in evaluation["evaluations"].items():
#             if "scores" not in result:
#                 continue
            
#             if rubric_id not in rubric_scores:
#                 rubric_scores[rubric_id] = {}
            
#             conv_rubric_scores = {}
            
#             for dim_id, dim_result in result["scores"].items():
#                 if "score" in dim_result:
#                     score = dim_result["score"]
#                     conv_rubric_scores[dim_id] = score
                    
#                     if dim_id not in rubric_scores[rubric_id]:
#                         rubric_scores[rubric_id][dim_id] = []
                    
#                     rubric_scores[rubric_id][dim_id].append(score)
            
#             conv_summary["rubric_scores"][rubric_id] = conv_rubric_scores
        
#         summary["conversation_summaries"].append(conv_summary)
    
#     # Calculate averages
#     for rubric_id, dimensions in rubric_scores.items():
#         rubric_summary = {}
#         for dim_id, scores in dimensions.items():
#             if scores:
#                 rubric_summary[dim_id] = {
#                     "average": round(sum(scores) / len(scores), 2),
#                     "min": min(scores),
#                     "max": max(scores),
#                     "count": len(scores)
#                 }
#         summary["rubric_summaries"][rubric_id] = rubric_summary
    
#     # Save summary
#     Path(output_file).parent.mkdir(parents=True, exist_ok=True)
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(summary, f, indent=2, ensure_ascii=False)
    

async def main():
    """Main function with CLI argument parsing."""
    
    parser = argparse.ArgumentParser(description="Evaluate LLM conversations using rubrics")
    parser.add_argument("--conversation", "-c", help="Single conversation file to evaluate")
    parser.add_argument("--folder", "-f", help="Folder containing conversation files")
    parser.add_argument("--rubrics", "-r", nargs="+", default=["helpfulness", "safety", "communication"],
                       help="Rubric files to use (without .json extension)")
    parser.add_argument("--judge-model", "-j", default="gpt-4",
                       help="Model to use for judging (default: gpt-4)")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of conversations to evaluate (for debugging)")
    
    args = parser.parse_args()
    
    if not args.conversation and not args.folder:
        print("Please specify either --conversation or --folder")
        return
    
    if args.conversation:
        await evaluate_single_conversation(args.conversation, args.rubrics, args.judge_model)
    elif args.folder:
        await batch_evaluate_conversations(args.folder, args.rubrics, args.judge_model, args.limit)

if __name__ == "__main__":
    asyncio.run(main())