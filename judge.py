#!/usr/bin/env python3
"""
Main script for judging existing conversations using the LLM Judge system.
This script is completely separate from conversation generation.
"""

import asyncio
import argparse
from judge import (
    LLMJudge, 
    judge_conversations, 
    judge_single_conversation, 
)

async def main(conversation_folder: str, rubric_folder: str, rubric_file: str, judge_model: str, limit: int, output_root: str):
    """Main function for judging conversations."""
    
    # parser = argparse.ArgumentParser(description="Judge existing LLM conversations using rubrics")
    
    # # Required: source of conversations
    # source_group = parser.add_mutually_exclusive_group(required=True)
    # source_group.add_argument("--conversation", "-c", help="Single conversation file to judge")
    # source_group.add_argument("--folder", "-f", default="conversations", 
    #                         help="Folder containing conversation files (default: conversations)")
    
    # # Optional parameters
    # parser.add_argument("--rubrics", "-r", nargs="+", 
    #                    default=["rubric"],
    #                    help="Rubric files to use (default: rubric)")
    
    # parser.add_argument("--judge-model", "-j", default="gpt-4",
    #                    help="Model to use for judging (default: gpt-4). Examples: claude-3-5-sonnet-20241022, gemini-1.5-pro, llama3:8b")
    
    # parser.add_argument("--limit", "-l", type=int,
    #                    help="Limit number of conversations to judge (for debugging)")
    
    # parser.add_argument("--output", "-o", default="evaluations",
    #                    help="Output folder for evaluation results (default: evaluations)")
    
    # args = parser.parse_args()
    
    # print(f"🎯 LLM Judge | Model: {judge_model} | Rubrics: {', '.join(rubric_file)}")
    

    # Initialize judge
    # judge = LLMJudge(judge_model=judge_model)
    
    # if args.conversation:
    #     # Judge single conversation
    #     await judge_single_conversation(judge, args.conversation, args.rubrics, args.output)
    # else:
        # Judge all conversations in folder using the function from judge package
    await judge_conversations(
            conversation_folder=conversation_folder,
            rubrics=[rubric_file],
            judge_model=judge_model,
            output_root=output_root,
            limit=limit,
            verbose=True
        )
            


if __name__ == "__main__":
    conversation_folder = "conversations/p_gpt_5__a_gpt_5__t30__r5__20250915_102113__{'max_completion_tokens': 5000}"
    
    # TODO: just one file?
    rubric_folder = "data"
    rubric_file = "rubric.csv"
    judge_model = "claude-opus-4-1-20250805"
    limit = None
    output_root = "evaluations"
    
    asyncio.run(main(conversation_folder, rubric_folder, rubric_file, judge_model, limit, output_root))



