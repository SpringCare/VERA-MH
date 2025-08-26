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

async def main():
    """Main function for judging conversations."""
    
    parser = argparse.ArgumentParser(description="Judge existing LLM conversations using rubrics")
    
    # Required: source of conversations
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--conversation", "-c", help="Single conversation file to judge")
    source_group.add_argument("--folder", "-f", default="conversations", 
                            help="Folder containing conversation files (default: conversations)")
    
    # Optional parameters
    parser.add_argument("--rubrics", "-r", nargs="+", 
                       default=["rubric"],
                       help="Rubric files to use (default: rubric)")
    
    parser.add_argument("--judge-model", "-j", default="gpt-4",
                       help="Model to use for judging (default: gpt-4). Examples: claude-3-5-sonnet-20241022, gemini-1.5-pro, llama3:8b")
    
    parser.add_argument("--limit", "-l", type=int,
                       help="Limit number of conversations to judge (for debugging)")
    
    parser.add_argument("--output", "-o", default="evaluations",
                       help="Output folder for evaluation results (default: evaluations)")
    
    args = parser.parse_args()
    
    print(f"🎯 LLM Judge | Model: {args.judge_model} | Rubrics: {', '.join(args.rubrics)}")
    
    try:
        # Initialize judge
        1/0 
        judge = LLMJudge(judge_model=args.judge_model)
        
        if args.conversation:
            # Judge single conversation
            await judge_single_conversation(judge, args.conversation, args.rubrics, args.output)
        else:
            # Judge all conversations in folder using the function from judge package
            await judge_conversations(
                conversation_folder=args.folder,
                rubrics=args.rubrics,
                judge_model=args.judge_model,
                output_folder=args.output,
                limit=args.limit,
                verbose=True
            )
            
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("Make sure you have the required API keys configured in .env file")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code or 0)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        exit(1)