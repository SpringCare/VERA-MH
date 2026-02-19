#!/usr/bin/env python3
"""
VERA-MH End-to-End Pipeline Runner (Python version)

This script orchestrates the complete workflow:
  1. Generate conversations (generate.py)
  2. Evaluate them with LLM judge (judge.py)
  3. Score and visualize results (judge/score.py)

It automatically passes the output folder from each step to the next step,
so you don't have to manually copy paths between commands.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from judge.score import (
    print_scores,
    score_results,
    score_results_by_risk,
)
from judge.score_viz import (
    create_risk_level_visualizations,
    create_visualizations,
)
from llm_clients.llm_interface import DEFAULT_START_PROMPT
from utils.utils import parse_key_value_list


def parse_arguments():
    """
    Parse command line arguments and separate them into three groups:
    - Arguments for generate.py
    - Arguments for judge.py
    - Arguments for judge/score.py
    """
    parser = argparse.ArgumentParser(
        description="VERA-MH Pipeline Runner: Generation → Evaluation → Scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --user-agent claude-sonnet-4-5-20250929 \\
           --provider-agent gpt-4o \\
           --runs 2 \\
           --turns 10 \\
           --judge-model claude-sonnet-4-5-20250929 \\
           --max-personas 5
        """,
    )

    # Required arguments for generation
    parser.add_argument(
        "--user-agent",
        "-u",
        required=True,
        help="User/persona model (e.g., claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--provider-agent",
        "-p",
        required=True,
        help="Provider/agent model (e.g., gpt-4o)",
    )
    parser.add_argument(
        "--runs", "-r", type=int, required=True, help="Number of runs per persona"
    )
    parser.add_argument(
        "--turns",
        "-t",
        type=int,
        required=True,
        help="Number of turns per conversation",
    )

    # Required arguments for judge
    parser.add_argument(
        "--judge-model",
        "-j",
        nargs="+",
        required=True,
        help="Judge model(s), format: model or model:count",
    )

    # Optional arguments for generation
    parser.add_argument(
        "--user-agent-extra-params",
        "-uep",
        help="Extra params for user agent (e.g., temperature=0.7)",
        type=parse_key_value_list,
        default={},
    )
    parser.add_argument(
        "--provider-agent-extra-params",
        "-pep",
        help="Extra params for provider agent (e.g., temperature=0.5)",
        type=parse_key_value_list,
        default={},
    )
    parser.add_argument(
        "--max-total-words",
        "-w",
        type=int,
        help="Maximum total words per conversation",
    )
    parser.add_argument(
        "--max-concurrent", type=int, help="Maximum concurrent conversations"
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        help="Maximum number of personas to load (for testing)",
    )
    parser.add_argument(
        "--folder-name", "-f", help="Custom folder name for conversations"
    )
    parser.add_argument(
        "--run-id",
        "-i",
        help="Custom run ID for conversation folder (default: timestamp)",
    )
    parser.add_argument(
        "-psf",
        "--provider-speaks-first",
        action="store_true",
        help=(
            "Provider speaks first. max_turns adjusted so provider has last turn. "
            "Default: persona speaks first."
        ),
    )
    parser.add_argument(
        "-usm",
        "--user-first-message",
        help="Static first message from user-agent (no LLM call for first turn).",
        default=None,
    )
    parser.add_argument(
        "-usp",
        "--user-start-prompt",
        help="Prompt sent to user-agent LLM when starting conversation (first turn).",
        default=DEFAULT_START_PROMPT,
    )
    parser.add_argument(
        "-pfm",
        "--provider-first-message",
        help="Static first message from provider (no LLM call for first turn).",
        default=None,
    )
    parser.add_argument(
        "-psp",
        "--provider-start-prompt",
        help="Prompt sent to provider LLM when starting conversation (first turn).",
        default=DEFAULT_START_PROMPT,
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging for generation"
    )

    # Optional arguments for judge
    parser.add_argument(
        "--judge-model-extra-params",
        "-jep",
        help="Extra params for judge model",
        type=parse_key_value_list,
        default={},
    )
    parser.add_argument(
        "--judge-max-concurrent", type=int, help="Maximum concurrent judge workers"
    )
    parser.add_argument(
        "--judge-per-judge",
        action="store_true",
        help="Apply concurrency limit per judge",
    )
    parser.add_argument(
        "--judge-limit", type=int, help="Limit conversations to judge (for testing)"
    )
    parser.add_argument(
        "--judge-verbose-workers",
        action="store_true",
        help="Enable verbose worker logging",
    )
    parser.add_argument(
        "--rubrics",
        nargs="+",
        default=["data/rubric.tsv"],
        help="Rubric file(s) to use for evaluation (default: data/rubric.tsv)",
    )
    parser.add_argument(
        "--judge-output",
        default="evaluations",
        help="Output folder for evaluation results (default: evaluations)",
    )

    # Optional arguments for scoring
    parser.add_argument(
        "--skip-risk-analysis", action="store_true", help="Skip risk-level analysis"
    )
    parser.add_argument(
        "--personas-tsv",
        default="data/personas.tsv",
        help="Path to personas.tsv (default: data/personas.tsv)",
    )

    return parser.parse_args()


async def main():
    """Main entry point for the pipeline runner."""

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("VERA-MH Pipeline: Generation → Evaluation → Scoring")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")

    # Parse command line arguments
    args = parse_arguments()

    # Import generate and judge main functions
    # We import here to avoid circular dependencies and to allow --debug flag to be set
    # Import judge.py main function
    # (note: judge.py is a module file, judge/ is a package)
    import importlib.util

    from generate import main as generate_main

    spec = importlib.util.spec_from_file_location("judge_script", "judge.py")
    judge_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge_script)
    judge_main = judge_script.main

    # Set debug mode if flag is provided
    if args.debug:
        from utils.debug import set_debug

        set_debug(True)

    # =========================================================================
    # Step 1: Generate conversations
    # =========================================================================
    print("▶ Step 1/3: Generating conversations...")

    # Build model configs for generation
    persona_model_config = {
        "model": args.user_agent,
        **args.user_agent_extra_params,
    }
    if args.user_first_message is not None:
        persona_model_config["first_message"] = args.user_first_message
    persona_model_config["start_prompt"] = args.user_start_prompt

    agent_model_config = {
        "model": args.provider_agent,
        "name": args.provider_agent,
        **args.provider_agent_extra_params,
    }
    if args.provider_first_message is not None:
        agent_model_config["first_message"] = args.provider_first_message
    agent_model_config["start_prompt"] = args.provider_start_prompt

    # Call generate.py's main function directly
    _, conversation_folder = await generate_main(
        persona_model_config=persona_model_config,
        agent_model_config=agent_model_config,
        max_turns=args.turns,
        runs_per_prompt=args.runs,
        persona_extra_run_params={
            k: v
            for k, v in persona_model_config.items()
            if k not in ["model", "model_name", "name", "temperature", "max_tokens"]
        },
        agent_extra_run_params={
            k: v
            for k, v in agent_model_config.items()
            if k not in ["model", "model_name", "name", "temperature", "max_tokens"]
        },
        folder_name=args.folder_name,
        run_id=args.run_id,
        max_concurrent=args.max_concurrent,
        max_total_words=args.max_total_words,
        max_personas=args.max_personas,
        persona_speaks_first=not args.provider_speaks_first,
    )

    print("")
    print(f"✓ Conversations saved to: {conversation_folder}/")
    print("")

    # Validate that Step 1 produced conversation files
    if not os.path.exists(conversation_folder):
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("❌ Pipeline failed at Step 1: Conversation folder not created")
        print("")
        print(f"Expected folder: {conversation_folder}")
        print("")
        print("Troubleshooting:")
        print("  - Check that generate.py returned a valid folder path")
        print("  - Verify file system permissions")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        sys.exit(1)

    # Count conversation files (exclude log files)
    conversation_files = [
        f
        for f in os.listdir(conversation_folder)
        if f.endswith(".txt") and not f.endswith(".log")
    ]

    if not conversation_files:
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("❌ Pipeline failed at Step 1: No conversations were generated")
        print("")
        print(f"Conversation folder: {conversation_folder}")
        print(f"Files in folder: {len(os.listdir(conversation_folder))}")
        print("")
        print("Possible causes:")
        print(
            "  1. Invalid model name (check that the model exists in the "
            "provider's API)"
        )
        print("  2. API authentication issues (check your API keys in .env)")
        print("  3. API rate limits or quota exceeded")
        print("  4. Network connectivity issues")
        print("")
        print("Troubleshooting:")
        print("  - Check files in the conversation folder for error messages")
        print("  - Look for API error responses in the output")
        print("  - Verify model names are valid for your provider")
        print("  - Run generate.py separately to isolate the issue")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        sys.exit(1)

    print(f"✓ Validated: {len(conversation_files)} conversation files generated")
    print("")

    # =========================================================================
    # Step 2: Evaluate conversations with LLM judge
    # =========================================================================
    print("▶ Step 2/3: Evaluating conversations...")

    # Build argparse.Namespace for judge.py's main function
    judge_args = argparse.Namespace(
        conversation=None,  # Not using single conversation mode
        folder=conversation_folder,
        rubrics=args.rubrics,
        judge_model=args.judge_model,
        judge_model_extra_params=args.judge_model_extra_params,
        limit=args.judge_limit,
        output=args.judge_output,
        max_concurrent=args.judge_max_concurrent,
        per_judge=args.judge_per_judge,
        verbose_workers=args.judge_verbose_workers,
    )

    # Call judge.py's main function directly
    evaluation_folder = await judge_main(judge_args)

    if not evaluation_folder:
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("❌ Pipeline failed at Step 2: Judge did not return an evaluation folder")
        print("")
        print("Troubleshooting:")
        print("  - Check error messages from the judge evaluation above")
        print("  - Run judge.py separately to isolate the issue")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        sys.exit(1)

    # Validate that Step 2 produced evaluation results
    if not os.path.exists(evaluation_folder):
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("❌ Pipeline failed at Step 2: Evaluation folder not created")
        print("")
        print(f"Expected folder: {evaluation_folder}")
        print("")
        print("Troubleshooting:")
        print("  - Check that judge.py returned a valid folder path")
        print("  - Verify file system permissions")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        sys.exit(1)

    # Check for results.csv file
    results_csv_path = os.path.join(evaluation_folder, "results.csv")
    if not os.path.exists(results_csv_path):
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("❌ Pipeline failed at Step 2: No evaluation results were generated")
        print("")
        print(f"Evaluation folder: {evaluation_folder}")
        print(f"Expected results file: {results_csv_path}")
        print("")

        # Check if folder is empty
        folder_files = (
            os.listdir(evaluation_folder) if os.path.exists(evaluation_folder) else []
        )
        print(f"Files in evaluation folder: {len(folder_files)}")
        if folder_files:
            print("  Found: " + ", ".join(folder_files[:5]))
            if len(folder_files) > 5:
                print(f"  ... and {len(folder_files) - 5} more")

        print("")
        print("Possible causes:")
        print("  1. All evaluations failed (check judge model name and API access)")
        print("  2. Invalid judge model name")
        print("  3. Judge API authentication issues")
        print(
            "  4. Conversation files from Step 1 contained errors instead of "
            "conversations"
        )
        print("")
        print("Troubleshooting:")
        print("  - Check the conversation files from Step 1 for API error messages")
        print("  - Look for judge evaluation errors in the output above")
        print("  - Verify judge model name is valid")
        print(
            "  - Run judge.py separately on the conversation folder to isolate the "
            "issue"
        )
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        sys.exit(1)

    print("")
    print(f"✓ Evaluations saved to: {evaluation_folder}/")
    print("✓ Validated: results.csv exists with evaluation data")
    print("")

    # =========================================================================
    # Step 3: Score results and create visualizations
    # =========================================================================
    print("▶ Step 3/3: Scoring and visualizing results...")

    # Build paths for scoring
    results_csv = os.path.join(evaluation_folder, "results.csv")

    # Call score_results for standard analysis
    results = score_results(results_csv_path=results_csv)
    print_scores(results)

    # Create standard visualizations
    viz_path = Path(evaluation_folder) / "scores_visualization.png"
    create_visualizations(results, viz_path)

    # Perform risk-level analysis unless skipped
    if not args.skip_risk_analysis:
        risk_results = score_results_by_risk(
            results_csv_path=results_csv,
            personas_tsv_path=args.personas_tsv,
        )
        risk_viz_path = Path(evaluation_folder) / "scores_by_risk_visualization.png"
        create_risk_level_visualizations(risk_results, risk_viz_path)

    # =========================================================================
    # Final summary
    # =========================================================================
    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✓ Pipeline complete!")
    print("")
    print("Output Locations:")
    print(f"  Conversations:     {conversation_folder}/")
    print(f"  Evaluations:       {evaluation_folder}/")
    print(f"  Scores (JSON):     {evaluation_folder}/scores.json")
    if not args.skip_risk_analysis:
        print(f"                     {evaluation_folder}/scores_by_risk.json")
    print(f"  Visualizations:    {evaluation_folder}/scores_visualization.png")
    if not args.skip_risk_analysis:
        print(
            f"                     {evaluation_folder}/scores_by_risk_visualization.png"
        )
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    asyncio.run(main())
