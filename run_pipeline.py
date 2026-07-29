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
from utils.conversation_layout import resolve_conversation_input
from utils.naming import is_generation_run_folder_basename, is_judge_run_folder_basename
from utils.utils import parse_key_value_list


def _display_path(path: str | os.PathLike[str]) -> str:
    """
    Format *path* for terminal output: relative to cwd when under it, else absolute.

    Keeps pipeline summaries consistent (generate often returns relative paths;
    judge may return absolute paths).
    """
    try:
        resolved = Path(path).resolve()
        cwd = Path.cwd().resolve()
        return str(resolved.relative_to(cwd))
    except (ValueError, OSError):
        return str(Path(path).resolve())


def resolve_pipeline_resume_paths(args: argparse.Namespace) -> None:
    """
    Attach pipeline fields from ``--conversation-output`` / ``--judge-output``:

    - Fresh: ``_pipeline_gen_folder`` = ``--conversation-output`` (parent for new
      ``p_*``); ``_pipeline_resume_generate`` = False; ``_pipeline_judge_output`` =
      None.

    - ``--resume-generate``: ``--conversation-output`` = existing ``p_*`` run folder
      (same as ``generate.py --resume --output``).

    - ``--resume-judge`` alone: ``--judge-output`` = existing ``j_*`` folder; step 1
      resumes the parent ``p_*`` co-located above ``evaluations/``.

    - Both flags: ``--conversation-output`` = ``p_*``; exactly one ``j_*`` under
      ``p_*/evaluations/`` (``--judge-output`` is ignored).
    """
    convo_output_raw = getattr(args, "conversation_output", None) or "output"
    convo_output = os.path.normpath(convo_output_raw)
    judge_output = getattr(args, "judge_output", None)
    if isinstance(judge_output, str) and not judge_output.strip():
        judge_output = None
    resume_gen, resume_judge = args.resume_generate, args.resume_judge

    if not resume_gen and not resume_judge:
        args._pipeline_gen_folder = convo_output
        args._pipeline_resume_generate = False
        args._pipeline_judge_output = None
        return

    if resume_gen and resume_judge:
        convo_output_path = Path(convo_output).resolve()
        if not convo_output_path.is_dir():
            print(
                "error: with --resume-generate and --resume-judge, "
                "--conversation-output must be an existing directory: "
                f"{convo_output!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        base = convo_output_path.name
        if not is_generation_run_folder_basename(base):
            print(
                "error: with both resume flags, --conversation-output must be the "
                "generation run folder itself (basename like p_*__a_*__...), "
                f"not {base!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        eval_parent = convo_output_path / "evaluations"
        if not eval_parent.is_dir():
            print(
                "error: expected evaluations/ under the generation run folder: "
                f"{eval_parent}",
                file=sys.stderr,
            )
            sys.exit(2)
        judge_dirs = sorted(
            p
            for p in eval_parent.iterdir()
            if p.is_dir() and is_judge_run_folder_basename(p.name)
        )
        if len(judge_dirs) == 0:
            print(
                "error: no j_* evaluation folder found under "
                f"{eval_parent}. Create one by judging first, or use only "
                "--resume-generate.",
                file=sys.stderr,
            )
            sys.exit(2)
        if len(judge_dirs) > 1:
            names = ", ".join(d.name for d in judge_dirs)
            print(
                "error: multiple j_* folders under evaluations/; use only one "
                "evaluation run for combined resume, or judge manually. "
                f"Found: {names}",
                file=sys.stderr,
            )
            sys.exit(2)
        args._pipeline_gen_folder = str(convo_output_path)
        args._pipeline_resume_generate = True
        args._pipeline_judge_output = str(judge_dirs[0])
        return

    if resume_gen:
        convo_output_path = Path(convo_output).resolve()
        if not convo_output_path.is_dir():
            print(
                "error: --resume-generate: --conversation-output must be an existing "
                f"directory: {convo_output!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        generate_basename = convo_output_path.name
        if not is_generation_run_folder_basename(generate_basename):
            print(
                "error: --resume-generate: --conversation-output must be the "
                "generation run folder (basename like p_*__a_*__...), not "
                f"{generate_basename!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        args._pipeline_gen_folder = str(convo_output_path)
        args._pipeline_resume_generate = True
        args._pipeline_judge_output = None
        return

    # resume_judge only
    if not judge_output:
        print(
            "error: --resume-judge requires --judge-output pointing at the existing "
            "evaluation run folder (j_*__*).",
            file=sys.stderr,
        )
        sys.exit(2)
    out_path = Path(os.path.normpath(judge_output)).resolve()
    if not out_path.is_dir():
        print(
            "error: --resume-judge: --judge-output must be an existing directory: "
            f"{judge_output!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    judge_basename = out_path.name
    if judge_basename == "evaluations":
        print(
            "error: --resume-judge: --judge-output must be the evaluation run folder "
            "(j_*__*), not .../evaluations/ alone.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not is_judge_run_folder_basename(judge_basename):
        print(
            "error: --resume-judge: --judge-output must be an evaluation run folder "
            f"(basename like j_*__*), not {judge_basename!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    if out_path.parent.name != "evaluations":
        print(
            "error: --resume-judge: expected path .../p_*/evaluations/j_*; "
            f"got parent {out_path.parent.name!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    gen_output_path = out_path.parent.parent
    if not is_generation_run_folder_basename(gen_output_path.name):
        print(
            "error: --resume-judge: could not find generation run folder above "
            "evaluations/ (expected .../p_*/evaluations/j_*). "
            f"Got {gen_output_path.name!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    args._pipeline_gen_folder = str(gen_output_path)
    args._pipeline_resume_generate = True
    args._pipeline_judge_output = str(out_path)


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
        "--conversation-output",
        "-co",
        default="output",
        help=(
            "Passed to generate.py as --output: parent directory for a new p_* run "
            "(default: output/). With --resume-generate (or both resume flags), must "
            "be the existing p_* generation run folder."
        ),
    )
    parser.add_argument(
        "--judge-output",
        "-jo",
        default=None,
        help=(
            "Passed to judge.py as --output: parent directory for a new batch "
            "evaluation (new j_* under this path). Default None for judge.py defaults "
            "(<conversation-output>/evaluations/, or repo evaluations/ for legacy flat "
            "transcript folders). With --resume-judge only, must be the existing "
            "j_* evaluation folder path."
        ),
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
    parser.add_argument(
        "--resume-generate",
        action="store_true",
        help=(
            "Continue generation in an existing p_* folder; set --conversation-output "
            "(-co) to that folder (see generate.py resume validation)."
        ),
    )
    parser.add_argument(
        "--resume-judge",
        action="store_true",
        help=(
            "Continue judging in an existing j_* folder; set --judge-output (-jo) to "
            "that full path (.../evaluations/j_*). With both resume flags, set "
            "--conversation-output (-co) to the p_* folder and ensure a single j_* "
            "exists under evaluations/."
        ),
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
        default=["data/SI/rubric_manifest.json"],
        help=(
            "Rubric bundle manifest(s) to use for evaluation "
            "(default: data/SI/rubric_manifest.json)"
        ),
    )
    parser.add_argument(
        "--rubric-manifest",
        default=None,
        help=(
            "Rubric bundle manifest to load generation personas from (see "
            "docs/architecture.md#rubric-bundle-manifest), instead of the "
            "default data/SI/personas.tsv. Independent of --rubrics: passing the "
            "same manifest to both attaches this run's personas to the rubric "
            "it's evaluated against."
        ),
    )

    # Optional arguments for scoring
    parser.add_argument(
        "--skip-risk-analysis", action="store_true", help="Skip risk-level analysis"
    )
    parser.add_argument(
        "--personas-tsv",
        default="data/SI/personas.tsv",
        help="Path to personas.tsv (default: data/SI/personas.tsv)",
    )

    return parser.parse_args()


async def main():
    """Main entry point for the pipeline runner."""

    args = parse_arguments()
    resolve_pipeline_resume_paths(args)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("VERA-MH Pipeline: Generation → Evaluation → Scoring")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")

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
        output_folder=args._pipeline_gen_folder,
        run_id=args.run_id,
        max_concurrent=args.max_concurrent,
        max_total_words=args.max_total_words,
        max_personas=args.max_personas,
        persona_speaks_first=not args.provider_speaks_first,
        resume=args._pipeline_resume_generate,
        rubric_manifest=args.rubric_manifest,
    )

    print("")
    print(f"✓ Conversations saved to: {_display_path(conversation_folder)}/")
    print("")

    # Validate that Step 1 produced conversation files
    if not os.path.exists(conversation_folder):
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("❌ Pipeline failed at Step 1: Conversation folder not created")
        print("")
        print(f"Expected folder: {_display_path(conversation_folder)}")
        print("")
        print("Troubleshooting:")
        print("  - Check that generate.py returned a valid folder path")
        print("  - Verify file system permissions")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        sys.exit(1)

    # Same nested vs flat transcript resolution as judge.py (legacy runs have .txt
    # at the generation run root; new layout uses .../conversations/*.txt).
    transcripts_dir, _, _ = resolve_conversation_input(conversation_folder)
    conversation_files = []
    if os.path.isdir(transcripts_dir):
        conversation_files = [
            f
            for f in os.listdir(transcripts_dir)
            if f.endswith(".txt") and not f.endswith(".log")
        ]

    if not conversation_files:
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("❌ Pipeline failed at Step 1: No conversations were generated")
        print("")
        print(f"Conversation folder: {_display_path(conversation_folder)}")
        n_listed = (
            len(os.listdir(transcripts_dir)) if os.path.isdir(transcripts_dir) else 0
        )
        print(
            f"Files in transcript directory ({_display_path(transcripts_dir)}): "
            f"{n_listed}"
        )
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
    judge_out = (
        args._pipeline_judge_output
        if args.resume_judge
        else getattr(args, "judge_output", None)
    )
    judge_args = argparse.Namespace(
        conversation=None,  # Not using single conversation mode
        folder=conversation_folder,
        rubrics=args.rubrics,
        judge_model=args.judge_model,
        judge_model_extra_params=args.judge_model_extra_params,
        limit=args.judge_limit,
        output=judge_out,
        max_concurrent=args.judge_max_concurrent,
        per_judge=args.judge_per_judge,
        verbose_workers=args.judge_verbose_workers,
        resume=args.resume_judge,
        debug=args.debug,
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
        print(f"Expected folder: {_display_path(evaluation_folder)}")
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
        print(f"Evaluation folder: {_display_path(evaluation_folder)}")
        print(f"Expected results file: {_display_path(results_csv_path)}")
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
    print(f"✓ Evaluations saved to: {_display_path(evaluation_folder)}/")
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

    scores_subdir = Path(evaluation_folder) / "scores"
    scores_subdir.mkdir(parents=True, exist_ok=True)
    viz_path = scores_subdir / "scores_visualization.png"
    create_visualizations(results, viz_path)

    if not args.skip_risk_analysis:
        risk_results = score_results_by_risk(
            results_csv_path=results_csv,
            personas_tsv_path=args.personas_tsv,
        )
        risk_viz_path = scores_subdir / "scores_by_risk_visualization.png"
        create_risk_level_visualizations(risk_results, risk_viz_path)

    # =========================================================================
    # Final summary
    # =========================================================================
    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✓ Pipeline complete!")
    print("")
    print("Output Locations:")
    gen_p = Path(conversation_folder)
    ev_p = Path(evaluation_folder)
    sc = ev_p / "scores"
    print(f"  Output folder:    {_display_path(gen_p)}/")
    print(f"  Conversations:       {_display_path(gen_p / 'conversations')}/")
    print(f"  Evaluations:       {_display_path(ev_p)}/")
    print(f"  Scores (JSON):     {_display_path(sc / 'scores.json')}")
    if not args.skip_risk_analysis:
        print(f"                     {_display_path(sc / 'scores_by_risk.json')}")
    print(f"  Visualizations:    {_display_path(sc / 'scores_visualization.png')}")
    if not args.skip_risk_analysis:
        v = sc / "scores_by_risk_visualization.png"
        print(f"                     {_display_path(v)}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    asyncio.run(main())
