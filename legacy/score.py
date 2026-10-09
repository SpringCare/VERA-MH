"""Legacy CLI for scoring: ``uv run python -m legacy.score -r <results.csv>``.

Deprecated; use ``vera score``. Its only differences from ``vera score`` are
the flag names and the ``--personas-tsv`` default, documented in
docs/legacy-scripts.md.
"""

import argparse
import sys

from score import run_scoring


def main():
    """Legacy CLI entry point: parse arguments, then call `run_scoring`."""
    parser = argparse.ArgumentParser(
        description=(
            "Score evaluation results from judge/runner.py output "
            "and generate visualizations"
        )
    )

    parser.add_argument(
        "--results-csv",
        "-r",
        required=True,
        help="Path to results.csv file from judge evaluation",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        default=None,
        help="Path to save JSON output "
        "(default: scores/scores.json next to results.csv)",
    )
    parser.add_argument(
        "--personas-tsv",
        "-p",
        default="data/SI/personas.tsv",
        help=(
            "Path to personas.tsv file for risk-level analysis "
            "(default: data/SI/personas.tsv)"
        ),
    )
    parser.add_argument(
        "--skip-risk-analysis",
        action="store_true",
        help="Skip risk-level analysis and visualization",
    )

    args = parser.parse_args()

    try:
        run_scoring(
            results_csv=args.results_csv,
            output_json=args.output_json,
            personas_tsv=args.personas_tsv,
            skip_risk_analysis=args.skip_risk_analysis,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
