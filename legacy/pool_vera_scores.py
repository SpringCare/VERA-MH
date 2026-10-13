"""Legacy CLI for pooling: ``uv run python -m legacy.pool_vera_scores <eval> <eval>``.

Deprecated; use ``vera pool --evaluations <eval> <eval>``. Both call
`score.run_pooling`. This wrapper keeps the legacy flags (positional paths,
``--output-dir``, ``--personas-tsv``) and the legacy ``data/SI/personas.tsv``
default, plus ``--extract-from-log``, which only
``legacy/run_recommended_vera_pipeline.sh`` uses to find each suite's
evaluation folder in ``legacy.run_pipeline`` output.
"""

import argparse
import re
import sys
from pathlib import Path

from score import run_pooling

REPO_ROOT = Path(__file__).resolve().parent.parent


def extract_last_evaluation_dir_from_pipeline_log(text: str) -> str:
    """
    Parse captured ``run_pipeline.py`` log text and return the last evaluation folder.

    The pipeline prints a line containing ``Evaluations saved to:`` followed by the
    ``j_*`` evaluation directory (with or without a trailing slash). When the log
    contains multiple runs, the **last** match is returned so callers can chain
    subprocess output from successive pipeline invocations.

    Args:
        text: Full stdout/stderr text (e.g. contents of a temp file used with ``tee``).

    Returns:
        Absolute path to the evaluation directory as a string.

    Raises:
        ValueError: If no matching line exists in *text*.
    """
    matches = re.findall(r"Evaluations saved to:\s*(.+)$", text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(
            "Could not find any 'Evaluations saved to:' line in pipeline log output."
        )
    path = matches[-1].strip().rstrip("/")
    return str(Path(path).resolve())


def _cli_pool(args: argparse.Namespace) -> int:
    """
    Handle the ``pool_vera_scores`` CLI in merge mode (one or more evaluation paths).

    Resolves ``--output-dir`` (defaulting to the repo ``output/`` directory), then
    delegates to `score.run_pooling`.

    Args:
        args: Parsed namespace with ``eval_paths``, ``output_dir``, ``personas_tsv``,
            and ``skip_risk_analysis``.

    Returns:
        Process exit code: ``0`` on success, ``2`` if no evaluation paths were given.
    """
    if len(args.eval_paths) < 1:
        print(
            "error: pass at least one evaluation directory or results.csv",
            file=sys.stderr,
        )
        return 2

    out_parent = Path(args.output_dir).resolve() if args.output_dir else None
    if out_parent is None:
        out_parent = (REPO_ROOT / "output").resolve()
    out_parent.mkdir(parents=True, exist_ok=True)

    personas = Path(args.personas_tsv).resolve() if args.personas_tsv else None
    run_pooling(
        evaluations=args.eval_paths,
        output_parent=out_parent,
        personas_tsv=personas,
        skip_risk_analysis=args.skip_risk_analysis,
    )
    return 0


def _cli_extract(args: argparse.Namespace) -> int:
    """
    Handle ``--extract-from-log``: print the last evaluation directory path.

    Reads the log file as UTF-8 (replacing undecodable bytes), parses it with
    :func:`extract_last_evaluation_dir_from_pipeline_log`, and prints the resolved
    path to stdout for use in shell command substitution.

    Args:
        args: Parsed namespace with ``extract_from_log`` set to the log file path.

    Returns:
        ``0`` on success, ``1`` if no evaluation line was found in the log.
    """
    log_path = Path(args.extract_from_log)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    try:
        path = extract_last_evaluation_dir_from_pipeline_log(text)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(path)
    return 0


def main() -> int:
    """
    CLI entry point: parse arguments and run pool or extract mode.

    In extract mode (``--extract-from-log``), runs :func:`_cli_extract` and exits.
    Otherwise requires at least one evaluation path and runs :func:`_cli_pool`.

    Returns:
        Process exit code from the selected subcommand.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple j_* evaluation runs (any mix of user agents and/or "
            "judge models) into one scored folder, or extract an evaluation path "
            "from run_pipeline log output."
        )
    )
    parser.add_argument(
        "eval_paths",
        nargs="*",
        help="Evaluation directories (j_*) or results.csv paths to merge",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Parent directory for pooled output (a synthetic j_<judge>__p_* folder is "
            "created inside, with pool_metadata.json next to results.csv). "
            "Default: output/ under the repo."
        ),
    )
    parser.add_argument(
        "--personas-tsv",
        default=str(REPO_ROOT / "data" / "SI" / "personas.tsv"),
        help="Personas file for risk-level scoring (default: data/SI/personas.tsv)",
    )
    parser.add_argument(
        "--skip-risk-analysis",
        action="store_true",
        help="Skip risk-level scores and visualization",
    )
    parser.add_argument(
        "--extract-from-log",
        metavar="FILE",
        help="Print the last evaluation directory path found in a pipeline log file",
    )
    args = parser.parse_args()

    if args.extract_from_log:
        return _cli_extract(args)

    if not args.eval_paths:
        parser.error("pass at least one eval path, or use --extract-from-log")
    return _cli_pool(args)


if __name__ == "__main__":
    raise SystemExit(main())
