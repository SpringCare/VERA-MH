#!/usr/bin/env python3
"""Walk a tree for judge evaluation dirs (each containing results.csv), write a CSV.

Columns: provider model slug from ``__a_<model>__t...`` in the folder name (agent /
provider LLM in pipeline terms), and the absolute path to that directory.

Rows are only included when ``canonical_evaluation_dir_name`` accepts the path (skips
non-eval folders that happen to contain a ``results.csv``).

Usage (from repo root)::

    python3 scripts/create_eval_comparison_csv.py /path/to/root -o out.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from judge.utils import canonical_evaluation_dir_name  # noqa: E402

# Agent / provider model: ...__a_<slug>__t<turns>...
AGENT_IN_EVAL_DIR = re.compile(r"__a_(?P<model>.+?)__t\d+")


def extract_provider_model_slug(dir_name: str) -> str | None:
    m = AGENT_IN_EVAL_DIR.search(dir_name)
    return m.group("model") if m else None


def eval_dirs_with_results(root: Path) -> list[Path]:
    return sorted(
        {p.parent.resolve() for p in root.rglob("results.csv") if p.is_file()}
    )


def collect_rows(root: Path) -> list[tuple[str, str]]:
    """Return list of (provider_model_slug, absolute_path)."""
    rows: list[tuple[str, str]] = []
    for eval_dir in eval_dirs_with_results(root):
        if (slug := canonical_evaluation_dir_name(eval_dir)) is None:
            continue
        slug = (
            slug.replace("claude-sonnet-4-5-20250929", "sonnet45")
            .replace("claude_sonnet_4_6", "sonnet46")
            .replace("claude_sonnet_4_20250514", "sonnet4")
            .replace("claude_opus_4_20250514", "opus4")
            .replace("claude_opus_4_5_20251101", "opus45")
            .replace("claude_opus_4_5", "opus45")
            .replace("opus_4_6", "opus46")
            .replace("gpt-4o", "gpt4o")
            .replace("gpt_4o", "gpt4o")
            .replace("gpt_5_2", "gpt52")
            .replace("gemini_2_5_flash", "gemini25")
            .replace("gemini_3_1", "gemini31")
            .replace("azure_grok_3", "grok3")
            .replace("grok_4", "grok4")
        )
        # elif (slug := extract_provider_model_slug(eval_dir.name)) is None:
        #     continue
        # if slug.startswith("j_gpt"):
        rows.append((slug, str(eval_dir)))
    # rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir",
        type=Path,
        nargs="?",
        default=Path(
            "/Users/josh.gieringer/Desktop/HEOR AIM 3/GPT4o + S4.5 Evaluations"
        ),
        help="Root directory to search (default: HEOR GPT4o+S4.5 example path)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path (default: write to stdout)",
    )
    args = parser.parse_args(argv)

    root = args.root_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    rows = collect_rows(root)
    fieldnames = ["Provider Model", "Path"]

    if args.output:
        out_fp = args.output.expanduser().open("w", newline="", encoding="utf-8")
    else:
        out_fp = sys.stdout

    try:
        w = csv.DictWriter(out_fp, fieldnames=fieldnames)
        w.writeheader()
        for slug, path_str in rows:
            w.writerow({"Provider Model": slug, "Path": path_str})
    finally:
        if args.output:
            out_fp.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
