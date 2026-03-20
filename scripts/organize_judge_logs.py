#!/usr/bin/env python3
"""Organize jumbled multi-worker judge logs into one log file per evaluation.

Reads one or more judge_*.log files (e.g. from a race where multiple workers
wrote to different files), groups log lines by LLMJudge_<id> (one ID per
evaluation), sorts lines by timestamp within each evaluation, and writes
organized_logs/<scenario>/<conversation_basename>.log.

Usage:
  python3 scripts/organize_judge_logs.py logs/judge_20260224_210740_be_heard.log logs/judge_20260224_210741_be_heard.log
  python3 scripts/organize_judge_logs.py logs/judge_*_be_heard.log
  python3 scripts/organize_judge_logs.py logs/judge_*_be_heard.log -o organized_logs --scenario be_heard
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Line that starts with timestamp and logger name: "2026-02-24 21:07:41,416 - LLMJudge_4686450832 - ..."
LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (LLMJudge_\d+) - "
)


def _scenario_from_log_path(log_path: Path) -> str:
    """Derive scenario from log filename: judge_YYYYMMDD_HHMMSS_<scenario>.log -> scenario."""
    stem = log_path.stem  # e.g. judge_20260224_210740_be_heard
    parts = stem.split("_")
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return "default"


def _parse_log_lines(log_paths: list[Path]):
    """Read all log paths and yield (judge_id, timestamp_str|None, line)."""
    for log_path in log_paths:
        if not log_path.exists():
            continue
        current_judge: str | None = None
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                mo = LOG_LINE_RE.match(line)
                if mo:
                    ts, judge_id = mo.groups()
                    current_judge = judge_id
                    yield (current_judge, ts, line)
                elif current_judge is not None:
                    yield (current_judge, None, line)


def _evaluation_filename_for_judge(
    judge_lines: list[tuple[str | None, str]],
) -> str | None:
    """Extract 'Starting evaluation: <filename>.txt' from lines for this judge."""
    for _ts, line in judge_lines:
        if "Starting evaluation:" in line:
            # message part after " - " (third occurrence: after timestamp, name, level)
            parts = line.split(" - ", 3)
            if len(parts) >= 4:
                msg = parts[3]
                if msg.startswith("Starting evaluation: "):
                    return msg.split("Starting evaluation: ", 1)[1].strip()
    return None


def organize_logs(
    log_paths: list[Path],
    out_dir: Path,
    scenario: str | None = None,
) -> None:
    """Group lines by judge ID, sort by time, write one file per evaluation."""
    if not log_paths:
        return
    scenario = scenario or _scenario_from_log_path(log_paths[0])
    out_subdir = out_dir / scenario
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Group by judge_id, preserving (timestamp, line) for sorting
    by_judge: dict[str, list[tuple[str | None, str]]] = {}
    for judge_id, ts, line in _parse_log_lines(log_paths):
        if judge_id not in by_judge:
            by_judge[judge_id] = []
        by_judge[judge_id].append((ts, line))

    for judge_id, lines in by_judge.items():
        eval_filename = _evaluation_filename_for_judge(lines)
        if not eval_filename or not eval_filename.endswith(".txt"):
            continue
        out_basename = Path(eval_filename).stem + ".log"
        out_path = out_subdir / out_basename

        # Assign block (timestamp) to each line so continuation lines stay with their header
        block_ts: str | None = None
        block_num = 0
        keyed: list[tuple[str, int, str]] = []
        for ts, line in lines:
            if ts is not None:
                block_ts = ts
                block_num += 1
            keyed.append((block_ts or "", block_num, line))
        sorted_lines = sorted(keyed, key=lambda x: (x[0], x[1]))

        with open(out_path, "w", encoding="utf-8") as f:
            for _bts, _bnum, line in sorted_lines:
                f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize jumbled judge logs into one file per evaluation."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="Judge log file(s), e.g. logs/judge_*_be_heard.log",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("organized_logs"),
        help="Output directory (default: organized_logs)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Override scenario subdir (default: from first log filename)",
    )
    args = parser.parse_args()
    organize_logs(args.logs, args.out_dir, args.scenario)


if __name__ == "__main__":
    main()
