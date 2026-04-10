#!/usr/bin/env python3
"""
Score how realistic each simulated mental-health conversation feels, with primary
weight on the user/persona side. One judge LLM call per transcript (0–10 scale).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic import BaseModel, Field  # noqa: E402

import llm_clients.config  # noqa: F401 E402
from judge.utils import extract_persona_name_from_filename  # noqa: E402
from llm_clients.llm_factory import LLMFactory  # noqa: E402
from llm_clients.llm_interface import JudgeLLM  # noqa: E402
from utils.role_reversal_transcript import (  # noqa: E402
    conversation_relpath,
    discover_conversation_txt_files,
    parse_user_and_provider_from_run_folder,
)

USER_REALISM_SYSTEM = """You evaluate transcripts from a simulation where one \
language model plays a human user seeking mental health-related support, and \
another model plays a helpful chatbot/provider.

The transcript uses speaker labels: lines starting with `user:` are the simulated \
human seeking help; lines starting with `chatbot:` (or `assistant:`) are the \
provider. Continuation lines belong to the same turn until the next labeled line.

Your task: assign a single realism score for the **whole conversation**, from the \
perspective of whether this could pass as a real human chatting with an AI \
assistant about distress or mental health.

**Primary weight on the user's messages:** Judge whether the persona sounds like a \
plausible help-seeker—natural wording, emotional and situational coherence, \
appropriate stance (seeking support or information), and avoiding obvious \
"LLM playing therapist" or generic counselor voice unless the persona context \
would allow it. Penalize clearly robotic repetition, unnatural politeness, or \
role confusion (e.g., the user counseling the bot as if the bot were the patient).

The provider's replies can slightly affect plausibility (e.g., wildly mismatched \
tone), but **do not** let long boilerplate provider answers dominate your score; \
focus on whether the **user side** could convince you they are human.

**Scale (integer 0–10):**
- 0–2: Clearly artificial; user does not read as a real help-seeker.
- 3–4: Weak; frequent tells of simulation or unnatural user voice.
- 5–6: Mixed; some believable stretches but notable issues.
- 7–8: Mostly believable; minor issues only.
- 9–10: Highly plausible; you could be convinced this is a real human user \
talking to a chatbot (minor imperfections allowed).

Use the full transcript. Return structured fields only (score, reasoning, \
focus_notes)."""

USER_REALISM_USER_TEMPLATE = """Transcript (UTF-8):

---
{transcript}
---

Return your score (0–10), brief reasoning, and focus_notes noting what most \
drove the score (especially regarding the user's voice)."""


class UserRealismJudgment(BaseModel):
    score: int = Field(
        ge=0,
        le=10,
        description="Realism 0–10; primary weight on simulated user",
    )
    reasoning: str = Field(description="Brief justification for the score")
    focus_notes: str = Field(
        description=(
            "What dominated the score (user vs provider; key strengths/weaknesses)"
        )
    )


def build_row(
    *,
    conversation_relpath_str: str,
    conversation_filename: str,
    run_folder: str,
    user_model_slug: str,
    provider_model_slug: str,
    persona_name: str,
    judge_model: str,
    score: str,
    reasoning: str,
    focus_notes: str,
    error: str,
) -> Dict[str, str]:
    return {
        "conversation_relpath": conversation_relpath_str,
        "conversation_filename": conversation_filename,
        "run_folder": run_folder,
        "user_model_slug": user_model_slug,
        "provider_model_slug": provider_model_slug,
        "persona_name": persona_name,
        "judge_model": judge_model,
        "score": score,
        "reasoning": reasoning,
        "focus_notes": focus_notes,
        "error": error,
    }


CSV_FIELDNAMES = list(
    build_row(
        conversation_relpath_str="",
        conversation_filename="",
        run_folder="",
        user_model_slug="",
        provider_model_slug="",
        persona_name="",
        judge_model="",
        score="",
        reasoning="",
        focus_notes="",
        error="",
    ).keys()
)


async def score_one(
    llm: JudgeLLM,
    sem: asyncio.Semaphore,
    transcript: str,
) -> tuple[str, str, str, str]:
    """Returns (score_str, reasoning, focus_notes, error)."""
    prompt = USER_REALISM_USER_TEMPLATE.format(transcript=transcript)
    async with sem:
        try:
            out = await llm.generate_structured_response(prompt, UserRealismJudgment)
            return (
                str(out.score),
                out.reasoning.strip(),
                out.focus_notes.strip(),
                "",
            )
        except Exception as e:
            return ("", "", "", str(e))


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


async def process_file(
    path: Path,
    root: Path,
    llm: JudgeLLM,
    sem: asyncio.Semaphore,
    judge_model: str,
) -> Dict[str, str]:
    run_folder = path.parent.name
    rel = conversation_relpath(path, root)
    persona = extract_persona_name_from_filename(path.name) or ""

    parsed = parse_user_and_provider_from_run_folder(run_folder)
    if parsed is None:
        _log_stderr(f"skip (folder name): {path}")
        return build_row(
            conversation_relpath_str=rel,
            conversation_filename=path.name,
            run_folder=run_folder,
            user_model_slug="",
            provider_model_slug="",
            persona_name=persona,
            judge_model=judge_model,
            score="",
            reasoning="",
            focus_notes="",
            error="unrecognized run folder name",
        )

    user_slug, provider_slug = parsed
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return build_row(
            conversation_relpath_str=rel,
            conversation_filename=path.name,
            run_folder=run_folder,
            user_model_slug=user_slug,
            provider_model_slug=provider_slug,
            persona_name=persona,
            judge_model=judge_model,
            score="",
            reasoning="",
            focus_notes="",
            error="empty transcript",
        )

    score, reason, focus, err = await score_one(llm, sem, text)
    return build_row(
        conversation_relpath_str=rel,
        conversation_filename=path.name,
        run_folder=run_folder,
        user_model_slug=user_slug,
        provider_model_slug=provider_slug,
        persona_name=persona,
        judge_model=judge_model,
        score=score if not err else "",
        reasoning=reason if not err else "",
        focus_notes=focus if not err else "",
        error=err,
    )


async def run_async(args: argparse.Namespace) -> List[Dict[str, str]]:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    files = discover_conversation_txt_files(root)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    extra: dict = {}
    if args.temperature is not None:
        extra["temperature"] = args.temperature

    llm = LLMFactory.create_judge_llm(
        model_name=args.model,
        name="user-realism-judge",
        system_prompt=USER_REALISM_SYSTEM,
        **extra,
    )

    sem = asyncio.Semaphore(args.max_concurrent)
    tasks = [
        asyncio.create_task(process_file(p, root, llm, sem, args.model)) for p in files
    ]
    if not tasks:
        return []
    return await asyncio.gather(*tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score simulated user realism (0–10) for each conversation .txt "
            "under run folders"
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root folder containing generated run subfolders with .txt transcripts",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output CSV path",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="claude-sonnet-4-5-20250929",
        help="Judge model (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=6,
        help="Max concurrent API calls (default: 6)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature (default: 0)",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Process at most this many conversation files (debug)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = asyncio.run(run_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    _log_stderr(f"Wrote {len(rows)} rows to {args.output} ({n_err} with errors)")


if __name__ == "__main__":
    main()
