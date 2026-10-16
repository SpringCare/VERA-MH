"""Record a small real pipeline run for the replay test.

Makes real API calls (a few cents with the default models) and rewrites
`calls.jsonl` and `expected/`. Run it only when a change is meant to alter what
the models are sent, then review the diff: it shows exactly which prompts and
outputs changed. See docs/safe-changes.md.

    uv run --env-file <your .env> python tests/fixtures/replay/record.py
"""

import csv
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))  # so it runs as a plain script

import vera  # noqa: E402
from llm_clients.llm_factory import LLMFactory  # noqa: E402
from tests.mocks.replay_llm import recording_llm_factory  # noqa: E402
from vera_cli import config as cli_config  # noqa: E402

RECORDING = HERE / "calls.jsonl"
EXPECTED = HERE / "expected"

# Small and cheap on purpose: enough to exercise every kind of call (user,
# chatbot, judge) and every stage, not to say anything about the models.
CHATBOT = "gpt-5.4-nano"
USER = "gpt-5.4-nano"
JUDGE = "gpt-5.4-nano"
PERSONAS = 2
TURNS = 4

# Each conversation file starts with a random 6-character tag.
RANDOM_TAG = re.compile(r"^[0-9a-f]{6}_")


def pipeline_config(output: Path) -> dict[str, Any]:
    """The run being recorded. One job at a time, so replies replay in order."""
    return {
        "target": "SI",
        "generation": {
            "chatbot": {"name": CHATBOT, "repeats": 1},
            "user": [{"name": USER, "repeats": 1}],
            "turns": TURNS,
            "output": str(output),
            "max_concurrent": 1,
            "max_total_words": None,
            "persona_speaks_first": True,
            "sessions": None,
        },
        "judging": {
            "models": [{"name": JUDGE, "repeats": 1}],
            "max_concurrent": 1,
            "per_judge": False,
        },
        "scoring": {"personas": None, "skip_risk_analysis": True},
    }


def run_pipeline(work: Path, create_llm: Callable[..., Any]) -> Path:
    """Run `vera pipeline` with `create_llm` standing in for the LLM factory.

    Returns the generation run folder.
    """
    work.mkdir(parents=True, exist_ok=True)
    config = work / "pipeline.json"
    config.write_text(json.dumps(pipeline_config(work / "output")), encoding="utf-8")
    os.environ.pop(cli_config.VERA_RUN_CONFIG_ENV, None)
    with patch.object(LLMFactory, "create_llm", side_effect=create_llm):
        exit_code = vera.main(
            ["pipeline", "--config", str(config), "--sample", str(PERSONAS)]
        )
    if exit_code != 0:
        raise RuntimeError(f"vera pipeline exited with {exit_code}")
    (run_folder,) = (work / "output").iterdir()
    return run_folder


def outputs(run_folder: Path) -> dict[str, str]:
    """The run's outputs, keyed by a stable name, with run-specific values removed.

    Random tags, timestamps, and folder names differ on every run; everything
    else must match exactly.
    """
    found = {}
    for transcript in sorted((run_folder / "conversations").glob("*.txt")):
        name = RANDOM_TAG.sub("", transcript.name)
        found[f"transcripts/{name}"] = transcript.read_text(encoding="utf-8")

    (evaluation,) = (run_folder / "evaluations").iterdir()
    with open(evaluation / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["filename"] = RANDOM_TAG.sub("", row["filename"])
        row.pop("run_id")
    rows.sort(key=lambda row: (row["filename"], row["judge_instance"]))
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    found["results.csv"] = table.getvalue()

    found["scores.json"] = (evaluation / "scores" / "scores.json").read_text(
        encoding="utf-8"
    )
    return found


def main() -> None:
    RECORDING.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_folder = run_pipeline(Path(tmp), recording_llm_factory(RECORDING))
        produced = outputs(run_folder)

    if EXPECTED.exists():
        shutil.rmtree(EXPECTED)
    for name, text in produced.items():
        path = EXPECTED / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    calls = len(RECORDING.read_text(encoding="utf-8").splitlines())
    print(f"Recorded {calls} calls and {len(produced)} expected outputs in {HERE}")


if __name__ == "__main__":
    main()
