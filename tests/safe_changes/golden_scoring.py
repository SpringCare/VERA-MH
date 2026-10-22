"""Golden scoring: committed results.csv files and the exact scores they produce.

Each case folder in tests/fixtures/golden_scoring/ holds a `results.csv` and,
in `expected/`, the `scores.json` and `scores_by_risk.json` scoring produced
from it. tests/unit/score/test_scoring_golden.py rescores every case and
requires the same output.

Run this only for an intended change to scoring, then review the diff of the
`expected/` files: that diff is the change in published numbers.

    uv run python -m tests.safe_changes.golden_scoring
    uv run python -m tests.safe_changes.golden_scoring --rebuild-cases

The first rescores every case; the second also rebuilds each case's input.
"""

import argparse
import csv
import shutil
import tempfile
from pathlib import Path

from score.run import run_scoring

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = REPO_ROOT / "tests/fixtures/golden_scoring"
SOURCE_RESULTS = (
    REPO_ROOT / "tests/fixtures/eval_tsv_rebuild__20260415_140037/results.csv"
)
SI_PERSONAS = REPO_ROOT / "data/SI/personas.tsv"
OUTPUTS = ("scores.json", "scores_by_risk.json")
DIMENSIONS = [
    "Detects Potential Risk",
    "Confirms Risk",
    "Guides to Human Care",
    "Supportive Conversation",
    "Follows AI Boundaries",
]


def cases() -> list[Path]:
    """Every case folder, in a stable order."""
    return sorted(
        path for path in CASES_DIR.iterdir() if (path / "results.csv").is_file()
    )


def score_case(case_dir: Path, work_root: Path) -> Path:
    """Score a copy of one case and return the folder holding its outputs.

    The copy keeps the case's folder name, because scoring reads model names
    from the folder name and they appear in the output.
    """
    work = work_root / case_dir.name
    work.mkdir()
    shutil.copy(case_dir / "results.csv", work / "results.csv")
    run_scoring(
        results_csv=str(work / "results.csv"),
        output_json=None,
        personas_tsv=str(SI_PERSONAS),
        skip_risk_analysis=False,
    )
    return work / "scores"


def rescore_all_cases() -> None:
    """Overwrite every case's `expected/` files with what scoring produces now."""
    with tempfile.TemporaryDirectory() as tmp:
        for case_dir in cases():
            scores = score_case(case_dir, Path(tmp))
            expected = case_dir / "expected"
            expected.mkdir(exist_ok=True)
            for name in OUTPUTS:
                if (scores / name).is_file():
                    shutil.copy(scores / name, expected / name)


def rebuild_cases() -> None:
    """Write each case's `results.csv`, all variations on one real SI evaluation."""
    with open(SOURCE_RESULTS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    variations = {
        # The real evaluation, unchanged: every rating band appears.
        "si_mixed": rows,
        # Every dimension Not Relevant: nothing to score.
        "all_not_relevant": [
            {**row, **dict.fromkeys(DIMENSIONS, "Not Relevant")} for row in rows
        ],
        # One conversation, one judge instance.
        "single_evaluation": rows[2:3],
        # One dimension with no ratings at all.
        "one_dimension_empty": [{**row, "Follows AI Boundaries": ""} for row in rows],
    }
    for name, case_rows in variations.items():
        folder = CASES_DIR / name
        folder.mkdir(exist_ok=True)
        with open(folder / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(case_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate the golden scoring files.")
    parser.add_argument(
        "--rebuild-cases",
        action="store_true",
        help="rebuild each case's results.csv before rescoring",
    )
    if parser.parse_args().rebuild_cases:
        rebuild_cases()
    rescore_all_cases()
