"""Build the golden scoring inputs from the checked-in evaluation fixture.

Run once to (re)create the case folders; the outputs are committed. Each case is
a variation on one real SI evaluation, chosen to exercise a different part of
the score math.
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOURCE = REPO_ROOT / "tests/fixtures/eval_tsv_rebuild__20260415_140037/results.csv"
DIMENSIONS = [
    "Detects Potential Risk",
    "Confirms Risk",
    "Guides to Human Care",
    "Supportive Conversation",
    "Follows AI Boundaries",
]


def write(case: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    folder = HERE / case
    folder.mkdir(exist_ok=True)
    with open(folder / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with open(SOURCE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # The real evaluation, unchanged: every rating band appears.
    write("si_mixed", rows, fieldnames)

    # Every dimension Not Relevant: nothing to score.
    write(
        "all_not_relevant",
        [{**row, **{d: "Not Relevant" for d in DIMENSIONS}} for row in rows],
        fieldnames,
    )

    # One conversation, one judge instance.
    write("single_evaluation", rows[2:3], fieldnames)

    # One dimension with no ratings at all.
    write(
        "one_dimension_empty",
        [{**row, "Follows AI Boundaries": ""} for row in rows],
        fieldnames,
    )


if __name__ == "__main__":
    main()
