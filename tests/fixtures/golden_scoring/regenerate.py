"""Recompute the expected scores for every golden case.

Run this only for an intended change to scoring, then review the diff of the
`expected/` files in the PR: that diff is the change in published numbers.

    uv run python tests/fixtures/golden_scoring/regenerate.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))  # so it runs as a plain script

from score.run import run_scoring  # noqa: E402

SI_PERSONAS = REPO_ROOT / "data/SI/personas.tsv"
OUTPUTS = ("scores.json", "scores_by_risk.json")


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


def cases() -> list[Path]:
    return sorted(p for p in HERE.iterdir() if (p / "results.csv").is_file())


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for case_dir in cases():
            scores = score_case(case_dir, Path(tmp))
            expected = case_dir / "expected"
            expected.mkdir(exist_ok=True)
            for name in OUTPUTS:
                if (scores / name).is_file():
                    shutil.copy(scores / name, expected / name)


if __name__ == "__main__":
    main()
