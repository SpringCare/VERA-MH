"""Math invariants for ``scripts/export_heor_scores_summary_csv.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MOD_NAME = "export_heor_scores_summary_csv"
_SPEC = importlib.util.spec_from_file_location(
    _MOD_NAME,
    _REPO_ROOT / "scripts" / "export_heor_scores_summary_csv.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_MOD_NAME] = _mod
_SPEC.loader.exec_module(_mod)

DIMENSIONS = _mod.DIMENSIONS
DimCounts = _mod.DimCounts
merge_bundles = _mod.merge_bundles
overall_vera_from_pooled_counts = _mod.overall_vera_from_pooled_counts
pct_of_total = _mod.pct_of_total


def _bundle(
    *,
    per_dim_counts: dict[str, tuple[int, int, int, int]],
    judge: str = "gpt4o",
) -> dict:
    """Build a minimal bundle dict (BP, sub, HPH, NR per dimension)."""
    per_dim: dict[str, DimCounts] = {}
    for d in DIMENSIONS:
        bp, sub, hph, nr = per_dim_counts.get(d, (0, 0, 0, 0))
        per_dim[d] = DimCounts(bp, sub, hph, nr)
    scored_bp = scored_sub = scored_hph = eval_nr = 0
    for dc in per_dim.values():
        scored_bp += dc.best_practice
        scored_sub += dc.suboptimal
        scored_hph += dc.hph
        eval_nr += dc.not_relevant
    n_scored = scored_bp + scored_sub + scored_hph
    n_eval = n_scored + eval_nr
    return {
        "persona_model": "opus45",
        "agent_model": "gemini31",
        "judge": judge,
        "total_score": 50.0,
        "per_dim": per_dim,
        "totals": {
            "scored_bp": scored_bp,
            "scored_sub": scored_sub,
            "scored_hph": scored_hph,
            "n_scored": n_scored,
            "eval_nr": eval_nr,
            "n_eval": n_eval,
        },
        "json_dim": {},
    }


def _uniform_dims(
    bp: int, sub: int, hph: int, nr: int
) -> dict[str, tuple[int, int, int, int]]:
    return {d: (bp, sub, hph, nr) for d in DIMENSIONS}


def test_merge_doubles_counts_preserves_overall_vera_when_proportions_identical():
    """Merging two identical evals doubles n_scored but BP/HPH shares stay the same."""
    c = _uniform_dims(8, 2, 2, 1)
    b1 = _bundle(per_dim_counts=c, judge="gpt4o")
    b2 = _bundle(per_dim_counts=c, judge="sonnet45")
    m = merge_bundles([b1, b2], judge="pooled")
    assert m["totals"]["n_scored"] == 2 * b1["totals"]["n_scored"]
    v1 = overall_vera_from_pooled_counts(
        b1["totals"]["scored_bp"],
        b1["totals"]["scored_sub"],
        b1["totals"]["scored_hph"],
    )
    assert m["total_score"] == pytest.approx(v1, rel=1e-9)


def test_overall_vera_matches_manual_aggregate_percentages():
    """Hand-check one merged pool against judge-style BP/HPH % over summed counts."""
    c = _uniform_dims(bp=30, sub=50, hph=20, nr=0)
    b1 = _bundle(per_dim_counts=c)
    b2 = _bundle(per_dim_counts=c)
    m = merge_bundles([b1, b2], judge="pooled")
    n = m["totals"]["n_scored"]
    assert n == 500 * 2  # 5 dims * 100 scored per bundle * 2 bundles
    bp = m["totals"]["scored_bp"]
    hph = m["totals"]["scored_hph"]
    assert pct_of_total(bp, n) == pytest.approx(30.0)
    assert pct_of_total(hph, n) == pytest.approx(20.0)
    assert m["total_score"] == pytest.approx(
        overall_vera_from_pooled_counts(bp, m["totals"]["scored_sub"], hph)
    )


def test_merge_sums_not_relevant_across_bundles():
    c1 = _uniform_dims(10, 0, 0, 2)
    c2 = _uniform_dims(10, 0, 0, 3)
    m = merge_bundles(
        [_bundle(per_dim_counts=c1), _bundle(per_dim_counts=c2)],
        judge="pooled",
    )
    assert m["totals"]["eval_nr"] == (2 + 3) * 5
    n_eval = m["totals"]["n_eval"]
    assert n_eval == m["totals"]["n_scored"] + m["totals"]["eval_nr"]
