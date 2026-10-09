"""Unit tests for the naming and input helpers in score/pool.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from score import pool

GEN_RUN = "p_claude_opus_4_5__a_gpt_5_4_nano__t30__r1__20260422_150000"


def _nested_eval(tmp_path: Path, gen_run: str, eval_name: str) -> Path:
    """An evaluation folder in the standard ``p_*/evaluations/j_*`` layout."""
    eval_dir = tmp_path / gen_run / "evaluations" / eval_name
    eval_dir.mkdir(parents=True)
    return eval_dir


def test_resolve_eval_input_accepts_a_folder_or_its_results_csv(
    tmp_path: Path,
) -> None:
    (tmp_path / "results.csv").write_text("x\n")
    assert pool._resolve_eval_input(tmp_path) == tmp_path.resolve()
    assert pool._resolve_eval_input(tmp_path / "results.csv") == tmp_path.resolve()


def test_resolve_eval_input_rejects_other_files_and_missing_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "other.csv").write_text("x\n")
    with pytest.raises(ValueError, match="results.csv"):
        pool._resolve_eval_input(tmp_path / "other.csv")
    with pytest.raises(FileNotFoundError):
        pool._resolve_eval_input(tmp_path / "absent")


def test_generation_folder_found_only_in_the_nested_layout(tmp_path: Path) -> None:
    nested = _nested_eval(tmp_path, GEN_RUN, "j_x")
    flat = tmp_path / "evaluations" / "j_y"
    flat.mkdir(parents=True)

    assert pool._generation_folder_for_eval(nested) == (tmp_path / GEN_RUN).resolve()
    assert pool._generation_folder_for_eval(flat) is None


def test_judge_slug_from_dataframe_counts_instances_per_model() -> None:
    df = pd.DataFrame(
        {
            "judge_model": ["gpt-4o", "gpt-4o", "claude-sonnet-4-5"],
            "judge_instance": [1, 2, 1],
        }
    )
    assert pool._judge_slug_from_dataframe(df) == "claude-sonnet-4-5x1+gpt-4ox2"


def test_judge_slug_from_dataframe_without_judge_column() -> None:
    assert pool._judge_slug_from_dataframe(pd.DataFrame({"a": [1]})) == "unknown"


def test_merge_judge_slug_tokens_keeps_the_highest_count_per_model() -> None:
    assert (
        pool._merge_judge_slug_tokens("gpt-4ox1", "gpt-4ox3+sonnetx1")
        == "gpt-4ox3+sonnetx1"
    )
    assert pool._merge_judge_slug_tokens("not-a-token") == "unknown"


def test_judge_slug_from_eval_dir_name() -> None:
    name = "j_gpt-5.4x1_20260422_150000_123__p_claude"
    assert pool._judge_slug_from_eval_dir_name(name) == "gpt-5.4x1"
    assert pool._judge_slug_from_eval_dir_name("j_pooled__p_x") is None
    assert pool._judge_slug_from_eval_dir_name("something_else") is None


def test_pooled_folder_name_uses_the_first_generation_run(tmp_path: Path) -> None:
    eval_a = _nested_eval(tmp_path, GEN_RUN, "j_a")
    eval_b = _nested_eval(
        tmp_path, "p_gpt_5_2__a_gpt_5_4_nano__t30__r1__20260422_160000", "j_b"
    )
    combined = pd.DataFrame({"judge_model": ["gpt-5.4"], "judge_instance": [1]})

    name = pool._synthetic_pooled_folder_basename([eval_a, eval_b], combined)

    assert name.startswith(
        "j_gpt-5.4x1__p_claude_opus_4_5+gpt_5_2__a_gpt_5_4_nano__t30__r1__"
    )


def test_pooled_folder_name_falls_back_without_a_generation_run(
    tmp_path: Path,
) -> None:
    combined = pd.DataFrame({"judge_model": ["gpt-5.4"], "judge_instance": [1]})

    name = pool._synthetic_pooled_folder_basename([tmp_path], combined)

    assert name.startswith("j_gpt-5.4x1__p_unknown__a_unknown__t30__r1__")
