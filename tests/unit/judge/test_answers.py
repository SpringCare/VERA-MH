"""Unit tests for judge/answers.py — per-question answer artifacts.

Covers the per-conversation long-format file, the run-level wide aggregate
(rubric-ordered columns, blanks for unvisited questions), and the reason the
visit log exists at all: dimension_answers is rewritten in place by
NOT_RELEVANT>> handling and cannot be used as the source.
"""

from pathlib import Path

import pandas as pd
import pytest

from judge.answers import (
    ANSWERS_TSV_NAME,
    answer_column,
    answers_tsv_path,
    build_answers_dataframe,
    conversation_answers_dir,
    explanation_column,
    write_answers_tsv,
    write_conversation_answers,
)
from judge.llm_judge import LLMJudge

# Opaque, non-lexical IDs: sorting these as strings gives 1, 10, 2 — the
# aggregate must follow rubric order instead.
QUESTION_ORDER = ["1", "2", "9", "10", "1a"]

EVAL_TSV_NAME = "abc123_Brian_gpt-5_run1_mock-llm_i1.tsv"


def _log(*pairs):
    return [
        {"question_id": qid, "answer": answer, "reasoning": f"why {qid}"}
        for qid, answer in pairs
    ]


@pytest.mark.unit
class TestWriteConversationAnswers:
    """Test the per-conversation long-format file."""

    def test_writes_header_and_one_row_per_visited_question(self, tmp_path: Path):
        out = tmp_path / "answers" / "conversations" / EVAL_TSV_NAME
        write_conversation_answers(out, _log(("1", "Yes"), ("9", "Immediate risk")))

        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "Question ID\tAnswer\tExplanation"
        assert lines[1] == "1\tYes\twhy 1"
        assert lines[2] == "9\tImmediate risk\twhy 9"
        assert len(lines) == 3

    def test_preserves_visit_order(self, tmp_path: Path):
        out = tmp_path / "a.tsv"
        write_conversation_answers(out, _log(("9", "A"), ("1", "B"), ("10", "C")))

        ids = [line.split("\t")[0] for line in out.read_text().splitlines()[1:]]
        assert ids == ["9", "1", "10"]

    def test_flattens_tabs_and_newlines_in_explanation(self, tmp_path: Path):
        out = tmp_path / "a.tsv"
        write_conversation_answers(
            out,
            [
                {
                    "question_id": "1",
                    "answer": "Yes",
                    "reasoning": "line one\nline\ttwo\r\nline three",
                }
            ],
        )

        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2, "explanation must not spill across rows"
        assert lines[1].count("\t") == 2, "explanation must not add columns"
        assert lines[1] == "1\tYes\tline one line two  line three"

    def test_empty_log_writes_header_only(self, tmp_path: Path):
        out = tmp_path / "a.tsv"
        write_conversation_answers(out, [])
        assert out.read_text(encoding="utf-8").splitlines() == [
            "Question ID\tAnswer\tExplanation"
        ]


@pytest.mark.unit
class TestBuildAnswersDataframe:
    """Test the run-level wide aggregate."""

    def _eval_folder(self, tmp_path: Path) -> Path:
        folder = (
            tmp_path / "j_mockx1_20260101_000000__p_x__a_y__t1__r1__20260101_000000"
        )
        folder.mkdir(parents=True)
        return folder

    def test_columns_follow_rubric_order_not_sort_order(self, tmp_path: Path):
        folder = self._eval_folder(tmp_path)
        write_conversation_answers(
            conversation_answers_dir(folder) / EVAL_TSV_NAME, _log(("1", "Yes"))
        )

        df = build_answers_dataframe(folder, QUESTION_ORDER)

        question_columns = [c for c in df.columns if c.startswith("Q")]
        assert question_columns == [
            "Q1",
            "Q1 explanation",
            "Q2",
            "Q2 explanation",
            "Q9",
            "Q9 explanation",
            "Q10",
            "Q10 explanation",
            "Q1a",
            "Q1a explanation",
        ]

    def test_one_row_per_conversation_with_identity_columns(self, tmp_path: Path):
        folder = self._eval_folder(tmp_path)
        answers = conversation_answers_dir(folder)
        write_conversation_answers(answers / EVAL_TSV_NAME, _log(("1", "Yes")))
        write_conversation_answers(
            answers / "def456_Dana_gpt-5_run2_mock-llm_i1.tsv", _log(("1", "No"))
        )

        df = build_answers_dataframe(folder, QUESTION_ORDER)

        assert len(df) == 2
        assert sorted(df["filename"]) == [
            "abc123_Brian_gpt-5_run1.txt",
            "def456_Dana_gpt-5_run2.txt",
        ]
        assert sorted(df["persona_name"]) == ["Brian", "Dana"]
        assert set(df["judge_model"]) == {"mock-llm"}
        assert set(df["judge_instance"]) == {1}
        assert set(df["judge_id"]) == {0}
        assert set(df["run_id"]) == {"20260101_000000"}

    def test_unvisited_questions_are_blank(self, tmp_path: Path):
        folder = self._eval_folder(tmp_path)
        write_conversation_answers(
            conversation_answers_dir(folder) / EVAL_TSV_NAME,
            _log(("1", "Yes"), ("9", "Immediate risk")),
        )

        row = build_answers_dataframe(folder, QUESTION_ORDER).iloc[0]

        assert row[answer_column("1")] == "Yes"
        assert row[explanation_column("1")] == "why 1"
        assert row[answer_column("9")] == "Immediate risk"
        # Q2, Q10 and Q1a were never reached by this conversation's flow.
        for question_id in ("2", "10", "1a"):
            assert row[answer_column(question_id)] == ""
            assert row[explanation_column(question_id)] == ""

    def test_question_not_in_rubric_is_ignored(self, tmp_path: Path):
        folder = self._eval_folder(tmp_path)
        write_conversation_answers(
            conversation_answers_dir(folder) / EVAL_TSV_NAME,
            _log(("1", "Yes"), ("999", "from another rubric")),
        )

        df = build_answers_dataframe(folder, QUESTION_ORDER)

        assert "Q999" not in df.columns
        assert df.iloc[0][answer_column("1")] == "Yes"

    def test_missing_answers_folder_yields_empty_but_columned_frame(
        self, tmp_path: Path
    ):
        df = build_answers_dataframe(self._eval_folder(tmp_path), QUESTION_ORDER)

        assert df.empty
        assert answer_column("9") in df.columns
        assert explanation_column("9") in df.columns

    def test_write_answers_tsv_round_trips(self, tmp_path: Path):
        folder = self._eval_folder(tmp_path)
        write_conversation_answers(
            conversation_answers_dir(folder) / EVAL_TSV_NAME,
            _log(("1", "Yes"), ("9", "Immediate risk")),
        )

        out_path = write_answers_tsv(folder, QUESTION_ORDER)

        assert out_path == answers_tsv_path(folder)
        assert out_path.name == ANSWERS_TSV_NAME
        df = pd.read_csv(out_path, sep="\t", keep_default_na=False)
        assert len(df) == 1
        assert df.iloc[0]["Q9"] == "Immediate risk"
        assert df.iloc[0]["Q9 explanation"] == "why 9"

    def test_aggregate_is_not_mistaken_for_an_evaluation_tsv(self, tmp_path: Path):
        """Both artifacts sit below the evaluation folder, not inside it.

        build_results_csv_from_tsv_files globs *.tsv directly in the evaluation
        folder, and judge.runner's resume scan does the same, so a stray TSV
        there would be read as an evaluation.
        """
        folder = self._eval_folder(tmp_path)
        write_conversation_answers(
            conversation_answers_dir(folder) / EVAL_TSV_NAME, _log(("1", "Yes"))
        )
        write_answers_tsv(folder, QUESTION_ORDER)

        assert list(folder.glob("*.tsv")) == []


@pytest.mark.unit
class TestVisitLogSurvivesNotRelevantGoto:
    """The reason the visit log is not sourced from dimension_answers."""

    async def test_not_relevant_goto_clobbers_dimension_answers(
        self, rubric_config_factory
    ):
        """_handle_not_relevant_goto replaces a dimension's whole answer list.

        A question whose GOTO is NOT_RELEVANT>> therefore erases its own
        recorded answer, which is exactly the answer an per-question artifact
        needs to keep.
        """
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(judge_model="mock-llm", rubric_config=rubric_config)
        dimension = rubric_config.question_flow_data[rubric_config.question_order[0]][
            "dimension"
        ]

        dimension_answers = {
            dimension: [
                {
                    "question_id": "9",
                    "question": "Select one of the following:",
                    "answer": "No suicidal thoughts present",
                    "severity": None,
                    "reasoning": "No indication of suicidal thoughts.",
                }
            ]
        }

        judge._handle_not_relevant_goto(
            "9", "No suicidal thoughts present", dimension, dimension_answers, False
        )

        surviving = dimension_answers[dimension]
        assert len(surviving) == 1
        assert surviving[0]["answer"] == "Not Relevant"
        assert all(
            entry["answer"] != "No suicidal thoughts present" for entry in surviving
        ), "Q9's real answer is gone from dimension_answers"

    def test_visit_log_retains_the_answer(self, tmp_path: Path):
        """The append-only log is never rewritten, so the answer persists."""
        folder = tmp_path / "j_mockx1_20260101_000000__run__20260101_000000"
        folder.mkdir(parents=True)
        write_conversation_answers(
            conversation_answers_dir(folder) / EVAL_TSV_NAME,
            _log(("9", "No suicidal thoughts present")),
        )

        row = build_answers_dataframe(folder, QUESTION_ORDER).iloc[0]

        assert row[answer_column("9")] == "No suicidal thoughts present"
