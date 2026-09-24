"""Persona annotation of judge results.

Covers the join that copies target-declared persona columns into results.csv
(`add_persona_columns_to_dataframe`), the generic loader behind it
(`load_persona_column`), and the runner wrapper that must never fail a paid
judging run over a reporting convenience (`_annotate_with_personas`).
"""

import pandas as pd
import pytest

from judge.runner import _annotate_with_personas
from judge.score_utils import (
    add_persona_columns_to_dataframe,
    load_persona_column,
    load_personas_risk_levels,
)

# Two targets with deliberately different persona schemas: the point of the
# generic loader is that neither column name is privileged in code.
HFO_PERSONAS = (
    "Name\tAge\tActive Life Threat\tBackground\n"
    "AAA1111\t34\tPresent\tlives with partner\n"
    "BBB2222\t27\tAbsent\tstaying with a friend\n"
)
SI_PERSONAS = (
    "Name\tAge\tShort Current Suicide Risk Level\n"
    "AAA1111\t34\tHigh\n"
    "BBB2222\t27\tNone\n"
)


@pytest.fixture
def hfo_personas(tmp_path):
    path = tmp_path / "personas_HFO.tsv"
    path.write_text(HFO_PERSONAS)
    return path


@pytest.fixture
def results_df():
    return pd.DataFrame(
        {
            "filename": [
                "abc123_AAA1111_gpt-5.2_run1.txt",
                "def456_BBB2222_gpt-5.2_run1.txt",
            ],
            "run_id": ["run1", "run1"],
            "judge_model": ["gpt-5.2", "gpt-5.2"],
            "Safety": [1, 0],
        }
    )


class TestLoadPersonaColumn:
    def test_reads_named_column(self, hfo_personas):
        assert load_persona_column(hfo_personas, "Active Life Threat") == {
            "AAA1111": "Present",
            "BBB2222": "Absent",
        }

    def test_missing_column_raises_rather_than_returning_unknowns(self, hfo_personas):
        # Silently yielding "Unknown" for every row is what made the SI-only
        # risk lookup fail invisibly against other targets.
        with pytest.raises(KeyError, match="Short Current Suicide Risk Level"):
            load_persona_column(hfo_personas, "Short Current Suicide Risk Level")

    def test_preserves_literal_none_string(self, tmp_path):
        path = tmp_path / "personas_SI.tsv"
        path.write_text(SI_PERSONAS)
        # "None" is a meaningful risk level, not a missing value.
        assert (
            load_persona_column(path, "Short Current Suicide Risk Level")["BBB2222"]
            == "None"
        )

    def test_risk_level_helper_still_joins_si_column(self, tmp_path):
        path = tmp_path / "personas_SI.tsv"
        path.write_text(SI_PERSONAS)
        assert load_personas_risk_levels(path) == {
            "AAA1111": "High",
            "BBB2222": "None",
        }


class TestAddPersonaColumns:
    def test_adds_persona_name_and_requested_column(self, results_df, hfo_personas):
        out = add_persona_columns_to_dataframe(
            results_df, hfo_personas, ["Active Life Threat"]
        )
        assert list(out["persona_name"]) == ["AAA1111", "BBB2222"]
        assert list(out["Active Life Threat"]) == ["Present", "Absent"]

    def test_annotation_sits_beside_the_identifying_columns(
        self, results_df, hfo_personas
    ):
        out = add_persona_columns_to_dataframe(
            results_df, hfo_personas, ["Active Life Threat"]
        )
        assert list(out.columns)[:4] == [
            "filename",
            "run_id",
            "persona_name",
            "Active Life Threat",
        ]

    def test_evaluation_columns_are_preserved(self, results_df, hfo_personas):
        out = add_persona_columns_to_dataframe(
            results_df, hfo_personas, ["Active Life Threat"]
        )
        assert list(out["Safety"]) == [1, 0]
        assert len(out) == len(results_df)

    def test_no_columns_requested_is_a_no_op(self, results_df, hfo_personas):
        out = add_persona_columns_to_dataframe(results_df, hfo_personas, [])
        assert list(out.columns) == list(results_df.columns)

    def test_reannotating_is_idempotent(self, results_df, hfo_personas):
        once = add_persona_columns_to_dataframe(
            results_df, hfo_personas, ["Active Life Threat"]
        )
        twice = add_persona_columns_to_dataframe(
            once, hfo_personas, ["Active Life Threat"]
        )
        pd.testing.assert_frame_equal(once, twice)

    def test_persona_absent_from_file_is_marked_unknown(self, hfo_personas):
        # A row whose persona is not in the file still has a judged result; the
        # annotation is unknown, the row is not dropped.
        df = pd.DataFrame(
            {
                "filename": ["zzz999_NOTHERE_gpt-5.2_run1.txt"],
                "run_id": ["run1"],
                "Safety": [1],
            }
        )
        out = add_persona_columns_to_dataframe(df, hfo_personas, ["Active Life Threat"])
        assert list(out["Active Life Threat"]) == ["Unknown"]
        assert len(out) == 1

    def test_multiple_columns_keep_requested_order(self, results_df, hfo_personas):
        out = add_persona_columns_to_dataframe(
            results_df, hfo_personas, ["Active Life Threat", "Age"]
        )
        assert list(out.columns)[2:5] == [
            "persona_name",
            "Active Life Threat",
            "Age",
        ]


class TestRunnerAnnotationIsNonFatal:
    """A reporting convenience must not discard a run that already cost money."""

    def test_annotates_when_configured(self, results_df, hfo_personas):
        out = _annotate_with_personas(
            results_df, [str(hfo_personas)], ["Active Life Threat"], verbose=False
        )
        assert list(out["Active Life Threat"]) == ["Present", "Absent"]

    def test_missing_column_warns_and_returns_unannotated(
        self, results_df, hfo_personas, capsys
    ):
        out = _annotate_with_personas(
            results_df, [str(hfo_personas)], ["Nonexistent Column"], verbose=True
        )
        assert list(out.columns) == list(results_df.columns)
        assert "Skipping persona annotation" in capsys.readouterr().out

    def test_unreadable_personas_file_returns_unannotated(self, results_df, tmp_path):
        out = _annotate_with_personas(
            results_df,
            [str(tmp_path / "gone.tsv")],
            ["Active Life Threat"],
            verbose=False,
        )
        assert list(out.columns) == list(results_df.columns)

    def test_no_personas_or_columns_is_a_no_op(self, results_df, hfo_personas):
        assert list(
            _annotate_with_personas(results_df, None, None, verbose=False).columns
        ) == list(results_df.columns)
        assert list(
            _annotate_with_personas(
                results_df, [str(hfo_personas)], [], verbose=False
            ).columns
        ) == list(results_df.columns)
