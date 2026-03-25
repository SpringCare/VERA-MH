"""Tests for utils.role_reversal_transcript."""

from pathlib import Path

from utils.role_reversal_transcript import (
    discover_conversation_txt_files,
    is_run_folder_name,
    iter_user_messages_with_turns,
    parse_transcript_to_turns,
    parse_user_and_provider_from_run_folder,
)


def test_parse_user_and_provider_from_run_folder() -> None:
    d = "p_claude_opus_4_5__a_gemini_3_1__t30__r1__20260223_213637"
    assert parse_user_and_provider_from_run_folder(d) == (
        "claude_opus_4_5",
        "gemini_3_1",
    )
    assert parse_user_and_provider_from_run_folder("not_a_run") is None
    assert parse_user_and_provider_from_run_folder("p_only__no_a") is None


def test_is_run_folder_name() -> None:
    assert is_run_folder_name(
        "p_claude_opus_4_5__a_gemini_3_1__t30__r1__20260223_213637"
    )
    assert not is_run_folder_name("random_folder")


def test_parse_transcript_to_turns_multiline_and_case() -> None:
    text = """User: line one
still user

CHATBOT: bot says
something

user: second user
"""
    turns = parse_transcript_to_turns(text)
    assert [(s, t, n) for s, t, n in turns] == [
        ("user", "line one\nstill user", 1),
        ("chatbot", "bot says\nsomething", 2),
        ("user", "second user", 3),
    ]


def test_parse_skips_conversation_ended_marker() -> None:
    text = """user: hello

[CONVERSATION ENDED - persona signaled termination]

chatbot: ok
"""
    turns = parse_transcript_to_turns(text)
    assert len(turns) == 2
    assert turns[0][0] == "user"
    assert turns[1][0] == "chatbot"


def test_assistant_mapped_to_chatbot() -> None:
    text = """user: hi
Assistant: there
"""
    turns = parse_transcript_to_turns(text)
    assert turns[1][0] == "chatbot"


def test_iter_user_messages_with_turns_indices() -> None:
    text = """user: a
chatbot: b
user: c
"""
    rows = iter_user_messages_with_turns(text)
    assert rows == [(1, 1, "a"), (2, 3, "c")]


def test_discover_conversation_txt_files(tmp_path: Path) -> None:
    good = "p_gpt_4o__a_claude_x__t6__r1__20250101_120000"
    bad = "other_folder"
    (tmp_path / good).mkdir()
    (tmp_path / bad).mkdir()
    (tmp_path / good / "1a_Kim_g4o_run1.txt").write_text("user: hi\n", encoding="utf-8")
    (tmp_path / bad / "x.txt").write_text("user: hi\n", encoding="utf-8")

    found = discover_conversation_txt_files(tmp_path)
    assert len(found) == 1
    assert found[0].name == "1a_Kim_g4o_run1.txt"
