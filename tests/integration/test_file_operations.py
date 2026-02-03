"""Integration tests for conversation file I/O operations.

Tests real file I/O operations including saving, loading, and handling
various conversation formats and edge cases.
"""

from pathlib import Path

import pytest

from utils.conversation_utils import (
    format_conversation_summary,
    generate_conversation_filename,
    save_conversation_to_file,
)


@pytest.mark.integration
class TestConversationFileOperations:
    """Integration tests for conversation file I/O operations."""

    def test_save_and_load_conversation(self, tmp_path: Path) -> None:
        """Test round-trip save and load of a conversation."""
        conversation = [
            {"speaker": "persona", "response": "Hello, how are you?"},
            {"speaker": "provider", "response": "I'm doing well, thanks for asking!"},
            {"speaker": "persona", "response": "That's great to hear."},
        ]

        filename = "test_conversation.txt"
        save_conversation_to_file(conversation, filename, str(tmp_path))

        # Verify file exists
        file_path = tmp_path / filename
        assert file_path.exists()

        # Read and verify content
        content = file_path.read_text(encoding="utf-8")
        assert "user: Hello, how are you?" in content
        assert "chatbot: I'm doing well, thanks for asking!" in content
        assert "user: That's great to hear." in content

    def test_save_handles_unicode(self, tmp_path: Path) -> None:
        """Test saving conversation with Unicode (emojis and accents)."""
        conversation = [
            {"speaker": "persona", "response": "Hello 👋 I'm feeling anxious 😟"},
            {
                "speaker": "provider",
                "response": (
                    "I understand. Let's work through this together. "
                    "C'est normal d'avoir peur."
                ),
            },
            {
                "speaker": "persona",
                "response": "Merci! That means a lot. 🙏 Ich bin dankbar.",
            },
        ]

        filename = "unicode_conversation.txt"
        save_conversation_to_file(conversation, filename, str(tmp_path))

        file_path = tmp_path / filename
        assert file_path.exists()

        content = file_path.read_text(encoding="utf-8")
        assert "👋" in content
        assert "😟" in content
        assert "C'est normal d'avoir peur." in content
        assert "Merci!" in content
        assert "Ich bin dankbar." in content

    def test_conversation_file_format(self, tmp_path: Path) -> None:
        """Test saved conversation follows format (user:/chatbot:)."""
        conversation = [
            {"speaker": "persona", "response": "First message"},
            {"speaker": "provider", "response": "Second message"},
        ]

        filename = "format_test.txt"
        save_conversation_to_file(conversation, filename, str(tmp_path))

        file_path = tmp_path / filename
        content = file_path.read_text(encoding="utf-8")

        # Verify proper formatting with user: and chatbot: prefixes
        lines = content.split("\n")
        assert any(line.startswith("user:") for line in lines)
        assert any(line.startswith("chatbot:") for line in lines)

    def test_save_to_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test saving to a nonexistent directory creates the directory."""
        conversation = [{"speaker": "persona", "response": "Test message"}]

        # Create a path to a subdirectory that doesn't exist
        subdir = tmp_path / "nested" / "conversations"
        filename = "test.txt"

        # The function should create the directory automatically
        assert not subdir.exists()
        save_conversation_to_file(conversation, filename, str(subdir))

        # Verify directory was created and file was saved
        assert subdir.exists()
        assert (subdir / filename).exists()

    def test_load_from_nonexistent_file(self, tmp_path: Path) -> None:
        """Test attempting to load from nonexistent file raises appropriate error."""
        nonexistent_path = tmp_path / "nonexistent.txt"

        # Verify that the file doesn't exist
        assert not nonexistent_path.exists()

        # Reading should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            nonexistent_path.read_text(encoding="utf-8")

    def test_multiple_conversations_same_directory(self, tmp_path: Path) -> None:
        """Test saving multiple conversation files to the same directory."""
        conversations = [
            [
                {"speaker": "persona", "response": "Message 1"},
                {"speaker": "provider", "response": "Response 1"},
            ],
            [
                {"speaker": "persona", "response": "Message 2"},
                {"speaker": "provider", "response": "Response 2"},
            ],
            [
                {"speaker": "persona", "response": "Message 3"},
                {"speaker": "provider", "response": "Response 3"},
            ],
        ]

        filenames = ["conv1.txt", "conv2.txt", "conv3.txt"]

        # Save all conversations (persona is LLM1 in all)
        for conversation, filename in zip(conversations, filenames):
            save_conversation_to_file(conversation, filename, str(tmp_path))

        # Verify all files exist
        for filename in filenames:
            file_path = tmp_path / filename
            assert file_path.exists()

        # Verify each file contains unique content
        contents = {}
        for filename in filenames:
            contents[filename] = (tmp_path / filename).read_text(encoding="utf-8")

        assert contents["conv1.txt"] != contents["conv2.txt"]
        assert contents["conv2.txt"] != contents["conv3.txt"]
        assert contents["conv1.txt"] != contents["conv3.txt"]

    def test_conversation_with_metadata(self, tmp_path: Path) -> None:
        """Test saving and preserving conversation with metadata (early termination)."""
        conversation = [
            {"speaker": "persona", "response": "I need help"},
            {"speaker": "provider", "response": "I'm here to help."},
            {
                "speaker": "persona",
                "response": "Thank you",
                "early_termination": True,
            },
        ]

        filename = "metadata_conversation.txt"
        save_conversation_to_file(conversation, filename, str(tmp_path))

        file_path = tmp_path / filename
        content = file_path.read_text(encoding="utf-8")

        # Verify early termination metadata is preserved
        assert "[CONVERSATION ENDED - persona signaled termination]" in content

    def test_empty_conversation(self, tmp_path: Path) -> None:
        """Test handling of empty conversation."""
        conversation: list = []

        filename = "empty_conversation.txt"
        save_conversation_to_file(conversation, filename, str(tmp_path))

        file_path = tmp_path / filename
        assert file_path.exists()

        content = file_path.read_text(encoding="utf-8")
        assert content == "No conversation recorded."

    def test_format_conversation_summary_basic(self) -> None:
        """Test formatting conversation summary with basic messages."""
        conversation = [
            {"speaker": "persona", "response": "Hello"},
            {"speaker": "provider", "response": "Hi there!"},
        ]

        summary = format_conversation_summary(conversation)

        assert "user: Hello" in summary
        assert "chatbot: Hi there!" in summary

    def test_format_conversation_summary_empty(self) -> None:
        """Test formatting empty conversation summary."""
        conversation: list = []

        summary = format_conversation_summary(conversation)

        assert summary == "No conversation recorded."

    def test_format_conversation_summary_persona_as_user(self) -> None:
        """Test formatting conversation summary: persona as user, else as chatbot."""
        conversation = [
            {"speaker": "persona", "response": "Hello, I need support."},
            {"speaker": "provider", "response": "Hi, I'm here to help."},
            {"speaker": "persona", "response": "Nice to meet you."},
        ]

        summary = format_conversation_summary(conversation)

        assert "user: Hello, I need support." in summary
        assert "chatbot: Hi, I'm here to help." in summary
        assert "user: Nice to meet you." in summary

    def test_generate_conversation_filename(self) -> None:
        """Test that generated filenames have correct format."""
        filename = generate_conversation_filename()

        assert filename.startswith("conversation_")
        assert filename.endswith(".txt")

        # Verify timestamp format (YYYYMMDD_HHMMSS)
        parts = filename.replace("conversation_", "").replace(".txt", "").split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS

    def test_generate_conversation_filename_with_prefix(self) -> None:
        """Test filename generation with custom prefix."""
        prefix = "mental_health"
        filename = generate_conversation_filename(prefix=prefix)

        assert filename.startswith(f"{prefix}_")
        assert filename.endswith(".txt")

    def test_save_conversation_with_long_messages(self, tmp_path: Path) -> None:
        """Test saving conversation with very long messages."""
        long_message = "This is a very long message. " * 100
        conversation = [
            {"speaker": "persona", "response": long_message},
            {"speaker": "provider", "response": "I understand your concern."},
        ]

        filename = "long_message.txt"
        save_conversation_to_file(conversation, filename, str(tmp_path))

        file_path = tmp_path / filename
        content = file_path.read_text(encoding="utf-8")

        assert long_message in content

    def test_save_conversation_with_special_characters(self, tmp_path: Path) -> None:
        """Test saving conversation with special characters in messages."""
        conversation = [
            {
                "speaker": "persona",
                "response": (
                    "I have questions about: 1) stress, 2) anxiety, 3) depression"
                ),
            },
            {"speaker": "provider", "response": "Let's tackle these one by one."},
            {
                "speaker": "persona",
                "response": "Great! (I'm) [really] {glad} you're helping... ~excited~",
            },
        ]

        filename = "special_chars.txt"
        save_conversation_to_file(conversation, filename, str(tmp_path))

        file_path = tmp_path / filename
        content = file_path.read_text(encoding="utf-8")

        assert "1) stress, 2) anxiety, 3) depression" in content
        assert "(I'm)" in content
        assert "[really]" in content
        assert "{glad}" in content

    def test_conversation_with_newlines_in_messages(self, tmp_path: Path) -> None:
        """Test handling conversation with newline characters in messages."""
        conversation = [
            {
                "speaker": "persona",
                "response": (
                    "I have multiple concerns:\n1. Sleep\n2. Appetite\n3. Focus"
                ),
            },
            {
                "speaker": "provider",
                "response": "Let's discuss each:\n- Sleep patterns\n- Eating habits",
            },
        ]

        filename = "multiline.txt"
        save_conversation_to_file(conversation, filename, str(tmp_path))

        file_path = tmp_path / filename
        content = file_path.read_text(encoding="utf-8")

        # Newlines in messages should be preserved
        assert "1. Sleep\n2. Appetite\n3. Focus" in content

    def test_conversation_user_chatbot_labels_by_role(self, tmp_path: Path) -> None:
        """Test that conversation uses user:/chatbot: labels by role."""
        conversation = [
            {"speaker": "persona", "response": "I can help with that."},
            {"speaker": "provider", "response": "Thank you for your help."},
        ]

        filename = "speaker_roles.txt"
        save_conversation_to_file(conversation, filename, str(tmp_path))

        file_path = tmp_path / filename
        content = file_path.read_text(encoding="utf-8")

        assert "user:" in content
        assert "chatbot:" in content
