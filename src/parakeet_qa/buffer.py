"""Buffer management for transcriptions and detected questions."""

import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from .question_detector import DetectedQuestion


@dataclass
class BufferedSentence:
    """A sentence stored in the context buffer."""

    text: str
    timestamp: float
    is_question: bool = False


class TranscriptionBuffer:
    """Manages transcription history for context and question tracking."""

    def __init__(self, context_size: int = 50, question_size: int = 20):
        """
        Initialize the buffer.

        Args:
            context_size: Maximum number of sentences to keep for context.
            question_size: Maximum number of questions to keep.
        """
        self._context: deque[BufferedSentence] = deque(maxlen=context_size)
        self._questions: deque[DetectedQuestion] = deque(maxlen=question_size)

    def add_sentence(self, text: str, is_question: bool = False) -> None:
        """
        Add a sentence to the context buffer.

        Args:
            text: The sentence text.
            is_question: Whether this sentence is a question.
        """
        content = BufferedSentence(
            text=text,
            timestamp=time.time(),
            is_question=is_question
        )
        self._context.append(content)

    def add_question(self, question: DetectedQuestion) -> bool:
        """
        Add a detected question to the question buffer.

        Also adds the question to the context buffer.

        Args:
            question: The detected question to add.

        Returns:
            True if question was added, False if it was a duplicate.
        """
        # Avoid duplicate questions (same text within last few entries)
        for existing in list(self._questions)[-5:]:
            if self._is_similar(existing.text, question.text):
                return False

        self._questions.append(question)
        self.add_sentence(question.text, is_question=True)
        return True

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.85) -> bool:
        """Check if two texts are similar using simple character ratio."""
        ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return ratio >= threshold

    def get_questions(self) -> list[DetectedQuestion]:
        """
        Get all buffered questions, most recent first.

        Returns:
            List of detected questions, newest first.
        """
        return list(reversed(self._questions))

    def get_context(self, max_chars: int = 2000) -> str:
        """
        Get conversation context as a single string.

        Args:
            max_chars: Maximum characters to return (truncates from beginning).

        Returns:
            Conversation context as a string.
        """
        items = list(self._context)
        context = " ".join(c.text for c in items)

        # Truncate from the beginning if too long
        if len(context) > max_chars:
            context = "..." + context[-(max_chars - 3):]

        return context

    def get_recent_context(self, num_sentences: int = 10) -> str:
        """
        Get the most recent sentences as context.

        Args:
            num_sentences: Number of recent sentences to include.

        Returns:
            Recent conversation context.
        """
        items = list(self._context)[-num_sentences:]
        return " ".join(c.text for c in items)

    def clear_questions(self) -> None:
        """Clear the question buffer."""
        self._questions.clear()

    def clear_all(self) -> None:
        """Clear both context and question buffers."""
        self._context.clear()
        self._questions.clear()

    @property
    def question_count(self) -> int:
        """Get the number of buffered questions."""
        return len(self._questions)

    @property
    def context_count(self) -> int:
        """Get the number of buffered sentences."""
        return len(self._context)
