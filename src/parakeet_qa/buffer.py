"""Buffer management for transcriptions and detected questions."""

import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from .question_detector import DetectedQuestion

# Time window for merging consecutive questions (e.g., "How does X work? And how does Y relate?")
QUESTION_MERGE_WINDOW = 3.0  # seconds

# Words that indicate a follow-up question
FOLLOWUP_STARTERS = ["and ", "also ", "what about ", "how about ", "or ", "but "]


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

        Also adds the question to the context buffer. Consecutive questions
        within QUESTION_MERGE_WINDOW that look like follow-ups are merged.

        Args:
            question: The detected question to add.

        Returns:
            True if question was added (or merged), False if it was a duplicate.
        """
        # Avoid duplicate questions (same text within last few entries)
        for existing in list(self._questions)[-5:]:
            if self._is_similar(existing.text, question.text):
                return False

        # Check if we should merge with the most recent question
        if self._questions:
            last_question = self._questions[-1]
            time_diff = question.timestamp - last_question.timestamp

            # If within merge window and new question looks like a follow-up
            if time_diff <= QUESTION_MERGE_WINDOW and self._is_followup(question.text):
                # Merge: combine texts
                merged_text = f"{last_question.text} {question.text}"
                merged_question = DetectedQuestion(
                    text=merged_text,
                    timestamp=last_question.timestamp,
                    confidence=min(last_question.confidence, question.confidence)
                )
                # Replace the last question with merged version
                self._questions[-1] = merged_question
                # Update context too
                self._update_last_context(merged_text)
                return True

        self._questions.append(question)
        self.add_sentence(question.text, is_question=True)
        return True

    def _is_similar(self, text1: str, text2: str, threshold: float = 0.85) -> bool:
        """Check if two texts are similar using simple character ratio."""
        ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return ratio >= threshold

    def _is_followup(self, text: str) -> bool:
        """Check if text looks like a follow-up question."""
        text_lower = text.lower().strip()
        return any(text_lower.startswith(starter) for starter in FOLLOWUP_STARTERS)

    def _update_last_context(self, new_text: str) -> None:
        """Update the last question entry in context with merged text."""
        if self._context:
            for i in range(len(self._context) - 1, -1, -1):
                if self._context[i].is_question:
                    self._context[i].text = new_text
                    break

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
