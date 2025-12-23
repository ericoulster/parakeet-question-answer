"""Question detection in streaming transcription text."""

import re
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectedQuestion:
    """A detected question from the transcription stream."""

    text: str
    timestamp: float
    confidence: float  # 0.0-1.0 based on detection method

    def __str__(self) -> str:
        return self.text


# Imperative words that indicate a request for information (implicit questions)
DEFAULT_REQUEST_WORDS = [
    "explain", "describe", "tell", "define", "elaborate", "clarify",
    "compare", "contrast", "discuss", "outline", "summarize", "list",
    "give", "show", "demonstrate", "illustrate", "walk", "help"
]


class QuestionDetector:
    """Detects questions in streaming transcription text."""

    def __init__(
        self,
        question_words: Optional[list[str]] = None,
        request_words: Optional[list[str]] = None
    ):
        """
        Initialize the question detector.

        Args:
            question_words: List of words that typically start questions.
            request_words: List of imperative words that indicate information requests.
        """
        default_words = [
            "what", "where", "when", "how", "why", "who", "which",
            "do", "does", "did", "is", "are", "was", "were",
            "can", "could", "would", "should", "will", "shall",
            "have", "has", "had"
        ]
        self.question_words = set(w.lower() for w in (question_words or default_words))
        self.request_words = set(w.lower() for w in (request_words or DEFAULT_REQUEST_WORDS))
        self._partial_sentence = ""

        # Regex for sentence boundary detection
        self._sentence_end_pattern = re.compile(r'([.!?]+)\s*')

    def _normalize_text(self, text: str) -> str:
        """Normalize text by collapsing multiple spaces and trimming."""
        return ' '.join(text.split())

    def process_text(self, text: str) -> tuple[list[str], list[DetectedQuestion]]:
        """
        Process incoming transcription text and return sentences and detected questions.

        Handles streaming text by maintaining partial sentence state.

        Args:
            text: Incoming transcription text chunk.

        Returns:
            Tuple of (all_sentences, detected_questions).
        """
        sentences = []
        questions = []

        # Normalize spacing
        text = self._normalize_text(text)

        if not text:
            return sentences, questions

        # Combine with partial buffer
        full_text = self._partial_sentence + (" " if self._partial_sentence else "") + text
        full_text = self._normalize_text(full_text)

        # Split into sentences while preserving delimiters
        parts = self._sentence_end_pattern.split(full_text)

        # Process pairs of (text, delimiter)
        i = 0
        while i < len(parts) - 1:
            sentence_text = parts[i]
            delimiter = parts[i + 1] if i + 1 < len(parts) else ""

            if sentence_text.strip():
                full_sentence = self._normalize_text(sentence_text.strip()) + delimiter.strip()
                sentences.append(full_sentence)

                # Check if it's a question
                question = self._detect_question(full_sentence)
                if question:
                    questions.append(question)

            i += 2

        # Keep the last part as partial (might be incomplete)
        if i < len(parts):
            self._partial_sentence = parts[i].strip()
        else:
            self._partial_sentence = ""

        return sentences, questions

    def _detect_question(self, sentence: str) -> Optional[DetectedQuestion]:
        """
        Determine if a sentence is a question or information request.

        Args:
            sentence: Complete sentence to analyze.

        Returns:
            DetectedQuestion if it's a question or request, None otherwise.
        """
        sentence_clean = sentence.strip()

        if not sentence_clean:
            return None

        # Primary method: explicit question mark
        if sentence_clean.endswith("?"):
            return DetectedQuestion(
                text=sentence_clean,
                timestamp=time.time(),
                confidence=1.0
            )

        # Only check word-based detection if sentence ends with punctuation
        if sentence_clean[-1] in ".!":
            first_word = sentence_clean.split()[0].lower().rstrip(".,!?")

            # Secondary method: starts with question word
            if first_word in self.question_words:
                return DetectedQuestion(
                    text=sentence_clean,
                    timestamp=time.time(),
                    confidence=0.7
                )

            # Tertiary method: starts with imperative request word
            # e.g., "Explain the difference between arrays and linked lists."
            if first_word in self.request_words:
                return DetectedQuestion(
                    text=sentence_clean,
                    timestamp=time.time(),
                    confidence=0.8
                )

        return None

    def reset(self) -> None:
        """Reset the partial sentence buffer."""
        self._partial_sentence = ""

    @property
    def partial_sentence(self) -> str:
        """Get the current partial sentence buffer."""
        return self._partial_sentence
