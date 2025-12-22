"""Ollama LLM client for answering questions."""

from typing import Generator, Optional

import ollama


class OllamaConnectionError(Exception):
    """Raised when Ollama is not reachable."""
    pass


class ModelNotFoundError(Exception):
    """Raised when the specified model is not available."""
    pass


class OllamaClient:
    """Wrapper for Ollama LLM interactions."""

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant answering questions based on a live conversation.
Use the provided conversation context to give accurate, concise answers.
If the context doesn't contain enough information to answer the question fully, acknowledge what you can answer and note what information is missing.
Keep your answers focused and to the point."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Ollama client.

        Args:
            host: Ollama server URL.
            model: Model name to use.
            system_prompt: Custom system prompt (uses default if not provided).
        """
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def check_availability(self) -> bool:
        """
        Check if Ollama is available and model exists.

        Returns:
            True if Ollama is available and model is found.
        """
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            return any(self.model in name or name.startswith(self.model) for name in model_names)
        except Exception:
            return False

    def verify_connection(self) -> None:
        """
        Verify connection to Ollama and model availability.

        Raises:
            OllamaConnectionError: If Ollama is not reachable.
            ModelNotFoundError: If the model is not available.
        """
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
        except Exception as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                f"Ensure Ollama is running: {e}"
            )

        if not any(self.model in name or name.startswith(self.model) for name in model_names):
            available = ", ".join(model_names[:5])
            raise ModelNotFoundError(
                f"Model '{self.model}' not found. "
                f"Available models: {available}"
            )

    def answer_question_stream(
        self,
        question: str,
        context: str
    ) -> Generator[str, None, None]:
        """
        Answer a question and yield response chunks.

        Args:
            question: The question to answer.
            context: Recent conversation context.

        Yields:
            Response text chunks.
        """
        user_content = f"Conversation context:\n{context}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        stream = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            content = chunk.message.content
            if content:
                yield content

    def list_models(self) -> list[str]:
        """
        List available models.

        Returns:
            List of model names.
        """
        try:
            models = self.client.list()
            return [m.model for m in models.models]
        except Exception:
            return []
