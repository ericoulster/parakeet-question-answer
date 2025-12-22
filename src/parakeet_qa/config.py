"""Configuration management for Parakeet Question Answerer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


DEFAULT_QUESTION_WORDS = [
    "what", "where", "when", "how", "why", "who", "which",
    "do", "does", "did", "is", "are", "was", "were",
    "can", "could", "would", "should", "will", "shall",
    "have", "has", "had"
]

# Default prompt file name (looked for in package root directory)
DEFAULT_PROMPT_FILENAME = "prompt.txt"


@dataclass
class Config:
    """Application configuration."""

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Parakeet settings
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v2"
    audio_device: Optional[str] = None  # None = auto-detect monitor

    # Buffer settings
    question_buffer_size: int = 20
    context_buffer_size: int = 50

    # Detection settings
    question_words: list[str] = field(default_factory=lambda: DEFAULT_QUESTION_WORDS.copy())

    # Display settings
    show_transcription: bool = False

    # Custom prompt file (optional)
    prompt_file: Optional[Path] = None

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        # Ollama settings
        if "ollama" in data:
            ollama = data["ollama"]
            if "host" in ollama:
                config.ollama_host = ollama["host"]
            if "model" in ollama:
                config.ollama_model = ollama["model"]

        # Parakeet settings
        if "parakeet" in data:
            parakeet = data["parakeet"]
            if "model" in parakeet:
                config.parakeet_model = parakeet["model"]
            if "audio_device" in parakeet:
                config.audio_device = parakeet["audio_device"]

        # Buffer settings
        if "buffers" in data:
            buffers = data["buffers"]
            if "question_size" in buffers:
                config.question_buffer_size = buffers["question_size"]
            if "context_size" in buffers:
                config.context_buffer_size = buffers["context_size"]

        # Detection settings
        if "detection" in data:
            detection = data["detection"]
            if "question_words" in detection:
                config.question_words = detection["question_words"]

        # Display settings
        if "display" in data:
            display = data["display"]
            if "show_transcription" in display:
                config.show_transcription = display["show_transcription"]

        # Prompt file
        if "prompt_file" in data and data["prompt_file"]:
            config.prompt_file = Path(data["prompt_file"]).expanduser()

        return config

    @classmethod
    def default_config_path(cls) -> Path:
        """Get the default configuration file path."""
        return Path.home() / ".config" / "parakeet-qa" / "config.yaml"

    def merge_cli_args(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        asr_model: Optional[str] = None,
        audio_device: Optional[str] = None,
        prompt_file: Optional[str] = None,
        show_transcription: Optional[bool] = None,
    ) -> "Config":
        """Merge CLI arguments into config (CLI args take precedence)."""
        if model is not None:
            self.ollama_model = model
        if host is not None:
            self.ollama_host = host
        if asr_model is not None:
            self.parakeet_model = asr_model
        if audio_device is not None:
            self.audio_device = audio_device
        if prompt_file is not None:
            self.prompt_file = Path(prompt_file).expanduser()
        if show_transcription is not None:
            self.show_transcription = show_transcription
        return self

    @classmethod
    def get_package_root(cls) -> Path:
        """Get the root directory of the package (where pyproject.toml lives)."""
        # Start from this file's directory and go up to find pyproject.toml
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        # Fallback to the src parent directory
        return Path(__file__).parent.parent.parent

    def load_prompt(self) -> Optional[str]:
        """Load custom prompt from file if configured, or from default prompt.txt.

        Priority:
        1. Explicitly configured prompt_file (via CLI or config)
        2. Default prompt.txt in the package root directory
        """
        # First check for explicitly configured prompt file
        if self.prompt_file and self.prompt_file.exists():
            return self.prompt_file.read_text().strip()

        # If no custom prompt configured, check for default prompt.txt in repo root
        if self.prompt_file is None:
            default_prompt_path = self.get_package_root() / DEFAULT_PROMPT_FILENAME
            if default_prompt_path.exists():
                return default_prompt_path.read_text().strip()

        return None
