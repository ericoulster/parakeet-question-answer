"""Rich terminal display for the application."""

import time
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .question_detector import DetectedQuestion


class Display:
    """Rich terminal display for the application."""

    def __init__(self):
        """Initialize the display."""
        self.console = Console()

    def show_questions(self, questions: list[DetectedQuestion]) -> None:
        """
        Display questions in a numbered table.

        Args:
            questions: List of detected questions to display.
        """
        if not questions:
            self.console.print("[yellow]No questions detected yet.[/yellow]")
            return

        table = Table(title="Detected Questions", show_lines=True, expand=True)
        table.add_column("#", style="cyan", width=4, justify="right")
        table.add_column("Question", style="white", ratio=1)
        table.add_column("Conf", style="green", width=6, justify="center")
        table.add_column("Time", style="dim", width=10)

        for i, q in enumerate(questions, 1):
            time_str = time.strftime("%H:%M:%S", time.localtime(q.timestamp))
            conf_str = f"{q.confidence:.0%}"
            table.add_row(str(i), q.text, conf_str, time_str)

        self.console.print(table)

    def print_answer_header(self, question: str) -> None:
        """
        Print the header for a streaming answer.

        Args:
            question: The question being answered.
        """
        self.console.print()
        self.console.print(f"[bold blue]Q: {question}[/bold blue]")
        self.console.print("[dim]" + "-" * 60 + "[/dim]")

    def print_answer_chunk(self, chunk: str) -> None:
        """
        Print a chunk of the streaming answer.

        Args:
            chunk: Text chunk to print.
        """
        self.console.print(chunk, end="")

    def print_answer_footer(self) -> None:
        """Print the footer after a streaming answer."""
        self.console.print()
        self.console.print("[dim]" + "-" * 60 + "[/dim]")
        self.console.print()

    def show_status(self, message: str, style: str = "info") -> None:
        """
        Show a status message.

        Args:
            message: Message to display.
            style: One of "info", "success", "warning", "error".
        """
        styles = {
            "info": "[blue]",
            "success": "[green]",
            "warning": "[yellow]",
            "error": "[red bold]"
        }
        prefix = styles.get(style, "[blue]")
        self.console.print(f"{prefix}{message}[/]")

    def show_new_question(self, question: DetectedQuestion) -> None:
        """
        Show notification of a new question detected.

        Args:
            question: The newly detected question.
        """
        text = question.text
        if len(text) > 60:
            text = text[:57] + "..."
        self.console.print(f"[cyan]New question:[/cyan] {text}")

    def show_transcription(self, text: str) -> None:
        """
        Show live transcription text.

        Args:
            text: Transcribed text to display.
        """
        self.console.print(f"[dim]{text}[/dim]")

    def show_help(self) -> None:
        """Display available commands."""
        help_text = """
[bold]Available Commands:[/bold]

  [cyan]q[/cyan]  - Show detected questions and select one to answer
  [cyan]c[/cyan]  - Clear question buffer
  [cyan]t[/cyan]  - Toggle live transcription display
  [cyan]h[/cyan]  - Show this help message
  [cyan]Ctrl+C[/cyan] - Exit the application
"""
        self.console.print(Panel(help_text.strip(), title="Help", border_style="blue"))

    def show_listening(self, model: str, asr_model: str) -> None:
        """
        Show the listening status message.

        Args:
            model: The Ollama model being used.
            asr_model: The ASR model being used.
        """
        self.console.print()
        self.console.print(Panel(
            f"[green]Listening for audio...[/green]\n"
            f"ASR Model: [cyan]{asr_model}[/cyan]\n"
            f"LLM Model: [cyan]{model}[/cyan]\n\n"
            f"Press [bold]q[/bold] to view questions, [bold]t[/bold] to toggle transcription, "
            f"[bold]h[/bold] for help, [bold]Ctrl+C[/bold] to quit",
            title="Parakeet Q&A",
            border_style="green"
        ))
        self.console.print()

    def clear(self) -> None:
        """Clear the console."""
        self.console.clear()
