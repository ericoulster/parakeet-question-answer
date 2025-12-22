"""CLI interface for Parakeet Question Answerer."""

import os
import select
import sys
import threading
from pathlib import Path
from typing import Optional

import click

from .audio_capture import AudioCapture, AudioCaptureError, NoMonitorDeviceError
from .buffer import TranscriptionBuffer
from .config import Config
from .display import Display
from .ollama_client import ModelNotFoundError, OllamaClient, OllamaConnectionError
from .question_detector import QuestionDetector
from .transcriber import ParakeetTranscriber, TranscriberError


class Application:
    """Main application class."""

    def __init__(self, config: Config, display: Display):
        """Initialize the application."""
        self.config = config
        self.display = display
        self.buffer = TranscriptionBuffer(
            context_size=config.context_buffer_size,
            question_size=config.question_buffer_size
        )
        self.detector = QuestionDetector(config.question_words)
        custom_prompt = config.load_prompt()
        self.ollama = OllamaClient(
            config.ollama_host,
            config.ollama_model,
            system_prompt=custom_prompt
        )

        self._transcriber: Optional[ParakeetTranscriber] = None
        self._audio_capture: Optional[AudioCapture] = None
        self._running = False
        self._show_transcription = config.show_transcription
        self._transcription_lock = threading.Lock()

    def _on_transcription(self, text: str) -> None:
        """Handle incoming transcription text."""
        with self._transcription_lock:
            if self._show_transcription:
                self.display.show_transcription(text)

            sentences, questions = self.detector.process_text(text)

            # Add all sentences to context buffer
            for sentence in sentences:
                is_question = any(q.text == sentence for q in questions)
                if not is_question:
                    self.buffer.add_sentence(sentence)

            # Add detected questions
            for q in questions:
                if self.buffer.add_question(q):
                    self.display.show_new_question(q)

    def _check_connections(self) -> None:
        """Verify all required connections."""
        # Check Ollama
        try:
            self.ollama.verify_connection()
        except OllamaConnectionError as e:
            self.display.show_status(str(e), "error")
            sys.exit(1)
        except ModelNotFoundError as e:
            self.display.show_status(str(e), "error")
            available = self.ollama.list_models()
            if available:
                self.display.show_status(f"Try: --model {available[0]}", "info")
            sys.exit(1)

        # Initialize Parakeet transcriber
        self.display.show_status("Loading Parakeet ASR model (this may take a moment)...", "info")
        try:
            self._transcriber = ParakeetTranscriber(
                callback=self._on_transcription,
                model_name=self.config.parakeet_model
            )
            self._transcriber.load_model()
            self.display.show_status(f"ASR model loaded: {self.config.parakeet_model}", "success")
        except TranscriberError as e:
            self.display.show_status(str(e), "error")
            sys.exit(1)

        # Initialize audio capture
        try:
            self._audio_capture = AudioCapture(
                callback=self._transcriber.add_audio,
                device=self.config.audio_device
            )
            # Show which device will be used
            device = self.config.audio_device or self._audio_capture._find_monitor_device()
            self.display.show_status(f"Audio device: {device}", "success")
        except AudioCaptureError as e:
            self.display.show_status(str(e), "error")
            sys.exit(1)

    def _show_questions_menu(self) -> None:
        """Show questions and allow selection."""
        questions = self.buffer.get_questions()
        self.display.show_questions(questions)

        if not questions:
            return

        try:
            choice = click.prompt(
                "Select question number (0 to cancel)",
                type=int,
                default=0
            )
        except click.Abort:
            return

        if 0 < choice <= len(questions):
            selected = questions[choice - 1]
            context = self.buffer.get_context()

            self.display.print_answer_header(selected.text)

            try:
                for chunk in self.ollama.answer_question_stream(selected.text, context):
                    self.display.print_answer_chunk(chunk)
                self.display.print_answer_footer()
            except Exception as e:
                self.display.show_status(f"Error getting answer: {e}", "error")

    def _has_tty(self) -> bool:
        """Check if we have a TTY for interactive input."""
        try:
            return os.isatty(sys.stdin.fileno())
        except (AttributeError, OSError):
            return False

    def _get_input_interactive(self) -> Optional[str]:
        """Get input using click.getchar() for interactive terminals."""
        try:
            return click.getchar()
        except (OSError, EOFError):
            return None

    def _get_input_line(self) -> Optional[str]:
        """Get input using line-based input for non-TTY environments."""
        try:
            if select.select([sys.stdin], [], [], 0.5)[0]:
                line = sys.stdin.readline().strip().lower()
                return line[0] if line else None
        except (OSError, ValueError):
            pass
        return None

    def run(self) -> None:
        """Run the main application loop."""
        # Verify connections first
        self._check_connections()

        self._running = True
        is_interactive = self._has_tty()

        if is_interactive:
            self.display.show_listening(
                self.config.ollama_model,
                self.config.parakeet_model
            )
        else:
            self.display.show_status(
                f"Listening (LLM: {self.config.ollama_model}, ASR: {self.config.parakeet_model})",
                "info"
            )
            self.display.show_status(
                "Type 'q' + Enter for questions, 'c' + Enter to clear, 'h' + Enter for help",
                "info"
            )

        # Start audio capture and transcription
        self._transcriber.start()
        self._audio_capture.start()

        # Main interaction loop
        try:
            while self._running:
                try:
                    if is_interactive:
                        key = self._get_input_interactive()
                    else:
                        key = self._get_input_line()

                    if key is None:
                        continue

                    if key.lower() == "q":
                        self._show_questions_menu()
                    elif key.lower() == "c":
                        self.buffer.clear_questions()
                        self.display.show_status("Questions cleared", "success")
                    elif key.lower() == "t":
                        self._show_transcription = not self._show_transcription
                        status = "on" if self._show_transcription else "off"
                        self.display.show_status(f"Live transcription: {status}", "info")
                    elif key.lower() == "h":
                        self.display.show_help()
                    elif key == "\x03":  # Ctrl+C
                        raise KeyboardInterrupt

                except (EOFError, KeyboardInterrupt):
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            self.display.show_status("\nShutting down...", "info")

        finally:
            self._running = False
            if self._audio_capture:
                self._audio_capture.stop()
            if self._transcriber:
                self._transcriber.stop()


def _load_config(
    config: Optional[str],
    model: Optional[str],
    host: Optional[str],
    asr_model: Optional[str],
    audio_device: Optional[str],
    prompt: Optional[str],
    show_transcription: bool,
) -> Config:
    """Load and merge configuration."""
    # Load config from file or defaults
    if config:
        cfg = Config.from_file(Path(config))
    else:
        default_path = Config.default_config_path()
        if default_path.exists():
            cfg = Config.from_file(default_path)
        else:
            cfg = Config()

    # Override with CLI args
    cfg.merge_cli_args(
        model=model,
        host=host,
        asr_model=asr_model,
        audio_device=audio_device,
        prompt_file=prompt,
        show_transcription=show_transcription if show_transcription else None
    )
    return cfg


@click.group(invoke_without_command=True)
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--model", "-m", help="Ollama model to use")
@click.option("--host", "-H", help="Ollama host URL")
@click.option("--asr-model", help="Parakeet ASR model (default: nvidia/parakeet-tdt-0.6b-v2)")
@click.option("--audio-device", help="Audio device to capture from")
@click.option("--prompt", "-p", type=click.Path(exists=True), help="Custom system prompt file")
@click.option("--show-transcription", "-t", is_flag=True, help="Show live transcription")
@click.pass_context
def cli(
    ctx: click.Context,
    config: Optional[str],
    model: Optional[str],
    host: Optional[str],
    asr_model: Optional[str],
    audio_device: Optional[str],
    prompt: Optional[str],
    show_transcription: bool,
) -> None:
    """Parakeet Question Answerer - Answer questions from real-time audio using NVIDIA Parakeet and Ollama."""
    ctx.ensure_object(dict)

    cfg = _load_config(config, model, host, asr_model, audio_device, prompt, show_transcription)
    ctx.obj["config"] = cfg
    ctx.obj["display"] = Display()

    # If no subcommand provided, run listen by default
    if ctx.invoked_subcommand is None:
        app = Application(cfg, ctx.obj["display"])
        app.run()


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--model", "-m", help="Ollama model to use")
@click.option("--host", "-H", help="Ollama host URL")
@click.option("--asr-model", help="Parakeet ASR model (default: nvidia/parakeet-tdt-0.6b-v2)")
@click.option("--audio-device", help="Audio device to capture from")
@click.option("--prompt", "-p", type=click.Path(exists=True), help="Custom system prompt file")
@click.option("--show-transcription", "-t", is_flag=True, help="Show live transcription")
def listen(
    config: Optional[str],
    model: Optional[str],
    host: Optional[str],
    asr_model: Optional[str],
    audio_device: Optional[str],
    prompt: Optional[str],
    show_transcription: bool,
) -> None:
    """Start listening to system audio and detecting questions."""
    cfg = _load_config(config, model, host, asr_model, audio_device, prompt, show_transcription)
    display = Display()
    app = Application(cfg, display)
    app.run()


@cli.command()
@click.pass_context
def models(ctx: click.Context) -> None:
    """List available Ollama models."""
    config = ctx.obj["config"]
    display = ctx.obj["display"]

    client = OllamaClient(host=config.ollama_host)

    try:
        available = client.list_models()
    except Exception as e:
        display.show_status(f"Cannot connect to Ollama: {e}", "error")
        sys.exit(1)

    if not available:
        display.show_status("No models found", "warning")
        return

    display.console.print("\n[bold]Available Models:[/bold]")
    for model_name in available:
        marker = "*" if config.ollama_model in model_name else " "
        display.console.print(f"  {marker} {model_name}")
    display.console.print()
    display.console.print("[dim]* = currently selected[/dim]")


@cli.command()
@click.pass_context
def devices(ctx: click.Context) -> None:
    """List available audio devices."""
    display = ctx.obj["display"]

    capture = AudioCapture(callback=lambda x: None)
    available = capture.get_available_devices()

    if not available:
        display.show_status("No audio devices found", "warning")
        return

    display.console.print("\n[bold]Available Audio Sources:[/bold]")
    for name, desc in available:
        marker = "[green]*[/green]" if ".monitor" in name else " "
        display.console.print(f"  {marker} {name}")
        display.console.print(f"      [dim]{desc}[/dim]")
    display.console.print()
    display.console.print("[dim]* = monitor device (captures system audio)[/dim]")


@cli.command()
@click.pass_context
def check(ctx: click.Context) -> None:
    """Check connectivity to Ollama and audio devices."""
    config = ctx.obj["config"]
    display = ctx.obj["display"]

    display.console.print("\n[bold]Connectivity Check[/bold]\n")

    # Check Ollama
    display.console.print(f"Ollama ({config.ollama_host})... ", end="")
    client = OllamaClient(host=config.ollama_host, model=config.ollama_model)
    try:
        client.verify_connection()
        display.console.print("[green]OK[/green]")
        display.console.print(f"  Model: {config.ollama_model}")
    except OllamaConnectionError:
        display.console.print("[red]FAILED[/red]")
        display.console.print("  [red]Cannot connect to Ollama[/red]")
    except ModelNotFoundError:
        display.console.print("[yellow]PARTIAL[/yellow]")
        display.console.print(f"  [yellow]Model '{config.ollama_model}' not found[/yellow]")

    # Check audio
    display.console.print("Audio capture (PulseAudio/PipeWire)... ", end="")
    capture = AudioCapture(callback=lambda x: None)
    try:
        device = capture._find_monitor_device()
        display.console.print("[green]OK[/green]")
        display.console.print(f"  Monitor device: {device}")
    except NoMonitorDeviceError:
        display.console.print("[red]FAILED[/red]")
        display.console.print("  [red]No monitor device found[/red]")
    except AudioCaptureError as e:
        display.console.print("[red]FAILED[/red]")
        display.console.print(f"  [red]{e}[/red]")

    # Check CUDA
    display.console.print("CUDA (for Parakeet)... ", end="")
    try:
        import torch
        if torch.cuda.is_available():
            display.console.print("[green]OK[/green]")
            display.console.print(f"  Device: {torch.cuda.get_device_name(0)}")
        else:
            display.console.print("[yellow]NOT AVAILABLE[/yellow]")
            display.console.print("  [yellow]Will use CPU (slower)[/yellow]")
    except Exception as e:
        display.console.print("[red]ERROR[/red]")
        display.console.print(f"  [red]{e}[/red]")

    display.console.print()
