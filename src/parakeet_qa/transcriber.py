"""Real-time transcription using NVIDIA Parakeet TDT."""

import threading
import queue
import time
from typing import Optional, Callable
import numpy as np

import torch
import nemo.collections.asr as nemo_asr


class TranscriberError(Exception):
    """Raised when transcription fails."""
    pass


class ParakeetTranscriber:
    """Real-time transcription using NVIDIA Parakeet TDT model."""

    # Buffer settings for real-time transcription
    MIN_AUDIO_LENGTH = 0.5  # Minimum seconds of audio before transcribing
    MAX_AUDIO_LENGTH = 10.0  # Maximum seconds to buffer before forcing transcription
    SILENCE_THRESHOLD = 0.01  # RMS threshold for silence detection
    SILENCE_DURATION = 0.8  # Seconds of silence to trigger transcription
    SAMPLE_RATE = 16000

    def __init__(
        self,
        callback: Callable[[str], None],
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the transcriber.

        Args:
            callback: Function called with transcribed text.
            model_name: Parakeet model to use from HuggingFace.
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
        """
        self.callback = callback
        self.model_name = model_name
        self._device = device

        self._model: Optional[nemo_asr.models.ASRModel] = None
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._last_voice_time = time.time()
        self._running = False
        self._process_thread: Optional[threading.Thread] = None

    @property
    def device(self) -> str:
        """Get the device to use for inference."""
        if self._device:
            return self._device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """Load the Parakeet model."""
        if self._model is not None:
            return

        try:
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
            self._model = self._model.to(self.device)
            self._model.eval()
        except Exception as e:
            raise TranscriberError(f"Failed to load model '{self.model_name}': {e}")

    def add_audio(self, audio: np.ndarray) -> None:
        """
        Add audio chunk to the buffer for transcription.

        Args:
            audio: Audio samples as float32 numpy array, normalized to [-1, 1].
        """
        if not self._running:
            return

        # Check for voice activity
        rms = np.sqrt(np.mean(audio ** 2))
        has_voice = rms > self.SILENCE_THRESHOLD

        with self._buffer_lock:
            self._audio_buffer.append(audio)

            if has_voice:
                self._last_voice_time = time.time()

    def _get_buffer_duration(self) -> float:
        """Get the current buffer duration in seconds."""
        with self._buffer_lock:
            total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        return total_samples / self.SAMPLE_RATE

    def _process_loop(self) -> None:
        """Background processing loop for transcription."""
        while self._running:
            try:
                buffer_duration = self._get_buffer_duration()
                silence_duration = time.time() - self._last_voice_time

                # Transcribe if we have enough audio and silence, or buffer is full
                should_transcribe = (
                    buffer_duration >= self.MIN_AUDIO_LENGTH and
                    (silence_duration >= self.SILENCE_DURATION or
                     buffer_duration >= self.MAX_AUDIO_LENGTH)
                )

                if should_transcribe:
                    self._transcribe_buffer()
                else:
                    # Small sleep to avoid busy-waiting
                    time.sleep(0.1)

            except Exception as e:
                # Log error but continue processing
                print(f"Transcription error: {e}")
                time.sleep(0.5)

    def _transcribe_buffer(self) -> None:
        """Transcribe the current audio buffer."""
        # Get and clear the buffer
        with self._buffer_lock:
            if not self._audio_buffer:
                return
            audio_chunks = self._audio_buffer.copy()
            self._audio_buffer.clear()

        # Concatenate all chunks
        audio = np.concatenate(audio_chunks)

        if len(audio) < self.SAMPLE_RATE * self.MIN_AUDIO_LENGTH:
            return

        try:
            # Transcribe using Parakeet
            with torch.no_grad():
                # NeMo expects a list of audio files or numpy arrays
                # verbose=False suppresses the progress bar
                transcription = self._model.transcribe([audio], verbose=False)

            if transcription and transcription[0]:
                result = transcription[0]
                # Handle both Hypothesis objects and plain strings
                if hasattr(result, 'text'):
                    text = result.text
                elif hasattr(result, 'strip'):
                    text = result
                else:
                    text = str(result)

                text = text.strip() if text else ""
                if text:
                    self.callback(text)

        except Exception as e:
            print(f"Transcription error: {e}")

    def start(self) -> None:
        """Start the transcription processing loop."""
        if self._running:
            return

        self.load_model()
        self._running = True
        self._last_voice_time = time.time()
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def stop(self) -> None:
        """Stop transcription processing."""
        self._running = False
        if self._process_thread:
            self._process_thread.join(timeout=2)
            self._process_thread = None

        # Process any remaining audio
        if self._audio_buffer:
            self._transcribe_buffer()

    def flush(self) -> None:
        """Force transcription of any buffered audio."""
        self._transcribe_buffer()
