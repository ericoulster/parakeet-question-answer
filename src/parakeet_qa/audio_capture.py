"""Audio capture from system audio using PulseAudio/PipeWire monitors."""

import subprocess
import threading
import queue
from typing import Optional, Callable
import numpy as np


class AudioCaptureError(Exception):
    """Raised when audio capture fails."""
    pass


class NoMonitorDeviceError(AudioCaptureError):
    """Raised when no monitor device is found."""
    pass


class AudioCapture:
    """Captures system audio using PulseAudio/PipeWire monitor devices."""

    SAMPLE_RATE = 16000  # Parakeet expects 16kHz
    CHANNELS = 1
    CHUNK_DURATION = 0.5  # seconds per chunk

    def __init__(self, callback: Callable[[np.ndarray], None], device: Optional[str] = None):
        """
        Initialize audio capture.

        Args:
            callback: Function called with audio chunks (numpy float32 arrays).
            device: Specific PulseAudio source to use. If None, auto-detects monitor.
        """
        self.callback = callback
        self.device = device
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None

    def _find_monitor_device(self) -> str:
        """Find a PulseAudio/PipeWire monitor device for system audio capture."""
        try:
            # First, try to get the default sink and use its monitor
            info_result = subprocess.run(
                ["pactl", "info"],
                capture_output=True,
                text=True,
                timeout=5
            )
            default_sink = None
            if info_result.returncode == 0:
                for line in info_result.stdout.split("\n"):
                    if line.startswith("Default Sink:"):
                        default_sink = line.split(":", 1)[1].strip()
                        break

            # If we found a default sink, its monitor is sink_name.monitor
            if default_sink:
                default_monitor = f"{default_sink}.monitor"
                # Verify it exists
                result = subprocess.run(
                    ["pactl", "list", "sources", "short"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if default_monitor in line:
                            return default_monitor

            # Fallback: find any monitor device
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise AudioCaptureError(f"pactl failed: {result.stderr}")

            # Look for monitor devices (they capture system output)
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    source_name = parts[1]
                    # Monitor devices typically have .monitor suffix
                    if ".monitor" in source_name:
                        return source_name

            raise NoMonitorDeviceError(
                "No monitor device found. Ensure PulseAudio/PipeWire is running "
                "and you have audio output devices available."
            )

        except subprocess.TimeoutExpired:
            raise AudioCaptureError("Timeout detecting audio devices")
        except FileNotFoundError:
            raise AudioCaptureError("pactl not found. Install pulseaudio-utils or pipewire-pulse.")

    def _capture_loop(self) -> None:
        """Main capture loop using parec (PulseAudio recorder)."""
        device = self.device or self._find_monitor_device()

        chunk_samples = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        bytes_per_chunk = chunk_samples * 2  # 16-bit = 2 bytes per sample

        # Use parec to capture audio from the monitor device
        cmd = [
            "parec",
            "--device", device,
            "--rate", str(self.SAMPLE_RATE),
            "--channels", str(self.CHANNELS),
            "--format", "s16le",  # 16-bit signed little-endian
            "--latency-msec", "100",
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            while self._running and self._process.poll() is None:
                data = self._process.stdout.read(bytes_per_chunk)
                if not data:
                    break

                # Convert to numpy float32 array normalized to [-1, 1]
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                if len(audio) > 0:
                    self.callback(audio)

        except Exception as e:
            if self._running:
                raise AudioCaptureError(f"Audio capture error: {e}")
        finally:
            self._cleanup_process()

    def _cleanup_process(self) -> None:
        """Clean up the subprocess."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except:
                self._process.kill()
            self._process = None

    def start(self) -> None:
        """Start audio capture in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False
        self._cleanup_process()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def get_available_devices(self) -> list[tuple[str, str]]:
        """
        Get list of available audio sources.

        Returns:
            List of (device_name, description) tuples.
        """
        try:
            result = subprocess.run(
                ["pactl", "list", "sources"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return []

            devices = []
            current_name = None
            current_desc = None

            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.startswith("Name:"):
                    current_name = line.split(":", 1)[1].strip()
                elif line.startswith("Description:"):
                    current_desc = line.split(":", 1)[1].strip()
                    if current_name:
                        devices.append((current_name, current_desc or current_name))
                    current_name = None
                    current_desc = None

            return devices

        except:
            return []
