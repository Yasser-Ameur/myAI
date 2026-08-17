"""Audio playback sink built on sounddevice (non-blocking, interruptible)."""

from __future__ import annotations

import logging
import threading

from companion.core.errors import AudioDeviceUnavailableError

log = logging.getLogger(__name__)


class SoundDevicePlaybackSink:
    """Play mono PCM int16 to the default output device.

    play() writes into a blocking stream on the calling thread so callers
    naturally wait; stop() from another thread interrupts playback.
    """

    def __init__(self, device: str | int | None = None) -> None:
        self.device = device
        self._stream = None
        self._stop = threading.Event()
        self._active = threading.Event()

    def _ensure_stream(self, sample_rate: int) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceUnavailableError(f"sounddevice not installed: {exc}") from exc
        try:
            self._stream = sd.RawOutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                device=self.device,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioDeviceUnavailableError(f"cannot open output stream: {exc}") from exc

    def play(self, samples: bytes, sample_rate: int) -> None:
        if not samples:
            return
        try:
            self._ensure_stream(sample_rate)
        except AudioDeviceUnavailableError as exc:
            log.warning("playback unavailable, dropping audio: %s", exc)
            return
        self._stop.clear()
        self._active.set()
        stream = self._stream
        try:
            if stream.samplerate != sample_rate:
                from companion.infrastructure.audio import resample_pcm

                samples = resample_pcm(samples, sample_rate, stream.samplerate)
            stream.write(samples)
        except Exception as exc:
            log.warning("playback failed: %s", exc)
        finally:
            self._active.clear()

    def stop(self) -> None:
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        self._stream = None
        self._active.clear()

    def is_playing(self) -> bool:
        return self._active.is_set()

    def close(self) -> None:
        self.stop()


class NullPlaybackSink:
    """No audio output: records that audio was produced but plays nothing."""

    def __init__(self) -> None:
        self.last_duration_ms = 0.0
        self._active = False

    def play(self, samples: bytes, sample_rate: int) -> None:
        duration = len(samples) * 1000.0 / max(1, sample_rate * 2)
        self.last_duration_ms = duration
        self._active = True
        self._active = False

    def stop(self) -> None:
        self._active = False

    def is_playing(self) -> bool:
        return self._active

    def close(self) -> None:
        pass
