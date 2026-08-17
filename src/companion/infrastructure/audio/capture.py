"""Microphone capture source built on sounddevice."""

from __future__ import annotations

import asyncio
import logging

from companion.core.contracts import AudioInput
from companion.core.errors import AudioDeviceUnavailableError
from companion.infrastructure.audio import resample_pcm

log = logging.getLogger(__name__)

CHUNK_MS = 20
TARGET_RATE = 16000


class MicrophoneCaptureSource:
    """Push 20 ms PCM chunks from the default mic into an asyncio queue.

    Capture runs in a sounddevice callback thread; chunks are handed to the
    loop via call_soon_threadsafe so the caller only awaits the queue.
    """

    def __init__(self, device: str | int | None = None,
                 sample_rate: int = TARGET_RATE, chunk_ms: int = CHUNK_MS) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self._stream = None
        self._queue = None
        self._loop = None
        self._device_rate = sample_rate

    def _info(self) -> str:
        try:
            import sounddevice as sd

            dev = sd.query_devices(kind="input")
            return f"{dev.get('name', 'unknown')} @ {dev.get('default_samplerate', '?')} Hz"
        except Exception:
            return "unknown"

    def start(self, queue, sample_rate: int | None = None) -> None:
        if sample_rate:
            self.sample_rate = sample_rate
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceUnavailableError(
                f"sounddevice not installed (mic '{self._info()}'): {exc}"
            ) from exc

        self._queue = queue
        self._loop = asyncio.get_running_loop() if self._queue is not None else None
        chunks = max(1, self.chunk_ms * self.sample_rate // 1000)
        started = False
        last_err = None
        for rate in (self.sample_rate, 44100, 48000):
            try:
                self._stream = sd.RawInputStream(
                    samplerate=rate,
                    blocksize=chunks * rate // self.sample_rate,
                    channels=1,
                    dtype="int16",
                    device=self.device,
                    callback=self._callback,
                )
                self._stream.start()
                self._device_rate = rate
                started = True
                break
            except Exception as exc:
                last_err = exc
                self._stream = None
        if not started:
            raise AudioDeviceUnavailableError(
                f"cannot open input stream on mic '{self._info()}': {last_err}"
            )
        log.info("mic capture started: %s @ %d Hz (target %d Hz)",
                 self._info(), self._device_rate, self.sample_rate)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("audio capture status: %s", status)
        samples = bytes(indata)
        if self._device_rate != self.sample_rate:
            samples = resample_pcm(samples, self._device_rate, self.sample_rate)
        chunk = AudioInput(samples=samples, sample_rate=self.sample_rate, source="mic")
        if self._loop is not None and self._queue is not None:
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
            except RuntimeError:
                pass  # loop closed during shutdown

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        log.info("mic capture stopped")


class NullCaptureSource:
    """No mic available: feeds the pipeline nothing (silence)."""

    def __init__(self) -> None:
        self._stream = None

    def start(self, queue, sample_rate: int = TARGET_RATE) -> None:
        self._stream = None

    def stop(self) -> None:
        pass
