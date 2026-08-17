"""Microphone capture controller: mic chunks -> PerceptionService audio pipeline."""

from __future__ import annotations

import asyncio
import logging

from companion.core.events import (
    EVENT_AUDIO_STARTED,
    EVENT_TRANSCRIPT_FINAL,
    EVENT_TRANSCRIPT_PARTIAL,
    EventBus,
)

log = logging.getLogger(__name__)


def _async_callback(fn):
    """Wrap a sync callback so it can be used as an event-bus handler."""
    import inspect

    async def handler(event):
        result = fn(event)
        if inspect.isawaitable(result):
            await result

    return handler


class MicrophoneController:
    """Run the mic -> VAD -> STT chain for a perception service.

    Reads chunks from a bounded queue produced by the capture source and feeds
    them to PerceptionService.process_audio_chunk. Events (audio started/ended,
    transcript partial/final) are published by the perception service onto the
    shared bus; subscribe via add_listener() before start().
    """

    def __init__(self, perception, capture=None, bus: EventBus | None = None,
                 sample_rate: int = 16000, queue_size: int = 20) -> None:
        self.perception = perception
        self.capture = capture
        self.bus = bus or EventBus()
        self.sample_rate = sample_rate
        self.queue = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task | None = None
        self._running = False

    def on_audio_started(self, callback) -> None:
        self.bus.subscribe(EVENT_AUDIO_STARTED, _async_callback(lambda _e: callback()))

    def on_transcript_partial(self, callback) -> None:
        self.bus.subscribe(
            EVENT_TRANSCRIPT_PARTIAL,
            _async_callback(lambda e: callback(e.payload.get("text", ""))),
        )

    def on_transcript_final(self, callback) -> None:
        self.bus.subscribe(
            EVENT_TRANSCRIPT_FINAL,
            _async_callback(lambda e: callback(
                e.payload.get("text", ""), e.payload.get("language", "auto")
            )),
        )

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self.capture is not None:
            self.capture.start(self.queue, self.sample_rate)
        self._task = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        while self._running:
            try:
                chunk = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                await self.perception.process_audio_chunk(chunk)
            except Exception as exc:
                log.warning("audio chunk processing failed: %s", exc)

    def stop(self) -> None:
        self._running = False
        if self.capture is not None:
            self.capture.stop()
        if self._task is not None:
            self._task.cancel()

    async def aclose(self) -> None:
        """Stop capture and wait for its consumer task to finish."""
        self.stop()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
