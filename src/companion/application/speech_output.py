"""Speech output service.

Synthesizes response text through the configured TTS provider and publishes
SpeechChunkReady events so the avatar animates in sync. Never blocks avatar or
conversation. Degrades to silent text output when TTS is unavailable.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass

from companion.core.clock import Clock, SystemClock
from companion.core.contracts import SpeechRequest, TextToSpeech
from companion.core.events import (
    EVENT_SPEECH_CHUNK_READY,
    EVENT_SPEECH_PLAYBACK_STARTED,
    EventBus,
)
from companion.domain.conversation import ResponsePlan

log = logging.getLogger(__name__)


def amplitude_envelope(samples: bytes, sample_rate: int, per_second: int = 24) -> list[float]:
    """Normalized RMS energy envelope (0..1) for int16 mono audio.

    Drives the avatar's mouth from real speech prosody instead of a sine wave.
    """
    n = len(samples) // 2
    if n == 0 or sample_rate <= 0:
        return []
    levels = max(1, int(n / sample_rate * per_second))
    step = max(1, n // levels)
    out = []
    for i in range(levels):
        raw = samples[i * step * 2:(i + 1) * step * 2]
        data = struct.unpack(f"<{len(raw) // 2}h", raw) if len(raw) >= 2 else ()
        if not data:
            out.append(0.0)
            continue
        rms = (sum((v / 32768.0) ** 2 for v in data) / len(data)) ** 0.5
        out.append(min(1.0, rms))
    return out


@dataclass
class SpeechResult:
    synthesized: bool
    samples: bytes | None = None
    sample_rate: int = 0
    duration_ms: float = 0.0


class SpeechOutputService:
    def __init__(self, tts: TextToSpeech | None, bus: EventBus | None = None,
                 clock: Clock | None = None, playback=None) -> None:
        self._tts = tts
        self.bus = bus or EventBus(clock=clock)
        self._clock = clock or SystemClock()
        self._playback = playback
        self._play_task = None

    async def speak(self, text: str, plan: ResponsePlan | None = None) -> SpeechResult:
        if self._tts is None or not text.strip():
            return SpeechResult(synthesized=False)
        speed = plan.speech_speed if plan else 1.0
        affect = plan.affect if plan else {}
        req = SpeechRequest(text=text, speed=speed, affect=affect)
        try:
            out = await self._tts.synthesize(req)
        except Exception as exc:
            log.warning("TTS failed, falling back to text-only: %s", exc)
            return SpeechResult(synthesized=False)
        self.bus.publish(EVENT_SPEECH_CHUNK_READY, {
            "duration_ms": out.duration_ms,
            "text": text[:80],
            "amplitude": amplitude_envelope(out.samples, out.sample_rate),
        })
        if self._playback is not None:
            import asyncio

            self.bus.publish(EVENT_SPEECH_PLAYBACK_STARTED, {
                "duration_ms": out.duration_ms,
            })
            await asyncio.to_thread(self._playback.play, out.samples, out.sample_rate)
        return SpeechResult(
            synthesized=True,
            samples=out.samples,
            sample_rate=out.sample_rate,
            duration_ms=out.duration_ms,
        )

    def stop(self) -> None:
        if self._playback is not None:
            self._playback.stop()
