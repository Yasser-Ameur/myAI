"""Audio capture and playback infrastructure.

Microphone capture (sounddevice) pushes 16 kHz mono PCM chunks into an asyncio
queue on a background thread. Playback (sounddevice) renders PCM to the default
output device with a stop() that supports barge-in. Everything degrades to a
null implementation when no audio device or dependency is available.
"""

from __future__ import annotations

from typing import Protocol


def resample_pcm(samples: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of 16-bit mono PCM. No-op when rates match."""
    if src_rate == dst_rate or not samples:
        return samples
    import array

    src = array.array("h", samples)
    n = len(src)
    if n < 2:
        return samples
    out = array.array("h")
    ratio = src_rate / dst_rate
    for i in range(int(n / ratio)):
        pos = i * ratio
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(int(src[lo] * (1 - frac) + src[hi] * frac))
    return out.tobytes()


class AudioCaptureSource(Protocol):
    def start(self, queue, sample_rate: int = 16000) -> None: ...
    def stop(self) -> None: ...


class AudioPlaybackSink(Protocol):
    def play(self, samples: bytes, sample_rate: int) -> None: ...
    def stop(self) -> None: ...
    def is_playing(self) -> bool: ...
    def close(self) -> None: ...
