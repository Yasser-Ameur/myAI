"""Observability metrics (from day one).

Tracks RAM, CPU, model load/unload, LLM first-token latency, tokens/sec, STT/TTS
latency, face FPS, queue depths, retrieval and memory write latency, and prompt
token counts. Exposed via /metrics and CLI.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class MetricSeries:
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=120))

    def record(self, value: float) -> None:
        self.values.append((time.monotonic(), value))

    def last(self) -> float:
        return self.values[-1][1] if self.values else 0.0

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(v for _, v in self.values) / len(self.values)

    def to_dict(self) -> dict:
        return {"last": self.last(), "mean": round(self.mean(), 3), "samples": len(self.values)}


class Metrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self.series: dict[str, MetricSeries] = {}
        self.counters: dict[str, int] = {}
        self.gauges: dict[str, float] = {}
        self._started = time.monotonic()

    def _series(self, name: str) -> MetricSeries:
        with self._lock:
            s = self.series.get(name)
            if s is None:
                s = MetricSeries(name)
                self.series[name] = s
            return s

    def record(self, name: str, value: float) -> None:
        self._series(name).record(float(value))

    def inc(self, name: str, by: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + by

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = float(value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_s": round(time.monotonic() - self._started, 1),
                "series": {k: v.to_dict() for k, v in self.series.items()},
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
            }

    # convenience recorders ------------------------------------------------

    def llm_latency_ms(self, ms: float, tokens: int) -> None:
        self.record("llm_latency_ms", ms)
        if ms > 0:
            self.record("llm_tokens_per_s", tokens / (ms / 1000.0))
        self.counters["llm_total_tokens"] = self.counters.get("llm_total_tokens", 0) + tokens

    def first_token_ms(self, ms: float) -> None:
        self.record("llm_first_token_ms", ms)

    def stt_latency_ms(self, ms: float) -> None:
        self.record("stt_latency_ms", ms)

    def tts_latency_ms(self, ms: float) -> None:
        self.record("tts_latency_ms", ms)

    def face_fps(self, fps: float) -> None:
        self.set_gauge("face_fps", fps)

    def queue_depth(self, name: str, depth: int) -> None:
        self.set_gauge(f"queue.{name}.depth", depth)

    def retrieval_latency_ms(self, ms: float) -> None:
        self.record("retrieval_latency_ms", ms)

    def memory_write_latency_ms(self, ms: float) -> None:
        self.record("memory_write_latency_ms", ms)

    def prompt_tokens(self, n: int) -> None:
        self.record("prompt_tokens", n)

    def response_tokens(self, n: int) -> None:
        self.record("response_tokens", n)
