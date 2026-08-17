"""Offline benchmark harness.

Measures what matters on a modest laptop: model load times, first-token
latency, tokens/sec, overall RAM and per-slot estimates. Results are honest —
mock/simulated providers are labelled as such, never reported as real numbers.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from companion.core.contracts import GenerationRequest
from companion.core.events import (
    EVENT_AUDIO_ENDED,
    EVENT_AUDIO_STARTED,
    EVENT_AVATAR_FIRST_MOTION,
    EVENT_RESPONSE_COMPLETE,
    EVENT_RESPONSE_PLAN_CREATED,
    EVENT_RESPONSE_TOKEN_GENERATED,
    EVENT_RETRIEVAL_COMPLETE,
    EVENT_SPEECH_CHUNK_READY,
    EVENT_SPEECH_PLAYBACK_STARTED,
    EVENT_TRANSCRIPT_FINAL,
    EventBus,
)
from companion.runtime.config import Config
from companion.runtime.orchestration import CompanionApp


@dataclass
class BenchmarkResult:
    slot: str
    provider: str
    model_id: str
    loaded: bool
    simulated: bool
    load_ms: float = 0.0
    first_token_ms: float = 0.0
    tokens_per_s: float = 0.0
    total_ms: float = 0.0
    est_ram_mb: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


PROMPT = "Explain the difference between a fact and a belief, using one example."
SIMULATED_PROVIDERS = {"mock", "null"}


def _is_simulated(provider: str) -> bool:
    return provider in SIMULATED_PROVIDERS


async def benchmark_llm_slot(app: CompanionApp, slot: str, max_tokens: int = 64) -> BenchmarkResult:
    comp = app.components
    registry = comp.registry
    provider = registry.get_optional(slot)
    if provider is None:
        return BenchmarkResult(slot=slot, provider="?", model_id="?",
                               loaded=False, simulated=True, notes="slot not configured")

    sim = _is_simulated(provider.provider_name)
    est = provider.estimate_ram_mb()

    t0 = time.perf_counter()
    try:
        if not provider.is_loaded():
            provider.load()
        load_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        return BenchmarkResult(slot=slot, provider=provider.provider_name,
                               model_id=provider.model_id, loaded=False,
                               simulated=sim, est_ram_mb=est, notes=f"load failed: {exc}")

    req = GenerationRequest(prompt=PROMPT, max_tokens=max_tokens)
    try:
        t0 = time.perf_counter()
        async for _chunk in provider.stream(req):
            break
        first_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        result = await provider.generate(req)
        total_ms = (time.perf_counter() - t0) * 1000.0
        tokens = max(1, len(result.text.split()))
        tps = tokens / (total_ms / 1000.0)
        if provider.is_loaded():
            provider.unload()
        return BenchmarkResult(
            slot=slot, provider=provider.provider_name, model_id=provider.model_id,
            loaded=True, simulated=sim, load_ms=load_ms, first_token_ms=first_ms,
            tokens_per_s=tps, total_ms=total_ms, est_ram_mb=est,
            notes="SIMULATED (not a real model)" if sim else "",
        )
    except Exception as exc:
        if provider.is_loaded():
            provider.unload()
        return BenchmarkResult(slot=slot, provider=provider.provider_name,
                               model_id=provider.model_id, loaded=True, simulated=sim,
                               est_ram_mb=est, notes=f"generation failed: {exc}")


async def benchmark_app(config: Config | None = None, max_tokens: int = 64) -> list[BenchmarkResult]:
    cfg = config or Config.load()
    app = CompanionApp(cfg)
    app.build()
    results: list[BenchmarkResult] = []
    for slot in sorted(app.components.registry.slots()):
        if slot.startswith("llm."):
            r = await benchmark_llm_slot(app, slot, max_tokens=max_tokens)
            results.append(r)
        else:
            provider = app.components.registry.get_optional(slot)
            est = provider.estimate_ram_mb() if provider else 0
            sim = _is_simulated(provider.provider_name) if provider else True
            results.append(BenchmarkResult(
                slot=slot, provider=provider.provider_name if provider else "?",
                model_id=provider.model_id if provider else "?",
                loaded=False, simulated=sim, est_ram_mb=est,
                notes="load-and-run measured under `companion benchmark --full`",
            ))
    await app.aclose()
    return results


async def resource_benchmark(iterations: int = 3, max_tokens: int = 32) -> dict:
    """End-to-end conversational round-trips; measures per-turn latency."""
    cfg = Config.load()
    app = CompanionApp(cfg)
    app.build()
    latencies: list[float] = []
    try:
        for i in range(iterations):
            t0 = time.perf_counter()
            await app.respond(f"turn number {i}: tell me a one line fact about memory.")
            latencies.append((time.perf_counter() - t0) * 1000.0)
    finally:
        await app.aclose()
    return {
        "iterations": iterations,
        "per_turn_ms": [round(x, 1) for x in latencies],
        "mean_turn_ms": round(sum(latencies) / len(latencies), 1),
        "max_turn_ms": round(max(latencies), 1),
    }


# ---------------------------------------------------------------------------
# Pipeline latency tracer
# ---------------------------------------------------------------------------

TRACED_EVENTS = (
    EVENT_AUDIO_STARTED,
    EVENT_AUDIO_ENDED,
    EVENT_TRANSCRIPT_FINAL,
    EVENT_RETRIEVAL_COMPLETE,
    EVENT_RESPONSE_PLAN_CREATED,
    EVENT_RESPONSE_TOKEN_GENERATED,
    EVENT_RESPONSE_COMPLETE,
    EVENT_SPEECH_CHUNK_READY,
    EVENT_SPEECH_PLAYBACK_STARTED,
    EVENT_AVATAR_FIRST_MOTION,
)


class PipelineTracer:
    """Records wall-clock stage timestamps from bus events for one turn.

    Attach before the turn, call reset() at turn start, and read breakdown()
    after EVENT_RESPONSE_COMPLETE. Stages that never fire (e.g. mic audio in a
    text-only turn) report None.
    """

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._subs = []
        self._times: dict[str, list[float]] = {}
        self.start_mono: float = 0.0

    def attach(self) -> None:
        if self._subs:
            return
        for kind in TRACED_EVENTS:
            self._subs.append(self.bus.subscribe(kind, self._record, policy="drop_oldest"))

    def detach(self) -> None:
        for sub in self._subs:
            self.bus.unsubscribe(sub)
        self._subs = []

    async def _record(self, event) -> None:
        self._times.setdefault(event.kind, []).append(time.perf_counter())

    def reset(self) -> None:
        self._times = {}
        self.start_mono = time.perf_counter()

    def _first(self, kind: str) -> float | None:
        ts = self._times.get(kind)
        return ts[0] if ts else None

    def _ms(self, kind: str, *, relative_to: str | None = None) -> float | None:
        ts = self._first(kind)
        if ts is None:
            return None
        base = self._first(relative_to) if relative_to else self.start_mono
        return (ts - base) * 1000.0

    def breakdown(self) -> dict:
        """Milestone stage latencies (ms). All honest wall-clock measurements."""
        speech_start = self._first(EVENT_AUDIO_STARTED)
        speech_end = self._first(EVENT_AUDIO_ENDED)
        stt_done = self._first(EVENT_TRANSCRIPT_FINAL)
        retrieval = self._first(EVENT_RETRIEVAL_COMPLETE)
        plan = self._first(EVENT_RESPONSE_PLAN_CREATED)
        first_token = self._first(EVENT_RESPONSE_TOKEN_GENERATED)
        done = self._first(EVENT_RESPONSE_COMPLETE)
        tts = self._first(EVENT_SPEECH_CHUNK_READY)
        playback = self._first(EVENT_SPEECH_PLAYBACK_STARTED)
        avatar = self._first(EVENT_AVATAR_FIRST_MOTION)

        def _d(a, b, base=None):
            if a is None or b is None:
                return None
            ref = base or a
            return (b - ref) * 1000.0

        return {
            "speech_input_ms": _d(speech_start, speech_end),
            "stt_ms": _d(speech_end, stt_done),
            "retrieval_ms": _d(speech_end, retrieval) if speech_end else _d(self.start_mono, retrieval),
            "plan_ms": _d(retrieval, plan),
            "llm_first_token_ms": _d(plan, first_token),
            "llm_generation_ms": _d(first_token, done),
            "llm_total_ms": _d(plan, done),
            "tts_synth_ms": _d(done, tts),
            "playback_start_ms": _d(tts, playback),
            "avatar_first_motion_ms": _d(plan, avatar),
            "total_ms": _d(self.start_mono, done or tts),
        }


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


async def e2e_benchmark(app: CompanionApp, iterations: int = 2, mute: bool = False,
                        prompt: str = "Tell me one line about memory.") -> dict:
    """Full text->LLM->TTS pipeline breakdown using the real models.

    mute=True swaps playback for a no-op sink so the benchmark runs silent.
    """
    if mute:
        from companion.infrastructure.audio.playback import NullPlaybackSink
        app.components.speech._playback = NullPlaybackSink()
    avatar_task = asyncio.create_task(app.components.avatar.run(), name="bench-avatar")
    tracer = PipelineTracer(app.bus)
    tracer.attach()
    turns: list[dict] = []
    try:
        for i in range(iterations):
            app.components.avatar._expect_motion = False
            tracer.reset()
            await app.respond(f"turn {i}: {prompt}", source="voice", speak=True)
            await asyncio.sleep(0.2)  # let late avatar-motion events settle in-window
            turns.append(tracer.breakdown())
    finally:
        tracer.detach()
        avatar_task.cancel()
        await asyncio.sleep(0)
    keys = list(turns[0].keys()) if turns else []
    means = {}
    for k in keys:
        vals = [t[k] for t in turns if t.get(k) is not None]
        means[k] = round(_median(vals), 1) if vals else None
    return {"iterations": iterations, "per_turn": turns, "median_ms": means,
            "honest": True}


async def speech_input_benchmark(app: CompanionApp, wav_path: str,
                                 prompt: str = "What did you hear?") -> dict:
    """Feed a real WAV file through VAD->STT->respond->TTS, measuring each stage."""
    import wave

    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        return {"error": f"expected 16-bit PCM WAV, got sample width {sampwidth}"}
    from companion.core.contracts import AudioInput
    from companion.infrastructure.audio import resample_pcm

    if channels > 1:
        raw = b"".join(raw[i * sampwidth * channels:(i + 1) * sampwidth * channels][:sampwidth]
                       for i in range(len(raw) // (sampwidth * channels)))
    pcm16 = resample_pcm(raw, rate, 16000) if rate != 16000 else raw

    comp = app.components
    if not (comp.lifecycle.is_loaded("vad.default")
            and comp.lifecycle.is_loaded("stt.default")):
        return {"error": "vad.default and stt.default must be loaded before --audio "
                         "(run `companion benchmark --audio` without other args)"}

    tracer = PipelineTracer(app.bus)
    tracer.attach()
    tracer.reset()
    transcript_ready = asyncio.Event()
    captured: dict = {}

    async def on_transcript(event) -> None:
        captured["text"] = event.payload.get("text", "")
        captured["language"] = event.payload.get("language", "")
        transcript_ready.set()

    app.bus.subscribe(EVENT_TRANSCRIPT_FINAL, on_transcript)
    perception = comp.perception
    chunk_size = 1600  # 100ms @16k
    for i in range(0, len(pcm16), chunk_size):
        chunk = AudioInput(samples=bytes(pcm16[i:i + chunk_size]),
                           sample_rate=16000, source="file")
        await perception.process_audio_chunk(chunk)
    try:
        await asyncio.wait_for(transcript_ready.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        tracer.detach()
        return {"error": "no transcript within 30s (VAD or STT did not trigger)"}

    await app.respond(captured["text"], source="voice", speak=True)
    await asyncio.sleep(0.2)  # let late avatar-motion events settle in-window
    result = tracer.breakdown()
    tracer.detach()
    result["input_wav"] = wav_path
    result["input_seconds"] = round(len(pcm16) / 2 / 16000.0, 2)
    result["transcript"] = captured["text"]
    return result
