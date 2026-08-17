import asyncio

import pytest

from companion.application.perception import PerceptionService
from companion.application.speech_output import SpeechOutputService
from companion.core.contracts import AudioInput, SpeechRequest
from companion.core.events import (
    EVENT_AUDIO_ENDED,
    EVENT_AUDIO_STARTED,
    EVENT_TRANSCRIPT_FINAL,
    EVENT_TRANSCRIPT_PARTIAL,
    EventBus,
)
from companion.infrastructure.audio import resample_pcm
from companion.infrastructure.audio.capture import NullCaptureSource
from companion.infrastructure.audio.controller import MicrophoneController
from companion.infrastructure.audio.playback import NullPlaybackSink
from companion.infrastructure.models.stt.whisper import MockSTTProvider


class _FakeVAD:
    async def is_speech(self, chunk: AudioInput) -> bool:
        return any(b != 0 for b in chunk.samples)

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def is_loaded(self) -> bool:
        return True

    def mark_used(self) -> None:
        pass


def _chunk(ms: int = 20) -> AudioInput:
    n = 16000 * ms // 1000
    return AudioInput(samples=b"\x00\x00" * n, sample_rate=16000)


@pytest.mark.asyncio
async def test_audio_pipeline_vad_to_transcript():
    bus = EventBus()
    events = []

    async def on_started(_e):
        events.append("started")

    async def on_ended(_e):
        events.append("ended")

    async def on_partial(e):
        events.append(("partial", e.payload["text"]))

    async def on_final(e):
        events.append(("final", e.payload["text"]))

    bus.subscribe(EVENT_AUDIO_STARTED, on_started)
    bus.subscribe(EVENT_AUDIO_ENDED, on_ended)
    bus.subscribe(EVENT_TRANSCRIPT_PARTIAL, on_partial)
    bus.subscribe(EVENT_TRANSCRIPT_FINAL, on_final)

    stt = MockSTTProvider(config={"model_id": "mock-stt"})
    per = PerceptionService(vad_provider=_FakeVAD(), stt_provider=stt, bus=bus, sample_rate=16000,
                            transcript_partial_threshold=5)
    speech = AudioInput(samples=b"\x01\x00" * 320, sample_rate=16000)
    for _ in range(10):
        await per.process_audio_chunk(speech)
    await per.process_audio_chunk(AudioInput(samples=b"\x00\x00" * 320, sample_rate=16000))
    await per.process_audio_chunk(AudioInput(samples=b"\x00\x00" * 320, sample_rate=16000))
    await asyncio.sleep(0.1)

    assert "started" in events
    assert "ended" in events
    finals = [t for item in events if isinstance(item, tuple) and item[0] == "final"
              for t in (item[1],)]
    assert finals and finals[0]  # non-empty final transcript


@pytest.mark.asyncio
async def test_microphone_controller_feeds_perception():
    bus = EventBus()
    finals = []

    async def on_final(e):
        finals.append(e.payload["text"])

    bus.subscribe(EVENT_TRANSCRIPT_FINAL, on_final)
    stt = MockSTTProvider(config={"model_id": "mock-stt"})
    per = PerceptionService(vad_provider=_FakeVAD(), stt_provider=stt, bus=bus, sample_rate=16000)

    class _FakeCapture:
        def start(self, queue, sample_rate):
            for _ in range(6):
                queue.put_nowait(AudioInput(samples=b"\x01\x00" * 320, sample_rate=16000))
            queue.put_nowait(AudioInput(samples=b"\x00\x00" * 320, sample_rate=16000))

        def stop(self):
            pass

    ctrl = MicrophoneController(per, capture=_FakeCapture(), bus=bus)
    ctrl.start()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if finals:
            break
    ctrl.stop()
    assert finals and finals[0]


@pytest.mark.asyncio
async def test_speech_output_plays_through_sink():
    class _FakeTTS:
        async def synthesize(self, request: SpeechRequest):
            assert request.text == "hello"
            return type("AudioOutput", (), {"samples": b"\x00\x00" * 8000,
                                            "sample_rate": 16000, "duration_ms": 500})()

    sink = NullPlaybackSink()
    svc = SpeechOutputService(tts=_FakeTTS(), bus=EventBus(), playback=sink)
    res = await svc.speak("hello")
    assert res.synthesized
    assert res.duration_ms == 500


def test_resample_pcm():
    out = resample_pcm(b"\x00\x00\x80\x01", 16000, 16000)
    assert out == b"\x00\x00\x80\x01"
    out = resample_pcm(b"\x00\x00" * 1600, 16000, 8000)
    assert len(out) == 1600  # half the samples
    assert out == b"\x00\x00" * 800


def test_null_capture_and_sink():
    capture = NullCaptureSource()
    capture.start(None)
    capture.stop()
    sink = NullPlaybackSink()
    sink.play(b"\x00\x00" * 1600, 16000)
    assert sink.last_duration_ms == 100.0
