import asyncio
import json
import types

import pytest

from companion.application.interact import InteractSession, SessionRecorder


class _FakeLifecycle:
    def __init__(self):
        self.loaded = []

    def load(self, slot):
        self.loaded.append(slot)


class _FakeApp:
    def __init__(self):
        self.components = types.SimpleNamespace(lifecycle=_FakeLifecycle())
        self.shutdown_called = False

    async def respond(self, text, source="text", speak=False):
        return {"text": f"echo: {text}", "plan": {"intent": "chat"}}

    def shutdown(self):
        self.shutdown_called = True


def test_session_recorder(tmp_path):
    path = str(tmp_path / "sessions.jsonl")
    rec = SessionRecorder(path)
    rec.open()
    rec.record_turn({"input": "hi", "reply": "hello", "latency_ms": 12.0})
    rec.close()
    lines = [json.loads(line) for line in open(path, encoding="utf-8").read().splitlines()]
    assert lines[0]["event"] == "session_start"
    assert lines[1]["event"] == "turn"
    assert lines[1]["turn"]["input"] == "hi"
    assert lines[-1]["event"] == "session_end"
    assert lines[0]["session_id"] == lines[1]["session_id"]


def test_session_recorder_disabled(tmp_path):
    rec = SessionRecorder(None)
    rec.open()
    rec.record_turn({"input": "hi"})
    rec.close()


@pytest.mark.asyncio
async def test_interact_text_mode(tmp_path):
    lines = iter(["hello there", "quit"])
    out = []
    app = _FakeApp()
    path = str(tmp_path / "session.jsonl")
    session = InteractSession(app, readline=lambda _prompt: next(lines), write=out.append,
                              record=path)
    await session.run()
    assert any(o.startswith("Companion is ready") for o in out)
    assert any(o.startswith("companion> echo: hello there") for o in out)
    assert app.shutdown_called
    assert app.components.lifecycle.loaded == ["llm.default", "embeddings.default"]
    records = [json.loads(line) for line in open(path, encoding="utf-8").read().splitlines()]
    turns = [r for r in records if r["event"] == "turn"]
    assert len(turns) == 1
    assert turns[0]["turn"]["input"] == "hello there"


@pytest.mark.asyncio
async def test_interact_voice_mode_teardown():
    from companion.core.events import EventBus
    from companion.infrastructure.audio.camera import NullCameraSource
    from companion.infrastructure.audio.capture import NullCaptureSource

    perception = types.SimpleNamespace(
        face=None,
        attach_camera=lambda *a, **k: None,
        face_tick=lambda *a, **k: asyncio.sleep(0),
    )

    class _Bus:
        def subscribe(self, *a, **k):
            return None

    class _FakeVoiceApp:
        def __init__(self):
            self.bus = EventBus()
            self.config = types.SimpleNamespace(barge_in_enabled=True)
            self.components = types.SimpleNamespace(
                lifecycle=_FakeLifecycle(), perception=perception,
                speech=types.SimpleNamespace(stop=self._on_stop),
            )
            self.shutdown_called = False
            self.stopped = 0

        def _on_stop(self):
            self.stopped += 1

        async def respond(self, text, source="voice", speak=False):
            return {"text": f"echo: {text}", "plan": {"intent": "chat"}}

        def shutdown(self):
            self.shutdown_called = True

    app = _FakeVoiceApp()
    out = []
    session = InteractSession(app, voice=True, camera=True, write=out.append,
                              mic_source=NullCaptureSource(),
                              camera_source=NullCameraSource())

    async def quit_soon():
        await asyncio.sleep(0.1)
        session._exit.set()

    await asyncio.gather(session.run(), quit_soon())
    assert app.shutdown_called
    assert not session._mic._running
    assert app.components.lifecycle.loaded == [
        "llm.default", "embeddings.default", "vad.default", "stt.default", "tts.default",
        "vision.face",
    ]
    assert session._avatar is not None  # avatar service started in camera mode
    assert session._display_task.done()


def test_barge_in_only_stops_playback_while_speaking():
    from companion.infrastructure.audio.capture import NullCaptureSource

    class _App:
        def __init__(self):
            self.config = types.SimpleNamespace(barge_in_enabled=True)
            self.components = types.SimpleNamespace(
                speech=types.SimpleNamespace(stop=lambda: setattr(self, "stopped", True)),
            )
            self.stopped = False

    app = _App()
    session = InteractSession(app, voice=True,
                              mic_source=NullCaptureSource(), write=lambda _s: None)

    session._on_barge_in()  # not speaking yet
    assert not app.stopped

    session._speaking = True
    session._on_barge_in()
    assert app.stopped

