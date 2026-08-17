"""Interactive companion session: text / voice / multimodal loops.

Dev-only. The session recorder logs every turn to a JSONL file with a session
id; it is OFF by default (privacy), enabled explicitly with --record.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime

from companion.core.ids import new_id
from companion.infrastructure.audio.camera import CameraCaptureSource, NullCameraSource
from companion.infrastructure.audio.capture import MicrophoneCaptureSource, NullCaptureSource
from companion.infrastructure.audio.controller import MicrophoneController

log = logging.getLogger(__name__)

EXIT_WORDS = ("exit", "quit", "bye", "au revoir")


class SessionRecorder:
    """Append-only JSONL recorder of turns. Dev-only; disabled by default."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.session_id = new_id("session")
        self._fh = None

    def open(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._write({"event": "session_start", "session_id": self.session_id,
                     "ts": datetime.now().isoformat()})

    def record_turn(self, turn: dict) -> None:
        if self._fh is None:
            return
        self._write({"event": "turn", "session_id": self.session_id,
                     "ts": datetime.now().isoformat(), "turn": turn})

    def _write(self, record: dict) -> None:
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            log.warning("session recorder write failed", exc_info=True)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._write({"event": "session_end", "session_id": self.session_id,
                             "ts": datetime.now().isoformat()})
            finally:
                self._fh.close()
                self._fh = None


class InteractSession:
    """Run an interactive loop on top of a built CompanionApp.

    Modes:
      text  -- REPL: prompt() -> app.respond -> write()
      voice -- mic -> VAD -> STT final -> respond -> speak (TTS + playback)
      camera-- webcam -> face pipeline (displayed alongside responses)

    All capture sources degrade to null implementations when unavailable.
    """

    def __init__(self, app, *, voice: bool = False, camera: bool = False,
                 record: str | None = None,
                 readline=input, write=print, sample_rate: int = 16000,
                 mic_source=None, camera_source=None) -> None:
        self.app = app
        self.voice = voice
        self.camera = camera
        self.sample_rate = sample_rate
        self._readline = readline
        self._write = write
        self._mic_source = mic_source
        self._camera_source = camera_source
        self.recorder = SessionRecorder(record)
        self._mic: MicrophoneController | None = None
        self._camera_queue = None
        self._camera_cap = None
        self._exit = asyncio.Event()
        self._speaking = False

    def _load_models(self) -> None:
        comp = self.app.components
        slots = ["llm.default", "embeddings.default"]
        if self.voice:
            # The microphone pipeline calls VAD and STT directly; unlike
            # conversational LLM loading, it has no lazy-loading boundary.
            # Load every real-time voice dependency before capture starts.
            slots.extend(("vad.default", "stt.default", "tts.default"))
        if self.camera:
            slots.append("vision.face")
        for slot in slots:
            try:
                comp.lifecycle.load(slot)
            except Exception as exc:
                log.warning("could not load %s: %s", slot, exc)

    async def run(self) -> None:
        self.recorder.open()
        self._load_models()
        comp = self.app.components
        try:
            if self.voice:
                await self._run_voice(comp)
            else:
                await self._run_text(comp)
        finally:
            await self._teardown()
            self.recorder.close()

    async def _run_text(self, comp) -> None:
        self._write("Companion is ready. Type your message (exit/quit to stop).")
        while not self._exit.is_set():
            line = await asyncio.to_thread(self._readline, "you> ")
            text = (line or "").strip()
            if not text:
                continue
            if text.lower() in EXIT_WORDS:
                break
            reply = await self._text_turn(text, comp)
            self._write(f"companion> {reply}")

    async def _text_turn(self, text: str, comp) -> str:
        started = time.monotonic()
        result = await self.app.respond(text, source="text", speak=False)
        latency_ms = (time.monotonic() - started) * 1000.0
        self.recorder.record_turn({"source": "text", "input": text,
                                   "reply": result["text"],
                                   "latency_ms": round(latency_ms, 1),
                                   "intent": result.get("plan", {}).get("intent")})
        return result["text"]

    async def _run_voice(self, comp) -> None:
        self._write("Voice mode: speak to your companion (Ctrl+C to stop).")
        mic_source = self._mic_source
        if mic_source is None:
            mic_source = MicrophoneCaptureSource() if self.voice else NullCaptureSource()
        try:
            self._mic = MicrophoneController(comp.perception, capture=mic_source,
                                             bus=self.app.bus, sample_rate=self.sample_rate)
        except Exception as exc:
            log.warning("mic unavailable, voice input disabled: %s", exc)
            self._mic = MicrophoneController(comp.perception, capture=NullCaptureSource(),
                                             bus=self.app.bus, sample_rate=self.sample_rate)
        self._mic.on_transcript_final(self._on_transcript_final)
        if self.app.config.barge_in_enabled and self.voice:
            self._mic.on_audio_started(self._on_barge_in)
        if self.camera:
            self._start_camera(comp)
        self._mic.start()
        if self.camera:
            self._avatar_task = asyncio.create_task(self._avatar_loop())
            self._display_task = asyncio.create_task(self._avatar_display())
        try:
            await self._exit.wait()
        except asyncio.CancelledError:
            pass

    def _start_camera(self, comp) -> None:
        import asyncio

        self._camera_queue = asyncio.Queue(maxsize=2)
        try:
            cap = self._camera_source or CameraCaptureSource(fps=10)
            cap.start(self._camera_queue)
            self._camera_cap = cap
        except Exception as exc:
            log.warning("camera unavailable: %s", exc)
            self._camera_cap = NullCameraSource()
        comp.perception.attach_camera(self._camera_queue, provider=comp.perception.face)
        self._camera_task = asyncio.create_task(self._camera_tick(comp))
        self._start_avatar(comp)

    async def _camera_tick(self, comp) -> None:
        while not self._exit.is_set():
            try:
                await comp.perception.face_tick()
            except Exception as exc:
                log.warning("face tick failed: %s", exc)
            await asyncio.sleep(0.05)

    def _start_avatar(self, comp) -> None:
        from companion.application.avatar import (
            AvatarService,
            ConsoleAvatarDriver,
            ExpressionController,
        )

        try:
            controller = ExpressionController(driver=ConsoleAvatarDriver(), fps=10)
            self._avatar = AvatarService(controller, bus=self.app.bus, fps=10)
            self._avatar.attach_bus(self.app.bus)
        except Exception as exc:
            log.warning("avatar unavailable: %s", exc)
            self._avatar = None

    async def _avatar_loop(self) -> None:
        if self._avatar is not None:
            await self._avatar.run()

    async def _avatar_display(self) -> None:
        """Redraw the ASCII face + user state a few times per second."""
        interval = 0.5
        while not self._exit.is_set():
            await asyncio.sleep(interval)
            if self._avatar is None:
                continue
            try:
                self._clear_lines()
                self._write(self._avatar._controller.driver.last_frame)
            except Exception as exc:
                log.warning("avatar display failed: %s", exc)

    def _clear_lines(self) -> None:
        if getattr(self, "_write", None) is not None and self._write is not print:
            return
        if self._write is print:
            print("\x1b[%dA\x1b[J" % 5, end="")

    async def _on_transcript_final(self, text: str, language: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if text.lower() in EXIT_WORDS:
            self._exit.set()
            return
        self._write(f"you> {text}")
        started = time.monotonic()
        self._speaking = True
        try:
            result = await self.app.respond(text, source="voice", speak=True)
        finally:
            self._speaking = False
        latency_ms = (time.monotonic() - started) * 1000.0
        self.recorder.record_turn({"source": "voice", "input": text,
                                   "reply": result["text"],
                                   "latency_ms": round(latency_ms, 1),
                                   "language": language,
                                   "intent": result.get("plan", {}).get("intent")})
        self._write(f"companion> {result['text']}")

    def _on_barge_in(self) -> None:
        """VAD heard the user while the assistant was talking: cut the TTS."""
        if not getattr(self, "_speaking", False):
            return
        comp = self.app.components
        if comp is not None and comp.speech is not None:
            log.info("barge-in: stopping TTS playback on new speech")
            comp.speech.stop()

    async def _teardown(self) -> None:
        if self._mic is not None:
            await self._mic.aclose()
        if getattr(self, "_camera_cap", None) is not None:
            self._camera_cap.stop()
        tasks = []
        for name in ("_display_task", "_camera_task", "_avatar_task"):
            task = getattr(self, name, None)
            if task is not None:
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if getattr(self, "_avatar", None) is not None:
            self._avatar.stop()
        if hasattr(self.app, "aclose"):
            await self.app.aclose()
        else:  # test and third-party lightweight app implementations
            self.app.shutdown()
