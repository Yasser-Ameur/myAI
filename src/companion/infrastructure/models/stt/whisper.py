"""Whisper-based speech-to-text providers.

Two interchangeable backends behind one adapter:
  - faster_whisper (CTranslate2, recommended)
  - whisper.cpp via subprocess (optional)

Multilingual: language is configurable or auto-detected.
"""

from __future__ import annotations

import logging

from companion.core.clock import SystemClock
from companion.core.contracts import AudioInput, ModelCapability, Transcript
from companion.core.errors import ProviderNotAvailableError, ProviderTimeoutError
from companion.infrastructure.models.base import BaseAdapter

log = logging.getLogger(__name__)


def _segment_confidence(seg) -> float:
    """Derive a 0..1 confidence from a faster-whisper segment."""
    no_speech = getattr(seg, "no_speech_prob", None)
    if no_speech is not None:
        return max(0.0, min(1.0, 1.0 - no_speech))
    avg_logprob = getattr(seg, "avg_logprob", None)
    if avg_logprob is not None:
        return max(0.0, min(1.0, avg_logprob + 1.0))
    return 0.5


class WhisperSTTProvider(BaseAdapter):
    provider_name = "whisper"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self._backend = config.get("backend", "faster_whisper")
        if self._backend == "faster_whisper":
            self.require(
                "faster_whisper",
                "faster-whisper is not installed. Run: pip install 'myai[stt]'",
            )
        self._model = None

    @property
    def capability(self) -> ModelCapability:
        langs = tuple(
            str(lang) for lang in self._params.get("languages", ["en", "fr", "de", "es", "it", "pt"])
        )
        return ModelCapability(
            name=self.model_id,
            languages=langs,
            supports_streaming=True,
            estimated_ram_mb=self.estimate_ram_mb(),
            supports_gpu=False,
        )

    def estimate_ram_mb(self) -> int:
        size_mb = {"tiny": 400, "base": 600, "small": 1200, "medium": 2500}.get(
            self.model_id.split("-")[0], 800
        )
        return int(self._params.get("estimated_ram_mb", size_mb))

    def _do_load(self) -> None:
        if self._backend == "faster_whisper":
            self._load_faster_whisper()
        elif self._backend == "whisper_cpp":
            self._load_whisper_cpp()
        else:
            raise ProviderNotAvailableError(f"unknown whisper backend: {self._backend}")

    def _load_faster_whisper(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "faster-whisper is not installed. Run: pip install 'myai[stt]'"
            ) from exc
        device = self._params.get("device", "cpu")
        compute_type = self._params.get("compute_type", "int8" if device == "cpu" else "float16")
        model_source = self._params.get("model_path") or self.model_id
        try:
            self._model = WhisperModel(
                model_source, device=device, compute_type=compute_type
            )
        except Exception as exc:
            raise ProviderNotAvailableError(f"faster-whisper failed to load {self.model_id}: {exc}") from exc

    def _load_whisper_cpp(self) -> None:
        import shutil

        self._cpp_binary = self._params.get("binary", shutil.which("whisper-cli") or "whisper-cli")
        self._cpp_model = self._params.get("model_path", "")
        if not self._cpp_model:
            raise ProviderNotAvailableError("whisper_cpp backend requires 'model_path'")

    async def transcribe(self, audio: AudioInput) -> Transcript:
        self.mark_used()
        started = SystemClock().monotonic()
        try:
            if self._backend == "faster_whisper":
                result = await self._transcribe_faster(audio)
            else:
                result = await self._transcribe_cpp(audio)
        except Exception as exc:
            raise ProviderTimeoutError(f"whisper transcription failed: {exc}") from exc
        result.latency_ms = (SystemClock().monotonic() - started) * 1000.0
        return result

    async def _transcribe_faster(self, audio: AudioInput) -> Transcript:
        import asyncio

        import numpy as np

        if self._model is None:
            raise ProviderNotAvailableError(f"model {self.model_id} not loaded")
        language = self._params.get("language", None) or None
        if language and language.lower() in ("auto", "autodetect", "multilingual"):
            language = None  # None -> auto-detect
        arr = np.frombuffer(audio.samples, dtype=np.int16).astype(np.float32) / 32768.0

        def _run():
            seg_iter, info = self._model.transcribe(
                arr,
                language=language,
                beam_size=int(self._params.get("beam_size", 1)),
                vad_filter=False,
            )
            text_parts: list[str] = []
            segments: list[dict] = []
            confidences: list[float] = []
            for seg in seg_iter:
                text_parts.append(seg.text.strip())
                segments.append(
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                        "confidence": _segment_confidence(seg),
                    }
                )
                confidences.append(_segment_confidence(seg))
            return " ".join(t for t in text_parts if t), info.language, segments, confidences

        text, lang, segments, confidences = await asyncio.to_thread(_run)
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return Transcript(
            text=text,
            language=lang or "auto",
            segments=segments,
            confidence=min(1.0, max(0.0, confidence)),
        )

    async def _transcribe_cpp(self, audio: AudioInput) -> Transcript:
        import asyncio
        import os
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import wave

            with wave.open(f, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(audio.sample_rate)
                w.writeframes(audio.samples)
            wav_path = f.name
        cmd = [
            self._cpp_binary,
            "-m", self._cpp_model,
            "-f", wav_path,
            "-nt", str(self._params.get("threads", 4)),
            "-l", self._params.get("language", "auto"),
            "-otxt",
        ]

        def _run():
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return proc.stdout or proc.stderr

        try:
            out = await asyncio.to_thread(_run)
        finally:
            os.unlink(wav_path)
        text = out.strip()
        return Transcript(text=text, language=self._params.get("language", "auto"))


class MockSTTProvider(BaseAdapter):
    """Deterministic STT for tests/offline demos. Maps a script of audio->text."""

    provider_name = "mock"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.script: dict[str, str] = {
            str(k): str(v) for k, v in config.get("script", {}).items()
        }
        self.default_text = config.get("default_text", "hello there")

    def _do_load(self) -> None:
        return None

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(name=self.model_id, supports_streaming=True, estimated_ram_mb=0)

    async def transcribe(self, audio: AudioInput) -> Transcript:
        self.mark_used()
        if not audio.samples:
            return Transcript(text="", language="auto")
        # Match by audio hash for deterministic tests, else default text.
        key = f"{len(audio.samples)}:{audio.samples[:32].hex()}"
        text = self.script.get(key, self.script.get("default", self.default_text))
        return Transcript(text=text, language="en")
