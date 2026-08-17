"""Text-to-speech providers: Kokoro and Piper behind one interface."""

from __future__ import annotations

import logging

from companion.core.contracts import AudioOutput, ModelCapability, SpeechRequest
from companion.core.errors import ProviderNotAvailableError, ProviderTimeoutError
from companion.infrastructure.models.base import BaseAdapter

log = logging.getLogger(__name__)


class KokoroTTSProvider(BaseAdapter):
    """Kokoro 82M via kokoro-onnx. CPU-only, small and fast.

    English voice quality is good; French support is thinner. The interface
    makes swapping to Piper (or anything else) a configuration change.
    """

    provider_name = "kokoro"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.require(
            "kokoro_onnx",
            "kokoro-onnx is not installed. Run: pip install 'myai[tts]'",
        )
        self._kokoro = None
        self._voices = None
        self._voice = config.get("voice", "ff_siwis")

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            name=self.model_id,
            languages=("en",),
            supports_streaming=True,
            estimated_ram_mb=self.estimate_ram_mb(),
            supports_gpu=False,
        )

    def estimate_ram_mb(self) -> int:
        return int(self._params.get("estimated_ram_mb", 300))

    def _do_load(self) -> None:
        try:
            import kokoro_onnx
            import soundfile  # noqa: F401  (ensures kokoro deps present)
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "kokoro-onnx is not installed. Run: pip install 'myai[tts]'"
            ) from exc

        model_path = self._params.get("path") or self._params.get("model_path")
        voices_path = self._params.get("voices_path") or self._params.get("voices")
        if not model_path or not voices_path:
            raise ProviderNotAvailableError(
                "kokoro provider requires 'path' (model) and 'voices_path' (voices bin)"
            )
        try:
            self._kokoro = kokoro_onnx.Kokoro(model_path, voices_path)
        except Exception as exc:
            raise ProviderNotAvailableError(f"kokoro failed to load: {exc}") from exc

    async def synthesize(self, request: SpeechRequest) -> AudioOutput:
        self.mark_used()
        if self._kokoro is None:
            raise ProviderNotAvailableError(f"model {self.model_id} not loaded")
        import asyncio

        voice = request.voice or self._voice
        speed = request.speed
        try:
            samples, rate = await asyncio.to_thread(
                self._kokoro.create,
                request.text,
                voice=voice,
                speed=speed,
                lang=self._params.get("lang", "en-us"),
            )
        except Exception as exc:
            raise ProviderTimeoutError(f"kokoro synthesis failed: {exc}") from exc
        import numpy as np

        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        duration = len(pcm) / float(rate) * 1000.0
        return AudioOutput(
            samples=pcm.tobytes(), sample_rate=int(rate), duration_ms=duration
        )


class PiperTTSProvider(BaseAdapter):
    provider_name = "piper"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self._engine = None

    def _do_load(self) -> None:
        try:
            import piper
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "piper-tts is not installed. Run: pip install 'myai[tts]'"
            ) from exc
        model_path = self._params.get("path") or self._params.get("model_path")
        if not model_path:
            raise ProviderNotAvailableError("piper provider requires 'path' (onnx model)")
        try:
            self._engine = piper.PiperVoice.load(model_path)
        except Exception as exc:
            raise ProviderNotAvailableError(f"piper failed to load: {exc}") from exc

    async def synthesize(self, request: SpeechRequest) -> AudioOutput:
        self.mark_used()
        if self._engine is None:
            raise ProviderNotAvailableError(f"model {self.model_id} not loaded")
        import asyncio

        import numpy as np

        rate = self._engine.config.sample_rate
        try:
            audio = await asyncio.to_thread(self._synthesize, request.text)
        except Exception as exc:
            raise ProviderTimeoutError(f"piper synthesis failed: {exc}") from exc
        pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
        return AudioOutput(
            samples=pcm.tobytes(),
            sample_rate=int(rate),
            duration_ms=len(pcm) / float(rate) * 1000.0,
        )

    def _synthesize(self, text: str) -> list[float]:
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            self._engine.synthesize_wav(text, wav)
        buf.seek(0)
        with wave.open(buf, "rb") as wav:
            n = wav.getnframes()
            frames = wav.readframes(n)
        import numpy as np

        return (np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0).tolist()


class MockTTSProvider(BaseAdapter):
    """Deterministic TTS that returns a sine-wave PCM buffer."""

    provider_name = "mock"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.sample_rate = int(config.get("sample_rate", 24000))

    def _do_load(self) -> None:
        return None

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(name=self.model_id, supports_streaming=True, estimated_ram_mb=0)

    async def synthesize(self, request: SpeechRequest) -> AudioOutput:
        self.mark_used()
        import numpy as np

        seconds = min(20.0, max(0.2, len(request.text) / 15.0))
        n = int(self.sample_rate * seconds)
        t = np.arange(n) / self.sample_rate
        samples = (np.sin(2 * np.pi * 220.0 * t) * 0.2 * 32767).astype(np.int16)
        return AudioOutput(
            samples=samples.tobytes(),
            sample_rate=self.sample_rate,
            duration_ms=seconds * 1000.0,
        )
