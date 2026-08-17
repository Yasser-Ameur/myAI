"""Voice activity detection providers."""

from __future__ import annotations

import logging

from companion.core.contracts import AudioInput, ModelCapability
from companion.core.errors import ProviderNotAvailableError
from companion.infrastructure.models.base import BaseAdapter

log = logging.getLogger(__name__)


class SileroVADProvider(BaseAdapter):
    """Silero VAD. Lightweight; intended to run always-on.

    Uses onnxruntime with a downloaded Silero .onnx model. Falls back cleanly
    to EnergyVAD when onnxruntime is missing.
    """

    provider_name = "silero"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.require(
            "onnxruntime",
            "onnxruntime is not installed; use provider 'energy' for a dependency-free VAD",
        )
        self._model = None
        self._state = None
        self._h = None
        self._c = None
        self._pending = None
        self._last_speech = False
        self.sr = int(config.get("sample_rate", 16000))

    def _do_load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "onnxruntime is not installed; use provider 'energy' for a dependency-free VAD"
            ) from exc
        model_path = self._params.get("path") or self._params.get("model_path")
        if not model_path:
            raise ProviderNotAvailableError("silero provider requires 'path' to the .onnx model")
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(self._params.get("threads", 1))
            self._model = ort.InferenceSession(
                model_path, sess_options=so, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ProviderNotAvailableError(f"silero failed to load: {exc}") from exc
        import numpy as np

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._last_speech = False

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(name=self.model_id, estimated_ram_mb=self.estimate_ram_mb())

    def estimate_ram_mb(self) -> int:
        return int(self._params.get("estimated_ram_mb", 50))

    async def is_speech(self, chunk: AudioInput) -> bool:
        import asyncio

        self.mark_used()
        if self._model is None:
            raise ProviderNotAvailableError(f"model {self.model_id} not loaded")
        import numpy as np

        arr = np.frombuffer(chunk.samples, dtype=np.int16).astype(np.float32) / 32768.0
        if arr.size == 0:
            return False
        return await asyncio.to_thread(self._infer, arr)

    def _infer(self, arr) -> bool:
        import numpy as np

        # Silero consumes 512-sample frames, while microphone capture emits
        # 20 ms / 320-sample chunks.  Accumulate capture chunks instead of
        # treating every short chunk as silence.
        threshold = float(self._params.get("threshold", 0.5))
        frame = 512
        pending = self._pending
        if pending is None:
            pending = np.empty(0, dtype=np.float32)
        samples = np.concatenate((pending, arr))
        speech_frames = 0
        total = 0
        offset = 0
        while offset + frame <= samples.size:
            x = samples[offset : offset + frame][None, :].astype(np.float32)
            inputs = {
                "input": x,
                "state": self._state,
                "sr": np.array(self.sr, dtype=np.int64),
            }
            out = self._model.run(None, inputs)
            prob = float(out[0].item())
            self._state = out[1]
            total += 1
            if prob >= threshold:
                speech_frames += 1
            offset += frame
        self._pending = samples[offset:].copy()
        if total:
            self._last_speech = (speech_frames / total) >= 0.4
        return self._last_speech

    async def reset(self) -> None:
        import numpy as np

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._last_speech = False


class EnergyVADProvider(BaseAdapter):
    """Dependency-free VAD using short-term energy + zero-crossing heuristics.

    A robust fallback when Silero's onnx model is not installed. Not as good as
    Silero, but never crashes the pipeline.
    """

    provider_name = "energy"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.threshold = float(config.get("threshold", 400.0))
        self.frame_ms = int(config.get("frame_ms", 30))
        self.min_speech_frames = int(config.get("min_speech_frames", 3))

    def _do_load(self) -> None:
        return None

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(name=self.model_id, estimated_ram_mb=0)

    async def is_speech(self, chunk: AudioInput) -> bool:
        self.mark_used()
        import array

        if not chunk.samples:
            return False
        samples = array.array("h", chunk.samples)
        if not samples:
            return False
        frame_len = max(1, int(chunk.sample_rate * self.frame_ms / 1000.0))
        speech_frames = 0
        total = 0
        for i in range(0, len(samples) - frame_len + 1, frame_len):
            frame = samples[i : i + frame_len]
            rms = (sum(s * s for s in frame) / len(frame)) ** 0.5
            total += 1
            if rms > self.threshold:
                speech_frames += 1
        return speech_frames >= min(self.min_speech_frames, max(1, total))


class NullVADProvider(BaseAdapter):
    """Always-on speech detection (or always-off via config)."""

    provider_name = "null"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.always_speech = bool(config.get("always_speech", True))

    def _do_load(self) -> None:
        return None

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(name=self.model_id, estimated_ram_mb=0)

    async def is_speech(self, chunk: AudioInput) -> bool:
        self.mark_used()
        return self.always_speech and bool(chunk.samples)

    async def reset(self) -> None:
        return None
