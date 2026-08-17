"""Provider contracts (Protocols).

These are the *only* dependency edges between application logic and models.
An implementation of any protocol lives under infrastructure/models/ and is
selected purely through configuration. Domain and application layers never
import concrete providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

from companion.core.types import Vector


@dataclass
class GenerationRequest:
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    json_schema: dict | None = None  # constrains structured output when supported
    context: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


@runtime_checkable
class LanguageModel(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...
    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]: ...


@dataclass
class AudioInput:
    # Raw mono PCM 16-bit at audio.sample_rate
    samples: bytes
    sample_rate: int = 16000
    source: str = "mic"


@dataclass
class Transcript:
    text: str
    language: str = "auto"
    segments: list[dict] = field(default_factory=list)
    confidence: float = 1.0


@runtime_checkable
class SpeechToText(Protocol):
    async def transcribe(self, audio: AudioInput) -> Transcript: ...


@dataclass
class SpeechRequest:
    text: str
    voice: str = ""
    speed: float = 1.0
    # Optional prosody hints coming from the ResponsePlan.
    affect: dict = field(default_factory=dict)


@dataclass
class AudioOutput:
    # Raw mono PCM 16-bit at the returned sample rate.
    samples: bytes
    sample_rate: int = 24000
    duration_ms: float = 0.0


@runtime_checkable
class TextToSpeech(Protocol):
    async def synthesize(self, request: SpeechRequest) -> AudioOutput: ...


@dataclass
class VideoFrame:
    rgb: bytes  # raw RGB bytes, row-major
    width: int
    height: int
    timestamp: float = 0.0


@dataclass
class BlendshapeSet:
    values: dict[str, float] = field(default_factory=dict)


@dataclass
class HeadPose:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass
class Gaze:
    estimated_attention: float = 0.5


@dataclass
class FaceObservation:
    """Measurable, non-interpretive face observations."""

    face_id: str = "primary"
    timestamp: str = ""
    detected: bool = True
    blendshapes: BlendshapeSet = field(default_factory=BlendshapeSet)
    head_pose: HeadPose = field(default_factory=HeadPose)
    gaze: Gaze = field(default_factory=Gaze)
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "face_id": self.face_id,
            "timestamp": self.timestamp,
            "detected": self.detected,
            "blendshapes": self.blendshapes.values,
            "head_pose": {
                "pitch": self.head_pose.pitch,
                "yaw": self.head_pose.yaw,
                "roll": self.head_pose.roll,
            },
            "gaze": {"estimated_attention": self.gaze.estimated_attention},
            "confidence": self.confidence,
        }


@runtime_checkable
class FacePerception(Protocol):
    async def analyze(self, frame: VideoFrame) -> FaceObservation: ...


@runtime_checkable
class VoiceActivityDetector(Protocol):
    async def is_speech(self, chunk: AudioInput) -> bool: ...
    async def reset(self) -> None: ...


@dataclass
class EmbeddingRequest:
    texts: list[str]


@runtime_checkable
class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int
    async def embed(self, texts: list[str]) -> list[Vector]: ...


# --- Model capability metadata ---------------------------------------------

@dataclass(frozen=True)
class ModelCapability:
    name: str
    version: str = "unknown"
    parameter_count: str = "unknown"
    quantization: str = "unknown"
    context_length: int = 0
    languages: tuple[str, ...] = ()
    supports_streaming: bool = False
    supports_json: bool = False
    supports_tools: bool = False
    estimated_ram_mb: int = 0
    supports_gpu: bool = False
