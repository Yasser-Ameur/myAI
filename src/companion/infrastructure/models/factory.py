"""Provider factory: config string -> concrete adapter instance.

This is the ONLY place that maps provider names to classes. Business logic
never names a concrete provider.
"""

from __future__ import annotations

import logging

from companion.core.errors import ConfigurationError, ProviderNotAvailableError
from companion.infrastructure.models.base import BaseAdapter

from .embeddings.onnx import MockEmbeddingProvider, OnnxEmbeddingProvider
from .llm.llama_cpp import LlamaCppProvider
from .llm.mock import MockLLMProvider
from .llm.openai_compatible import OpenAICompatibleProvider
from .stt.whisper import MockSTTProvider, WhisperSTTProvider
from .tts.kokoro import KokoroTTSProvider, MockTTSProvider, PiperTTSProvider
from .vad.silero import EnergyVADProvider, NullVADProvider, SileroVADProvider
from .vision.mediapipe import MediaPipeFaceProvider, MockFaceProvider, NullFaceProvider

log = logging.getLogger(__name__)


# Fallback mock used when the configured provider's optional dependency is
# missing (e.g. llama-cpp-python). Keeps the whole system functional offline.
_FALLBACKS = {
    "llm": MockLLMProvider,
    "stt": MockSTTProvider,
    "tts": MockTTSProvider,
    "vad": NullVADProvider,
    "vision": MockFaceProvider,
    "embeddings": MockEmbeddingProvider,
}


def build_provider(slot: str, provider_name: str, config: dict) -> BaseAdapter:
    kind = config.get("kind", "")
    registry = {
        "llm": {
            "llama_cpp": LlamaCppProvider,
            "openai_compatible": OpenAICompatibleProvider,
            "mock": MockLLMProvider,
        },
        "stt": {
            "whisper": WhisperSTTProvider,
            "whisper_cpp": WhisperSTTProvider,
            "mock": MockSTTProvider,
        },
        "tts": {
            "kokoro": KokoroTTSProvider,
            "piper": PiperTTSProvider,
            "mock": MockTTSProvider,
        },
        "vad": {
            "silero": SileroVADProvider,
            "energy": EnergyVADProvider,
            "null": NullVADProvider,
            "mock": NullVADProvider,
        },
        "vision": {
            "mediapipe": MediaPipeFaceProvider,
            "mock": MockFaceProvider,
            "null": NullFaceProvider,
        },
        "embeddings": {
            "onnx": OnnxEmbeddingProvider,
            "mock": MockEmbeddingProvider,
        },
    }
    table = registry.get(kind, {})
    cls = table.get(provider_name)
    if cls is None:
        raise ConfigurationError(
            f"unknown provider '{provider_name}' for slot '{slot}' (kind={kind}). "
            f"Known: {', '.join(table) or 'none for this kind'}"
        )
    from companion.runtime.model_installer import resolve_provider_paths

    config = resolve_provider_paths(config)
    try:
        provider = cls(config=config)
    except ProviderNotAvailableError as exc:
        provider = build_fallback_provider(kind, slot, provider_name, config, exc)
    return provider


def build_fallback_provider(kind: str, slot: str, provider_name: str,
                            config: dict, exc: Exception) -> BaseAdapter:
    """Construct the fallback mock for a provider whose dependency is missing."""
    fallback_cls = _FALLBACKS.get(kind)
    if fallback_cls is None:
        raise
    log.warning("provider '%s' for slot '%s' unavailable (%s); using fallback %s",
                provider_name, slot, exc, fallback_cls.__name__)
    provider = fallback_cls(config={**config, "provider": provider_name, "fallback_reason": str(exc)})
    provider._fallback = True
    return provider
