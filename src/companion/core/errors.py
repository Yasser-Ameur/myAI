"""Typed error hierarchy for the companion.

Domain/application code raises these. Infrastructure adapters wrap foreign
exceptions into these so layer boundaries stay clean.
"""

from __future__ import annotations


class CompanionError(Exception):
    """Base error for all companion failures."""


class ConfigurationError(CompanionError):
    """Invalid or missing configuration."""


class ProviderError(CompanionError):
    """A model/provider adapter failed at runtime."""


class ProviderNotAvailableError(ProviderError):
    """A provider is not installed or its weights are missing."""


class ProviderTimeoutError(ProviderError):
    """A provider call exceeded its deadline."""


class ModelNotInstalledError(ProviderError):
    """The model is not present in the local cache; install it first."""


class ModelLoadError(ProviderError):
    """A model artifact exists but failed to load at runtime."""


class ModelOutOfMemoryError(ModelLoadError):
    """The model does not fit within the runtime's memory budget."""


class UnsupportedBackendError(ProviderError):
    """The configured execution backend is not available on this machine."""


class InferenceTimeoutError(ProviderTimeoutError):
    """An inference call exceeded its deadline and was abandoned."""


class AudioDeviceUnavailableError(CompanionError):
    """No usable microphone/capture device could be opened."""


class CameraUnavailableError(CompanionError):
    """No usable camera could be opened."""


class InvalidModelArtifactError(CompanionError):
    """A downloaded model file failed format/magic validation."""


class ChecksumMismatchError(CompanionError):
    """A downloaded file's sha256 did not match the manifest."""


class MemoryUnavailableError(CompanionError):
    """The cognitive store is not reachable; caller should degrade gracefully."""


class ModelNotFoundError(ProviderError):
    """A requested model id is unknown to the registry/manifest."""


class ModelNotLoadedError(ProviderError):
    """A model was referenced before being loaded."""


class ValidationError(CompanionError):
    """A structured result failed schema validation."""


class HardwareError(CompanionError):
    """Hardware detection/backend selection failed."""


class NetworkAccessDeniedError(CompanionError):
    """An external call was attempted while privacy.cloud_enabled is false."""
