"""Base classes for model adapters.

Every adapter exposes lifecycle (load/unload/warm), a ModelCapability and a
memory estimate so the runtime can make memory-safe decisions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from companion.core.contracts import ModelCapability
from companion.core.errors import ProviderError, ProviderNotAvailableError

log = logging.getLogger(__name__)


class BaseAdapter(ABC):
    provider_name: str = "base"
    slot: str = ""

    def __init__(self, config: dict, model_id: str = "") -> None:
        self.config = config or {}
        self._params = self.config
        self.model_id = model_id or self.config.get("model_id", "") or self.provider_name
        self._loaded = False
        self._last_used = 0.0
        self._load_time_ms = 0.0
        self._load_error: str | None = None

    # -- capability -----------------------------------------------------

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(name=self.model_id, estimated_ram_mb=self.estimate_ram_mb())

    # -- lifecycle ------------------------------------------------------

    def load(self) -> None:
        """Load the model into memory. May raise ProviderNotAvailableError."""
        if self._loaded:
            return
        try:
            self._do_load()
            self._loaded = True
            self._load_error = None
        except (ProviderError, ProviderNotAvailableError):
            raise
        except Exception as exc:  # wrap foreign errors
            self._load_error = str(exc)
            log.exception("model %s failed to load", self.model_id)
            raise ProviderNotAvailableError(f"{self.provider_name}/{self.model_id}: {exc}") from exc

    @abstractmethod
    def _do_load(self) -> None: ...

    def warm(self) -> None:
        """Best-effort optional warm-up; default no-op."""
        self.load()

    def unload(self) -> None:
        self._loaded = False
        self._load_error = None

    def is_loaded(self) -> bool:
        return self._loaded

    def mark_used(self) -> None:
        from companion.core.clock import SystemClock

        self._last_used = SystemClock().monotonic()

    def last_used(self) -> float:
        return self._last_used

    def load_time_ms(self) -> float:
        return self._load_time_ms

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def estimate_ram_mb(self) -> int:
        return 0

    @staticmethod
    def require(module_name: str, hint: str) -> None:
        """Raise ProviderNotAvailableError if an optional dependency is missing.

        Called from __init__ so the factory can substitute a fallback provider
        at build time instead of failing later during a live conversation.
        """
        try:
            __import__(module_name)
        except ImportError as exc:
            raise ProviderNotAvailableError(hint) from exc

    def health(self) -> dict:
        return {
            "slot": self.slot,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "loaded": self._loaded,
            "load_error": self._load_error,
            "ram_mb": self.estimate_ram_mb(),
        }
