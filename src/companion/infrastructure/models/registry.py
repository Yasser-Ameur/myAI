"""Model registry: maps logical slots (llm.default, stt.default, ...) to
provider instances. Providers are created from configuration only; application
code never names a concrete provider class.
"""

from __future__ import annotations

import logging
from typing import Any

from companion.core.errors import ConfigurationError, ModelNotFoundError, ProviderNotAvailableError

log = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, models_config: dict | None = None,
                 providers_config: dict | None = None) -> None:
        self._config = models_config or {}
        self._providers_config = providers_config or {}
        self._slots: dict[str, Any] = {}
        self._slot_meta: dict[str, dict] = {}

    def register_slot(self, slot: str, provider: Any, meta: dict | None = None) -> None:
        provider.slot = slot
        self._slots[slot] = provider
        self._slot_meta[slot] = meta or {}

    def get(self, slot: str) -> Any:
        provider = self._slots.get(slot)
        if provider is None:
            raise ModelNotFoundError(f"model slot not registered: {slot}")
        return provider

    def get_optional(self, slot: str) -> Any | None:
        return self._slots.get(slot)

    def has(self, slot: str) -> bool:
        return slot in self._slots

    def slots(self) -> list[str]:
        return list(self._slots.keys())

    def meta(self, slot: str) -> dict:
        return self._slot_meta.get(slot, {})

    def ensure_loaded(self, slot: str) -> Any:
        """Get and load a slot, remembering last_used.

        If the configured provider cannot load (missing optional dependency or
        model file), it is replaced with the kind's fallback mock so the system
        keeps working offline; the fallback is surfaced in health().
        """
        provider = self.get(slot)
        if not provider.is_loaded():
            try:
                provider.load()
            except ProviderNotAvailableError as exc:
                log.warning("slot %s failed to load (%s); replacing with fallback provider", slot, exc)
                self._install_fallback(slot, exc)
                provider = self.get(slot)
                provider.load()
        provider.mark_used()
        return provider

    def _install_fallback(self, slot: str, exc: Exception) -> None:
        from companion.infrastructure.models.factory import build_fallback_provider

        meta = self._slot_meta.get(slot, {})
        config = getattr(self._slots[slot], "config", {})
        fallback = build_fallback_provider(
            meta.get("kind", ""), slot, meta.get("provider", ""), dict(config), exc
        )
        fallback.slot = slot
        self._slots[slot] = fallback
        self._slot_meta[slot] = {**meta, "fallback": True, "fallback_reason": str(exc)}

    def unload(self, slot: str) -> None:
        provider = self._slots.get(slot)
        if provider is not None and provider.is_loaded():
            provider.unload()

    def loaded_slots(self) -> list[str]:
        return [s for s, p in self._slots.items() if p.is_loaded()]

    def health(self) -> dict[str, dict]:
        return {slot: provider.health() for slot, provider in self._slots.items()}

    # ---- config-driven construction -----------------------------------

    def build_from_config(self, factory) -> None:
        """Build providers from models_config using the given factory.

        The factory signature: factory(slot, provider_name, config, slot_config) -> BaseAdapter
        """
        for kind, slots in self._config.items():
            for slot_name, slot_cfg in slots.items():
                slot = f"{kind}.{slot_name}"
                provider_name = slot_cfg.get("provider", "")
                if not provider_name:
                    raise ConfigurationError(f"model slot {slot} has no provider")
                provider_settings = self._providers_config.get(provider_name, {}) or {}
                cfg = {**provider_settings, **slot_cfg}
                cfg["kind"] = kind
                try:
                    provider = factory(slot, provider_name, cfg)
                except ProviderNotAvailableError:
                    raise
                except Exception as exc:
                    raise ConfigurationError(f"cannot build provider for {slot}: {exc}") from exc
                self.register_slot(slot, provider, {"kind": kind, "provider": provider_name})
                if getattr(provider, "_fallback", False):
                    self._slot_meta[slot]["fallback"] = True
                    self._slot_meta[slot]["fallback_reason"] = str(
                        provider.config.get("fallback_reason", "")
                    )
                log.info("registered model slot %s -> %s (%s)", slot, provider_name, cfg.get("model_id", ""))
