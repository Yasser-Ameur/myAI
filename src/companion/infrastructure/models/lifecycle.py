"""ModelLifecycleManager.

Owns which heavy models are resident at any time. Enforces
max_heavy_models_resident and the AI memory budget via lazy loading, LRU
eviction and unload hooks. Heavy background jobs check it before running
concurrently with conversational inference.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class ModelLifecycleManager:
    def __init__(
        self,
        registry,
        max_heavy_resident: int = 1,
        max_ai_ram_mb: int = 8192,
        heavy_kinds: frozenset[str] = frozenset({"llm", "stt"}),
    ) -> None:
        self._registry = registry
        self._max_heavy_resident = max_heavy_resident
        self._max_ai_ram_mb = max_ai_ram_mb
        self._heavy_kinds = heavy_kinds

    def load(self, slot: str) -> None:
        provider = self._registry.get(slot)
        if provider.is_loaded():
            provider.mark_used()
            return
        if self._is_heavy(slot):
            self._make_room_for(provider.estimate_ram_mb(), exclude=slot)
        self._registry.ensure_loaded(slot)
        log.info("loaded %s (est %d MB, resident now %s)",
                 slot, provider.estimate_ram_mb(), self._registry.loaded_slots())

    def warm(self, slot: str) -> None:
        provider = self._registry.get_optional(slot)
        if provider is not None and not provider.is_loaded():
            try:
                self.load(slot)
            except Exception as exc:  # warm failures are non-fatal
                log.warning("warm failed for %s: %s", slot, exc)

    def unload(self, slot: str) -> None:
        self._registry.unload(slot)

    def is_loaded(self, slot: str) -> bool:
        provider = self._registry.get_optional(slot)
        return bool(provider and provider.is_loaded())

    def memory_estimate(self, slot: str) -> int:
        provider = self._registry.get(slot)
        return provider.estimate_ram_mb()

    def last_used(self, slot: str) -> float:
        provider = self._registry.get(slot)
        return provider.last_used()

    def resident_ram_mb(self) -> int:
        total = 0
        for slot in self._registry.loaded_slots():
            total += self.memory_estimate(slot)
        return total

    def can_run_heavy_background(self) -> bool:
        """True if no heavy interactive model is resident (or RAM is safe)."""
        heavy_loaded = [s for s in self._registry.loaded_slots() if self._is_heavy(s)]
        return len(heavy_loaded) == 0 and self.resident_ram_mb() < self._max_ai_ram_mb

    def stats(self) -> dict:
        return {
            "loaded": self._registry.loaded_slots(),
            "resident_ram_mb": self.resident_ram_mb(),
            "max_ai_ram_mb": self._max_ai_ram_mb,
            "max_heavy_resident": self._max_heavy_resident,
        }

    # ---- internals -----------------------------------------------------

    def _is_heavy(self, slot: str) -> bool:
        kind = self._registry.meta(slot).get("kind", "")
        return kind in self._heavy_kinds

    def _make_room_for(self, needed_mb: int, exclude: str) -> None:
        # Evict LRU heavy models until either below resident cap or safe to fit.
        heavy = sorted(
            (s for s in self._registry.loaded_slots() if self._is_heavy(s) and s != exclude),
            key=lambda s: self._registry.get(s).last_used(),
        )
        while len(heavy) >= self._max_heavy_resident:
            victim = heavy.pop(0)
            log.info("evicting LRU heavy model %s to make room", victim)
            self._registry.unload(victim)
        # Also respect RAM budget
        budget_headroom = self._max_ai_ram_mb - self.resident_ram_mb() - needed_mb
        while budget_headroom < 0 and heavy:
            victim = heavy.pop(0)
            log.info("evicting %s to respect RAM budget", victim)
            self._registry.unload(victim)
            budget_headroom = self._max_ai_ram_mb - self.resident_ram_mb() - needed_mb
