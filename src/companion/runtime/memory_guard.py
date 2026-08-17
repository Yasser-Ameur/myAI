"""Runtime memory monitoring.

If RAM climbs above threshold the guard escalates: stop background inference,
unload idle models, reduce embedding concurrency and perception frequency —
while preserving interactive responsiveness. Never let the machine freeze.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class MemoryGuardState:
    level: str = "normal"       # normal | elevated | critical
    ram_used_mb: int = 0
    threshold_elevated_mb: int = 0
    threshold_critical_mb: int = 0
    actions_taken: list[str] = field(default_factory=list)


class MemoryGuard:
    def __init__(self, threshold_elevated_mb: int = 12000,
                 threshold_critical_mb: int = 14000,
                 process_ram_provider=None) -> None:
        self.threshold_elevated_mb = threshold_elevated_mb
        self.threshold_critical_mb = threshold_critical_mb
        self.state = MemoryGuardState(
            threshold_elevated_mb=threshold_elevated_mb,
            threshold_critical_mb=threshold_critical_mb,
        )
        self._process_ram = process_ram_provider or self._default_process_ram

    def current_process_ram_mb(self) -> int:
        try:
            return self._process_ram()
        except Exception:
            return 0

    def check(self, lifecycle=None, scheduler=None) -> str:
        used = self.current_process_ram_mb()
        self.state.ram_used_mb = used
        if used >= self.threshold_critical_mb:
            self.state.level = "critical"
            self._escalate_critical(lifecycle, scheduler)
        elif used >= self.threshold_elevated_mb:
            self.state.level = "elevated"
            self._escalate_elevated(lifecycle, scheduler)
        else:
            self.state.level = "normal"
        return self.state.level

    def _escalate_elevated(self, lifecycle, scheduler) -> None:
        actions = ["stop_background"]
        if scheduler is not None:
            if scheduler.stats.background_active:
                scheduler.end_background()
                actions.append("force_end_background")
        if lifecycle is not None:
            for slot in lifecycle.stats()["loaded"]:
                if slot.startswith(("stt", "tts", "embeddings")):
                    lifecycle.unload(slot)
                    actions.append(f"unload_{slot}")
        self._record(actions)

    def _escalate_critical(self, lifecycle, scheduler) -> None:
        self._escalate_elevated(lifecycle, scheduler)
        if lifecycle is not None:
            for slot in lifecycle.stats()["loaded"]:
                if slot.startswith(("llm",)):
                    lifecycle.unload(slot)
                    self._record([f"unload_{slot}"])
        self._record(["preserve_interactivity"])

    def _record(self, actions: list[str]) -> None:
        for a in actions:
            if a not in self.state.actions_taken:
                self.state.actions_taken.append(a)
        if self.state.actions_taken:
            log.warning("memory guard [%s] ram=%dMB actions=%s",
                        self.state.level, self.state.ram_used_mb, self.state.actions_taken)

    @staticmethod
    def _default_process_ram() -> int:
        try:
            import psutil

            return int(psutil.Process().memory_info().rss / (1024 * 1024))
        except ImportError:
            return 0
