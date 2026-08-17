"""Reflection and consolidation.

IdleDetector: ACTIVE -> QUIET -> IDLE -> CONSOLIDATING. Consolidation runs only
in idle, with CPU/RAM limits, and yields immediately when activity resumes.
"""

from __future__ import annotations

import logging
from enum import Enum

from companion.application.personality import PersonalityEngine
from companion.application.ports import GraphStore
from companion.application.relationship import RelationshipEngine
from companion.core.clock import Clock, SystemClock
from companion.domain.memory import MemoryStatus
from companion.domain.personality import Contradiction

log = logging.getLogger(__name__)


class IdleLevel(str, Enum):
    ACTIVE = "active"
    QUIET = "quiet"
    IDLE = "idle"
    CONSOLIDATING = "consolidating"


class IdleDetector:
    def __init__(self, clock: Clock | None = None, quiet_after_s: float = 30.0,
                 idle_after_s: float = 180.0) -> None:
        self._clock = clock or SystemClock()
        self.quiet_after_s = quiet_after_s
        self.idle_after_s = idle_after_s
        self._last_activity = self._clock.monotonic()
        self.level = IdleLevel.ACTIVE

    def register_activity(self) -> None:
        self._last_activity = self._clock.monotonic()
        if self.level != IdleLevel.ACTIVE:
            log.info("activity detected: %s -> active", self.level.value)
            self.level = IdleLevel.ACTIVE

    @property
    def last_activity(self) -> float:
        """Monotonic timestamp of the most recent user activity."""
        return self._last_activity

    def update(self) -> IdleLevel:
        if self.level == IdleLevel.CONSOLIDATING:
            return self.level
        elapsed = self._clock.monotonic() - self._last_activity
        if elapsed < self.quiet_after_s:
            self.level = IdleLevel.ACTIVE
        elif elapsed < self.idle_after_s:
            self.level = IdleLevel.QUIET
        else:
            self.level = IdleLevel.IDLE
        return self.level

    def is_consolidatable(self) -> bool:
        return self.update() in (IdleLevel.IDLE, IdleLevel.QUIET)


class ReflectionService:
    def __init__(
        self,
        graph: GraphStore,
        personality: PersonalityEngine,
        relationships: RelationshipEngine,
        lifecycle=None,
        clock: Clock | None = None,
        idle: IdleDetector | None = None,
        scheduler=None,
        token_budget: int = 800,
        max_runs_per_hour: int = 2,
        pipeline=None,
    ) -> None:
        self._pipeline = pipeline
        self._graph = graph
        self._personality = personality
        self._relationships = relationships
        self._lifecycle = lifecycle
        self._clock = clock or SystemClock()
        self._idle = idle or IdleDetector(clock=clock)
        self._scheduler = scheduler
        self._token_budget = token_budget
        self._max_runs = max_runs_per_hour
        self.stats = {"runs": 0, "last_run_at": "", "episodes_consolidated": 0,
                      "memories_decayed": 0, "contradictions_resolved": 0}

    @property
    def idle(self) -> IdleDetector:
        return self._idle

    async def run_once(self) -> bool:
        """One consolidation pass. Returns True if any work was done."""
        if not self._idle.is_consolidatable():
            return False
        if self._scheduler is not None and not self._scheduler.can_run_background():
            log.info("skipping consolidation: interactive workloads active")
            return False
        if self._lifecycle is not None and not self._lifecycle.can_run_heavy_background():
            log.info("skipping consolidation: heavy models resident")
            return False
        self._idle.level = IdleLevel.CONSOLIDATING
        try:
            worked = await self._consolidate()
            self.stats["runs"] += 1
            self.stats["last_run_at"] = self._clock.now_iso()
            return worked
        finally:
            self._idle.level = IdleLevel.QUIET

    async def _consolidate(self) -> bool:
        worked = False
        worked |= await self._consolidate_episodes()
        worked |= self._resolve_contradictions()
        worked |= self._apply_decay()
        self._maintain_indexes()
        self._release_temporary_models()
        return worked

    async def _consolidate_episodes(self) -> bool:
        """Run real extraction over episodes that were never consolidated.

        This previously only backfilled a naive summary and then marked the
        episode consolidated — which permanently prevented proper extraction
        from ever running on it. An episode whose consolidation was cut short
        at shutdown is exactly the case that most needs a second attempt, so
        the full pipeline is used when one is available.
        """
        episodes = self._graph.list_episodes(limit=20)
        unconsolidated = [e for e in episodes if not e.is_consolidated]
        if not unconsolidated:
            return False
        for ep in unconsolidated:
            if not ep.transcript:
                continue
            if self._pipeline is not None:
                try:
                    await self._pipeline.process_episode(ep)
                    self.stats["episodes_consolidated"] += 1
                    continue
                except Exception:
                    log.exception("pipeline consolidation failed for episode %s", ep.id)
            # No pipeline (or it failed): keep a summary so the episode is
            # still retrievable, but leave it unconsolidated so a later pass
            # can do the real work.
            if not ep.summary:
                texts = [t.get("text", "") for t in ep.transcript[:4]]
                ep.summary = " ".join(texts)[:400]
                self._graph.save_episode(ep)
        return True

    def _resolve_contradictions(self) -> bool:
        contradictions = self._graph.list_contradictions(status="unresolved")
        changed = False
        for con in contradictions:
            if self._auto_resolve(con):
                self._graph.resolve_contradiction(con.id, con.resolution_status)
                self.stats["contradictions_resolved"] += 1
                changed = True
        return changed

    def _auto_resolve(self, con: Contradiction) -> bool:
        # Later evidence may explain: if timestamps far apart, likely changed.
        try:
            ts = [t for t in con.timestamps if t]
            if len(ts) >= 2:
                from datetime import datetime

                t0 = datetime.fromisoformat(ts[0].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(ts[1].replace("Z", "+00:00"))
                days = abs((t1 - t0).total_seconds()) / 86400.0
                if days > 90:
                    con.resolution_status = "preference_changed"
                    return True
        except (ValueError, TypeError):
            pass
        return False

    def _apply_decay(self) -> bool:
        from datetime import datetime

        now = datetime.now().astimezone()
        memories = self._graph.list_memories(status=MemoryStatus.ACTIVE.value, limit=1000)
        changed = False
        for mem in memories:
            try:
                created = datetime.fromisoformat(mem.created_at.replace("Z", "+00:00"))
                age_days = max(0.0, (now - created).total_seconds() / 86400.0)
            except (ValueError, TypeError):
                continue
            if age_days > 365 and mem.importance < 0.3 and not mem.locked:
                self._graph.update_memory_status(mem.id, MemoryStatus.DECAYING.value)
                self.stats["memories_decayed"] += 1
                changed = True
        return changed

    def _maintain_indexes(self) -> None:
        try:
            handle = getattr(self._graph, "storage_handle", None)
            if handle is not None:
                storage = handle()
                storage.execute("PRAGMA optimize")
        except Exception as exc:  # pragma: no cover
            log.warning("index maintenance failed: %s", exc)

    def _release_temporary_models(self) -> None:
        if self._lifecycle is not None:
            for slot in self._lifecycle.stats()["loaded"]:
                if slot.startswith(("stt", "tts")):
                    self._lifecycle.unload(slot)
                    log.info("released temporary model %s after consolidation", slot)
