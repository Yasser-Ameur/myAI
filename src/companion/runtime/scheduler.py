"""CPU scheduling: workload classes and a background gate.

REALTIME (VAD, face tracking, audio buffering, avatar) and INTERACTIVE (STT,
LLM, TTS) tasks take priority; BACKGROUND work (memory extraction, embedding,
graph maintenance, reflection, consolidation) must never monopolize cores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


class WorkloadClass(str, Enum):
    REALTIME = "realtime"
    INTERACTIVE = "interactive"
    BACKGROUND = "background"


@dataclass
class WorkloadStats:
    interactive_active: bool = False
    background_active: bool = False
    background_count: int = 0
    max_background_workers: int = 1
    last_background_blocked: str = ""


class Scheduler:
    """Tracks interactive vs background work and gates background tasks."""

    def __init__(self, max_background_workers: int = 1) -> None:
        self.max_background_workers = max_background_workers
        self._background_tokens = 0
        self.stats = WorkloadStats(max_background_workers=max_background_workers)
        self._interactive_depth = 0

    def begin_interactive(self) -> None:
        self._interactive_depth += 1
        self.stats.interactive_active = self._interactive_depth > 0

    def end_interactive(self) -> None:
        self._interactive_depth = max(0, self._interactive_depth - 1)
        self.stats.interactive_active = self._interactive_depth > 0

    def can_run_background(self) -> bool:
        if self._interactive_depth > 0:
            self.stats.last_background_blocked = "interactive_load"
            return False
        if self.stats.background_active and self.stats.background_count >= self.max_background_workers:
            self.stats.last_background_blocked = "worker_limit"
            return False
        return True

    def begin_background(self) -> None:
        self.stats.background_active = True
        self.stats.background_count += 1

    def end_background(self) -> None:
        self.stats.background_count = max(0, self.stats.background_count - 1)
        self.stats.background_active = self.stats.background_count > 0

    def class_of(self, task: str) -> WorkloadClass:
        if task in ("vad", "face_tracking", "audio_buffering", "avatar_render"):
            return WorkloadClass.REALTIME
        if task in ("stt", "llm", "tts", "conversation"):
            return WorkloadClass.INTERACTIVE
        return WorkloadClass.BACKGROUND
