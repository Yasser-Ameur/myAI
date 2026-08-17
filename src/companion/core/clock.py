"""Time source abstraction so the rest of the system is testable and monotonic."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def utc_now(self) -> datetime: ...
    def now_iso(self) -> str: ...
    def monotonic(self) -> float: ...
    def unix(self) -> float: ...


class SystemClock:
    """Real clock. All timestamps the system persists go through this."""

    def utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def now_iso(self) -> str:
        return self.utc_now().isoformat()

    def monotonic(self) -> float:
        return time.monotonic()

    def unix(self) -> float:
        return time.time()


class FakeClock:
    """Deterministic clock for tests."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._t = start

    def advance(self, seconds: float) -> None:
        self._t += seconds

    def utc_now(self) -> datetime:
        return datetime.fromtimestamp(self._t, tz=timezone.utc)

    def now_iso(self) -> str:
        return self.utc_now().isoformat()

    def monotonic(self) -> float:
        return self._t

    def unix(self) -> float:
        return self._t
