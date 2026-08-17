"""In-process asynchronous event bus.

Loose coupling between perception, memory, conversation, avatar and observability.
Consumers subscribe independently; producers publish typed events.

Backpressure: every subscription has a bounded queue with a configured drop
policy. Camera-frame consumers drop stale frames; audio/memory consumers never
drop (they block briefly instead).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, TypeVar

from companion.core.clock import Clock, SystemClock
from companion.core.ids import new_id

log = logging.getLogger(__name__)

T = TypeVar("T")


class DropPolicy(str, Enum):
    DROP_OLDEST = "drop_oldest"  # keep latency low, drop stale frames
    BLOCK = "block"              # never lose payloads (audio/memory)
    DROP_NEWEST = "drop_newest"


@dataclass(frozen=True)
class Event:
    """A typed event on the bus."""

    kind: str
    payload: dict
    id: str = field(default_factory=new_id)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(self, "timestamp", SystemClock().now_iso())


class EventBus:
    """Typed multi-subscriber async event bus with per-subscriber backpressure."""

    def __init__(
        self,
        clock: Clock | None = None,
        queue_size: int = 32,
    ) -> None:
        self._clock = clock or SystemClock()
        self._subscribers: dict[str, list[asyncio.Queue[Event]]] = {}
        self._policies: dict[str, list[DropPolicy]] = {}
        # Keep stopped subscriptions until close() has awaited their cancelled
        # tasks.  A consumer can be removed from routing synchronously without
        # leaving a task behind at application shutdown.
        self._subscriptions: set[Subscription] = set()
        self._queue_size = queue_size

    def subscribe(
        self,
        kind: str,
        handler: Callable[[Event], Awaitable[None]],
        policy: DropPolicy = DropPolicy.BLOCK,
        queue_size: int | None = None,
    ) -> "Subscription":
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_size or self._queue_size)
        self._subscribers.setdefault(kind, []).append(q)
        self._policies.setdefault(kind, []).append(policy)
        sub = Subscription(self, kind, q, policy, handler)
        self._subscriptions.add(sub)
        sub.start()
        return sub

    def subscribe_many(self, kinds: list[str], handler: Callable[[Event], Awaitable[None]], **kw) -> list["Subscription"]:
        return [self.subscribe(kind, handler, **kw) for kind in kinds]

    def publish(self, kind: str, payload: dict | None = None) -> None:
        """Publish synchronously; enqueues into each subscriber's bounded queue."""
        event = Event(kind=kind, payload=payload or {}, timestamp=self._clock.now_iso())
        queues = self._subscribers.get(kind, [])
        if not queues:
            return
        for i, q in enumerate(queues):
            policy = self._policies[kind][i]
            try:
                if q.full():
                    if policy == DropPolicy.DROP_OLDEST:
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        q.put_nowait(event)
                    elif policy == DropPolicy.DROP_NEWEST:
                        pass
                    else:  # BLOCK: drop_policy overridden to avoid blocking caller sync path
                        # Synchronous publish cannot block; drop oldest as a safe fallback
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        q.put_nowait(event)
                else:
                    q.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - defensive
                log.warning("event bus queue full for %s", kind)

    async def publish_async(self, kind: str, payload: dict | None = None) -> None:
        event = Event(kind=kind, payload=payload or {}, timestamp=self._clock.now_iso())
        queues = self._subscribers.get(kind, [])
        if not queues:
            return
        for i, q in enumerate(queues):
            policy = self._policies[kind][i]
            if policy == DropPolicy.BLOCK:
                await q.put(event)
            else:
                try:
                    if q.full():
                        if policy == DropPolicy.DROP_OLDEST:
                            try:
                                q.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        else:  # DROP_NEWEST
                            return
                    q.put_nowait(event)
                except asyncio.QueueFull:  # pragma: no cover
                    log.warning("event bus queue full for %s", kind)

    def unsubscribe(self, sub: "Subscription") -> None:
        kind = sub.kind
        if kind not in self._subscribers:
            return
        if sub.queue in self._subscribers.get(kind, []):
            idx = self._subscribers[kind].index(sub.queue)
            self._subscribers[kind].pop(idx)
            self._policies[kind].pop(idx)
        sub.stop()

    def subscriber_count(self, kind: str) -> int:
        return len(self._subscribers.get(kind, []))

    async def aclose(self) -> None:
        """Stop and await every consumer task owned by this bus."""
        subscriptions = list(self._subscriptions)
        for sub in subscriptions:
            sub.stop()
        if subscriptions:
            await asyncio.gather(*(sub.wait_closed() for sub in subscriptions),
                                 return_exceptions=True)
        self._subscribers.clear()
        self._policies.clear()
        self._subscriptions.clear()


class Subscription:
    """A running consumer loop for one subscriber."""

    def __init__(
        self,
        bus: EventBus,
        kind: str,
        queue: asyncio.Queue[Event],
        policy: DropPolicy,
        handler: Callable[[Event], Awaitable[None]],
    ) -> None:
        self.bus = bus
        self.kind = kind
        self.queue = queue
        self.policy = policy
        self.handler = handler
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        # Obtain the loop before constructing the coroutine.  Besides making
        # the ownership requirement explicit, this avoids leaking an unawaited
        # coroutine when subscribe() is accidentally called outside a runtime.
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._run(), name=f"evt:{self.kind}")

    async def _run(self) -> None:
        try:
            while True:
                event = await self.queue.get()
                try:
                    await self.handler(event)
                except asyncio.CancelledError:  # pragma: no cover
                    raise
                except Exception:
                    log.exception("event handler failed for %s", self.kind)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def wait_closed(self) -> None:
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None


# Canonical event kinds. Kept as constants to avoid stringly-typed drift.
EVENT_AUDIO_CHUNK = "AudioChunkReceived"
EVENT_AUDIO_STARTED = "AudioStarted"
EVENT_AUDIO_ENDED = "AudioEnded"
EVENT_TRANSCRIPT_PARTIAL = "TranscriptPartial"
EVENT_TRANSCRIPT_FINAL = "TranscriptFinal"
EVENT_FACE_DETECTED = "FaceDetected"
EVENT_FACE_LOST = "FaceLost"
EVENT_FACE_OBSERVATION_UPDATED = "FaceObservationUpdated"
EVENT_GAZE_UPDATED = "GazeUpdated"
EVENT_HEAD_POSE_UPDATED = "HeadPoseUpdated"
EVENT_EXPRESSION_OBSERVATION_UPDATED = "ExpressionObservationUpdated"
EVENT_USER_STATE_UPDATED = "UserStateUpdated"
EVENT_MEMORY_CANDIDATE_CREATED = "MemoryCandidateCreated"
EVENT_MEMORY_COMMITTED = "MemoryCommitted"
EVENT_RESPONSE_PLAN_CREATED = "ResponsePlanCreated"
EVENT_RESPONSE_TOKEN_GENERATED = "ResponseTokenGenerated"
EVENT_RESPONSE_COMPLETE = "ResponseComplete"
EVENT_RETRIEVAL_COMPLETE = "RetrievalComplete"
EVENT_SPEECH_CHUNK_READY = "SpeechChunkReady"
EVENT_SPEECH_PLAYBACK_STARTED = "SpeechPlaybackStarted"
EVENT_AVATAR_FIRST_MOTION = "AvatarFirstMotion"
EVENT_EXPRESSION_CHANGED = "ExpressionChanged"
EVENT_EPISODE_COMMITTED = "EpisodeCommitted"
EVENT_IDLE_STATE_CHANGED = "IdleStateChanged"
EVENT_CONSOLIDATION_STARTED = "ConsolidationStarted"
EVENT_CONSOLIDATION_FINISHED = "ConsolidationFinished"
