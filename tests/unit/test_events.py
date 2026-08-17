import pytest

from companion.core.clock import SystemClock
from companion.core.events import (
    EVENT_MEMORY_COMMITTED,
    EVENT_SPEECH_CHUNK_READY,
    DropPolicy,
    Event,
    EventBus,
)


@pytest.mark.asyncio
async def test_publish_async_delivers_in_order():
    bus = EventBus(clock=SystemClock())
    received: list[str] = []

    async def handler(event: Event):
        received.append(event.payload["msg"])

    bus.subscribe(EVENT_MEMORY_COMMITTED, handler)
    for i in range(5):
        await bus.publish_async(EVENT_MEMORY_COMMITTED, {"msg": str(i)})
    import asyncio

    await asyncio.sleep(0.02)
    assert received == ["0", "1", "2", "3", "4"]


@pytest.mark.asyncio
async def test_publish_enqueues_for_async_subscriber():
    bus = EventBus(clock=SystemClock())
    got: list[str] = []

    async def handler(event: Event):
        got.append(event.kind)

    bus.subscribe(EVENT_SPEECH_CHUNK_READY, handler)
    bus.publish(EVENT_SPEECH_CHUNK_READY)
    import asyncio

    await asyncio.sleep(0.02)
    assert got == [EVENT_SPEECH_CHUNK_READY]


@pytest.mark.asyncio
async def test_drop_newest_policy_drops_when_full():
    bus = EventBus(clock=SystemClock())
    processed = []

    async def slow(event: Event):
        import asyncio

        await asyncio.sleep(0.05)
        processed.append(event.payload["n"])

    bus.subscribe(EVENT_MEMORY_COMMITTED, slow, policy=DropPolicy.DROP_NEWEST,
                  queue_size=1)
    for i in range(20):
        await bus.publish_async(EVENT_MEMORY_COMMITTED, {"n": i})
    import asyncio

    await asyncio.sleep(0.2)
    # DROP_NEWEST keeps the first enqueued item; the backlog never grows unbounded
    assert len(processed) <= 4


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    bus = EventBus(clock=SystemClock())
    calls: list[int] = []

    async def h(event: Event):
        calls.append(1)

    sub = bus.subscribe(EVENT_MEMORY_COMMITTED, h)
    bus.unsubscribe(sub)
    bus.publish(EVENT_MEMORY_COMMITTED)
    import asyncio

    await asyncio.sleep(0.02)
    assert calls == []
    assert bus.subscriber_count(EVENT_MEMORY_COMMITTED) == 0
