import asyncio

import pytest

from companion.core.events import (
    EVENT_AVATAR_FIRST_MOTION,
    EVENT_RESPONSE_COMPLETE,
    EVENT_RESPONSE_PLAN_CREATED,
    EVENT_RESPONSE_TOKEN_GENERATED,
    EVENT_RETRIEVAL_COMPLETE,
    EVENT_SPEECH_CHUNK_READY,
    EVENT_SPEECH_PLAYBACK_STARTED,
    EVENT_TRANSCRIPT_FINAL,
    EventBus,
)
from companion.runtime.benchmarks import PipelineTracer


@pytest.mark.asyncio
async def test_tracer_records_stage_breakdown():
    bus = EventBus()
    tracer = PipelineTracer(bus)
    tracer.attach()
    tracer.reset()

    await asyncio.sleep(0.01)
    bus.publish(EVENT_TRANSCRIPT_FINAL, {"text": "hi", "language": "en"})
    await asyncio.sleep(0.02)
    bus.publish(EVENT_RETRIEVAL_COMPLETE, {"query": "hi"})
    await asyncio.sleep(0.02)
    bus.publish(EVENT_RESPONSE_PLAN_CREATED, {"intent": "chat"})
    await asyncio.sleep(0.03)
    bus.publish(EVENT_RESPONSE_TOKEN_GENERATED, {"token": "He"})
    await asyncio.sleep(0.02)
    bus.publish(EVENT_RESPONSE_COMPLETE, {"text": "Hello"})
    await asyncio.sleep(0.03)
    bus.publish(EVENT_SPEECH_CHUNK_READY, {"duration_ms": 500})
    await asyncio.sleep(0.01)
    bus.publish(EVENT_SPEECH_PLAYBACK_STARTED, {"duration_ms": 500})
    await asyncio.sleep(0.02)
    bus.publish(EVENT_AVATAR_FIRST_MOTION, {"emotion": "joy"})
    await asyncio.sleep(0.05)

    b = tracer.breakdown()
    assert b["retrieval_ms"] is not None and b["retrieval_ms"] > 10.0
    assert b["plan_ms"] is not None and b["plan_ms"] > 10.0
    assert b["llm_first_token_ms"] is not None
    assert b["llm_generation_ms"] is not None
    assert b["tts_synth_ms"] is not None
    assert b["playback_start_ms"] is not None
    assert b["avatar_first_motion_ms"] is not None
    assert b["total_ms"] is not None
    assert b["speech_input_ms"] is None  # no mic audio in this turn
    assert b["stt_ms"] is None
    tracer.detach()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_tracer_attaches_once_and_detaches():
    bus = EventBus()
    tracer = PipelineTracer(bus)
    tracer.attach()
    tracer.attach()
    assert len(tracer._subs) == 10
    tracer.detach()
    tracer.detach()
    assert tracer._subs == []
    await asyncio.sleep(0.02)
