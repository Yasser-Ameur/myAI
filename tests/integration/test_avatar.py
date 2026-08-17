import asyncio
import time

import pytest

from companion.application.avatar import (
    AvatarService,
    ConsoleAvatarDriver,
    ExpressionController,
)
from companion.core.clock import SystemClock
from companion.core.events import (
    EVENT_FACE_OBSERVATION_UPDATED,
    EVENT_RESPONSE_PLAN_CREATED,
    EVENT_SPEECH_CHUNK_READY,
    EventBus,
)


@pytest.mark.asyncio
async def test_expression_controller_smooths():
    driver = ConsoleAvatarDriver()
    controller = ExpressionController(driver=driver)
    from companion.domain.emotion import AffectVector

    controller.set_affect(AffectVector(valence=0.9, arousal=0.5))
    controller.tick()
    first_smile = controller.state.smile
    controller.tick()
    assert controller.state.smile >= first_smile  # converges monotonically upward
    assert driver.last_frame


@pytest.mark.asyncio
async def test_avatar_reacts_to_events():
    driver = ConsoleAvatarDriver()
    bus = EventBus(clock=SystemClock())
    avatar = AvatarService(ExpressionController(driver=driver), bus=bus)
    avatar.attach_bus(bus)

    bus.publish(EVENT_RESPONSE_PLAN_CREATED, {
        "affect": {"valence": 0.8, "arousal": 0.3, "blend": {}},
        "gaze_target": "user",
    })
    bus.publish(EVENT_SPEECH_CHUNK_READY, {"text": "hi", "duration_ms": 100})
    await asyncio.sleep(0.05)
    assert avatar._agent_state.speaking
    avatar.stop()


def test_console_driver_output():
    from companion.application.avatar import CanonicalFaceState

    driver = ConsoleAvatarDriver()
    driver.render(CanonicalFaceState(smile=0.8, emotion="joy"))
    assert "D" in driver.last_frame  # smiling mouth


def test_expression_controller_blinks():
    driver = ConsoleAvatarDriver()
    controller = ExpressionController(driver=driver)
    controller._next_blink = time.monotonic() - 0.01
    controller.tick()
    assert controller.state.eye_open_left < 0.5  # blink in progress
    controller._blink_until = time.monotonic() - 0.01
    controller.tick()
    assert controller.state.eye_open_left > 0.5  # blink finished


@pytest.mark.asyncio
async def test_avatar_mirrors_user_attention():
    driver = ConsoleAvatarDriver()
    bus = EventBus(clock=SystemClock())
    avatar = AvatarService(ExpressionController(driver=driver), bus=bus)
    avatar.attach_bus(bus)

    bus.publish(EVENT_FACE_OBSERVATION_UPDATED, {
        "gaze": {"estimated_attention": 0.1},
        "head_pose": {"yaw": 0.5, "pitch": 0.0, "roll": 0.0},
    })
    await asyncio.sleep(0.05)
    assert avatar._controller.state.gaze_x > 0.0  # glance toward user

    bus.publish(EVENT_FACE_OBSERVATION_UPDATED, {
        "gaze": {"estimated_attention": 0.9},
        "head_pose": {"yaw": 0.1, "pitch": 0.0, "roll": 0.0},
    })
    await asyncio.sleep(0.05)
    assert avatar._controller.state.gaze_x == 0.0  # eye contact
    avatar.stop()


def test_mouth_follows_prosody_envelope():
    driver = ConsoleAvatarDriver()
    controller = ExpressionController(driver=driver)
    controller.set_speech_envelope([0.9, 0.1, 0.9], duration_ms=600)
    controller.set_speaking(True)
    controller._envelope_start = time.monotonic() - 0.15  # ~1/4 through 600ms
    controller.tick()
    assert controller.state.jaw_open > 0.5  # loud syllable
    controller.set_speaking(False)
    controller.tick()
    controller.tick()
    assert controller.state.jaw_open < 0.3  # mouth closes when quiet
