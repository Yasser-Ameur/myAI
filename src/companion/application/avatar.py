"""Avatar application layer.

The avatar renders the AI's face and is completely decoupled from perception
(MediaPipe sees the user; the avatar renders the AI). The language model never
directly controls the face: ResponsePlan -> CanonicalFaceState -> adapter.

CanonicalFaceState is the internal interchange format; adapters map it to
VRM/Godot/Three.js/Live2D/robotics later.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Protocol

from companion.core.events import (
    EVENT_AVATAR_FIRST_MOTION,
    EVENT_FACE_OBSERVATION_UPDATED,
    EVENT_RESPONSE_PLAN_CREATED,
    EVENT_SPEECH_CHUNK_READY,
    EVENT_USER_STATE_UPDATED,
    EventBus,
)
from companion.domain.agent import AgentState
from companion.domain.emotion import AffectVector

log = logging.getLogger(__name__)

TARGET_FPS = 30


@dataclass
class CanonicalFaceState:
    smile: float = 0.0
    frown: float = 0.0
    brow_left: float = 0.0
    brow_right: float = 0.0
    eye_open_left: float = 1.0
    eye_open_right: float = 1.0
    jaw_open: float = 0.0
    lip_press: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    emotion: str = "neutral"

    def to_dict(self) -> dict:
        return {k: float(v) for k, v in self.__dataclass_fields__.items() if k != "emotion"}


class AvatarDriver(Protocol):
    def render(self, state: CanonicalFaceState) -> None: ...
    def close(self) -> None: ...


class NullAvatarDriver:
    def render(self, state: CanonicalFaceState) -> None:
        return None

    def close(self) -> None:
        return None


class ConsoleAvatarDriver:
    """ASCII avatar proving the pipeline end-to-end.

    Renders a 3-line face with blink, brows, gaze and mouth states so the
    multimodal session can visibly react without a GPU.
    """

    def __init__(self) -> None:
        self.last_frame = ""
        self._frames = []

    def render(self, state: CanonicalFaceState) -> None:
        open_l = "O" if state.eye_open_left > 0.5 else "."
        open_r = "O" if state.eye_open_right > 0.5 else "."
        if state.smile > 0.4:
            mouth = "D"
        elif state.frown > 0.4:
            mouth = "("
        elif state.jaw_open > 0.3:
            mouth = "o"
        elif state.lip_press > 0.3:
            mouth = "-"
        else:
            mouth = "_"
        brow_l = "\\" if state.brow_left > 0.4 else ("/" if state.brow_left < -0.3 else " ")
        brow_r = "/" if state.brow_right > 0.4 else ("\\" if state.brow_right < -0.3 else " ")
        gx = int(max(-1, min(1, state.gaze_x)) * 1)
        gx = max(-1, min(1, gx))
        pos = "left" if gx < 0 else ("right" if gx > 0 else "center")
        frame = (
            f" {brow_l}   {brow_r}\n"
            f"({open_l})--({open_r})  {pos}\n"
            f"  \\{mouth}/\n"
            f"  {state.emotion} head({state.head_yaw:+.1f},{state.head_pitch:+.1f})"
        )
        self.last_frame = frame
        self._frames.append(frame)
        if len(self._frames) > 300:
            self._frames.pop(0)

    def close(self) -> None:
        return None


class ExpressionController:
    """Maintains a smoothed canonical face state driven by affect + speech."""

    def __init__(self, driver: AvatarDriver | None = None,
                 smooth: float = 0.35, fps: int = TARGET_FPS) -> None:
        self._driver = driver or NullAvatarDriver()
        self.smooth = smooth
        self.state = CanonicalFaceState()
        self._speaking = False
        self._talking_amp = 0.0
        self._next_blink = time.monotonic() + 3.0 + _rand(0.0, 4.0)
        self._blink_until = 0.0
        self._fps = fps
        self._envelope: list[float] = []
        self._envelope_duration_s = 0.0
        self._envelope_start = 0.0

    def set_affect(self, affect: AffectVector) -> None:
        target = CanonicalFaceState(
            smile=max(0.0, min(1.0, (affect.valence + 1.0) / 2.0)),
            frown=max(0.0, min(1.0, -(affect.valence - 0.1) / 2.0)),
            brow_left=affect.blend.get("browInnerUp", 0.0),
            brow_right=affect.blend.get("browInnerUp", 0.0),
            emotion=_affect_label(affect),
        )
        self._blend(target)

    def set_gaze(self, gaze_x: float, gaze_y: float) -> None:
        self.state.gaze_x = gaze_x
        self.state.gaze_y = gaze_y

    def set_head_pose(self, pitch: float, yaw: float, roll: float = 0.0) -> None:
        self.state.head_pitch = pitch
        self.state.head_yaw = yaw
        self.state.head_roll = roll

    def set_speaking(self, speaking: bool) -> None:
        self._speaking = speaking
        if not speaking:
            self._envelope = []
            self._envelope_duration_s = 0.0

    def set_speech_envelope(self, envelope: list[float], duration_ms: float = 0.0) -> None:
        """Attach a real prosody envelope so the mouth follows the audio."""
        if envelope:
            self._envelope = envelope
            self._envelope_duration_s = max(duration_ms / 1000.0, 1e-3)
            self._envelope_start = time.monotonic()

    def tick(self) -> None:
        """Per-frame update: blink, talking jaw, render to driver."""
        now = time.monotonic()
        if self._speaking:
            if self._envelope:
                pos = ((now - self._envelope_start) / self._envelope_duration_s) * len(self._envelope)
                idx = min(len(self._envelope) - 1, max(0, int(pos)))
                amp = self._envelope[idx]
                self.state.jaw_open = min(1.0, amp * 1.4 + 0.15)
            else:
                t = now * 8.0
                self._talking_amp = max(self._talking_amp - 0.08, 0.0)
                self.state.jaw_open = 0.5 + 0.3 * math.sin(t) if self._talking_amp < 0.3 else 0.8
        else:
            self.state.jaw_open *= 0.4
        # autonomous blink every ~3-7s, lasting ~120ms
        if now >= self._next_blink:
            self._blink_until = now + 0.12
            self._next_blink = now + 3.0 + _rand(0.0, 4.0)
        blinking = now < self._blink_until
        self.state.eye_open_left = 0.15 if blinking else 1.0
        self.state.eye_open_right = 0.15 if blinking else 1.0
        self._driver.render(self.state)

    def _blend(self, target: CanonicalFaceState) -> None:
        s = self.smooth
        cur = self.state
        cur.smile = cur.smile + (target.smile - cur.smile) * s
        cur.frown = cur.frown + (target.frown - cur.frown) * s
        cur.brow_left = cur.brow_left + (target.brow_left - cur.brow_left) * s
        cur.brow_right = cur.brow_right + (target.brow_right - cur.brow_right) * s
        cur.emotion = target.emotion

    def close(self) -> None:
        self._driver.close()


class AvatarService:
    """Drives the avatar from events at ~30fps. Never blocks on LLM."""

    def __init__(
        self,
        controller: ExpressionController,
        bus: EventBus | None = None,
        fps: int = TARGET_FPS,
        agent_state: AgentState | None = None,
    ) -> None:
        self._controller = controller
        self.bus = bus or EventBus()
        self.fps = fps
        self._agent_state = agent_state or AgentState()
        self._affect = AffectVector()
        self._subs = []
        self._running = False
        self._expect_motion = False

    def attach_bus(self, bus: EventBus) -> None:
        if self._subs:
            return
        self.bus = bus
        self._subs.append(bus.subscribe(EVENT_RESPONSE_PLAN_CREATED, self._on_plan, policy="drop_oldest"))
        self._subs.append(bus.subscribe(EVENT_SPEECH_CHUNK_READY, self._on_speech, policy="drop_oldest"))
        self._subs.append(bus.subscribe(EVENT_USER_STATE_UPDATED, self._on_user_state, policy="drop_oldest"))
        self._subs.append(bus.subscribe(EVENT_FACE_OBSERVATION_UPDATED, self._on_face, policy="drop_oldest"))

    async def _on_plan(self, event) -> None:
        payload = event.payload
        try:
            affect = AffectVector.from_dict(payload.get("affect", {}))
        except Exception:
            affect = self._affect
        self._controller.set_affect(affect)
        if payload.get("gaze_target") == "away":
            self._controller.set_gaze(0.4, 0.1)
        else:
            self._controller.set_gaze(0.0, 0.0)
        self._expect_motion = True

    async def _on_speech(self, event) -> None:
        payload = event.payload
        self._controller.set_speaking(True)
        self._controller.set_speech_envelope(
            list(payload.get("amplitude") or []), float(payload.get("duration_ms", 0.0))
        )
        self._agent_state.speaking = True

    async def _on_face(self, event) -> None:
        """Mirror the user's attention as gaze (a proxy, not a claim of intent).

        Eye contact when the user looks at the camera; glance toward the user's
        direction when they look away.
        """
        payload = event.payload
        attention = float(payload.get("gaze", {}).get("estimated_attention", 0.5))
        yaw = float(payload.get("head_pose", {}).get("yaw", 0.0))
        if attention > 0.45:
            self._controller.set_gaze(0.0, 0.0)
        else:
            self._controller.set_gaze(max(-1.0, min(1.0, yaw * 3.0)), 0.0)

    async def _on_user_state(self, event) -> None:
        # Gentle empathetic mirroring, damped (not a copy).
        dims = event.payload.get("dimensions", {})
        valence = float(dims.get("valence", {}).get("value", 0.0))
        arousal = float(dims.get("arousal", {}).get("value", 0.0))
        self._affect = AffectVector(valence=valence * 0.4, arousal=arousal * 0.4)
        self._controller.set_affect(self._affect)
        self._controller.set_speaking(False)
        self._agent_state.speaking = False

    async def run(self) -> None:
        self._running = True
        interval = 1.0 / max(1, self.fps)
        while self._running:
            self._controller.tick()
            if self._expect_motion:
                st = self._controller.state
                if st.jaw_open > 0.05 or st.emotion != "neutral":
                    self.bus.publish(EVENT_AVATAR_FIRST_MOTION, {
                        "emotion": st.emotion, "jaw_open": st.jaw_open,
                    })
                    self._expect_motion = False
            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._running = False
        for sub in self._subs:
            self.bus.unsubscribe(sub)
        self._subs = []
        self._controller.close()


def _rand(lo: float, hi: float) -> float:
    import random

    return lo + random.random() * (hi - lo)


def _affect_label(affect: AffectVector) -> str:
    if affect.valence > 0.3 and affect.arousal > 0.2:
        return "joy"
    if affect.valence < -0.3 and affect.arousal > 0.2:
        return "concern"
    if affect.valence < -0.3:
        return "calm-support"
    if affect.valence > 0.2:
        return "warm"
    return "neutral"
