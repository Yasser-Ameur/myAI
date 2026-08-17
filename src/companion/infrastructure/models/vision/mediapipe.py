"""Face perception providers.

MediaPipe sees the USER's face and produces measurable observations. It never
decides what the user "feels". The avatar controller (rendering the AI's face)
is completely independent.
"""

from __future__ import annotations

import logging

from companion.core.clock import SystemClock
from companion.core.contracts import (
    BlendshapeSet,
    FaceObservation,
    Gaze,
    HeadPose,
    ModelCapability,
    VideoFrame,
)
from companion.core.errors import ProviderNotAvailableError, ProviderTimeoutError
from companion.infrastructure.models.base import BaseAdapter

log = logging.getLogger(__name__)


class MediaPipeFaceProvider(BaseAdapter):
    """MediaPipe Face Landmarker in live-stream mode.

    Outputs blendshapes, head pose (computed from landmarks) and a crude
    estimated-attention gaze signal. Temporal smoothing happens in the
    application layer (TemporalObservationBuffer).
    """

    provider_name = "mediapipe"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.require(
            "mediapipe",
            "mediapipe is not installed. Run: pip install 'myai[vision]'",
        )
        self._detector = None
        self._model_path = config.get("model_path", "")
        self._last_ts_ms = 0

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(name=self.model_id, estimated_ram_mb=self.estimate_ram_mb())

    def estimate_ram_mb(self) -> int:
        return int(self._params.get("estimated_ram_mb", 250))

    def _do_load(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "mediapipe is not installed. Run: pip install 'myai[vision]'"
            ) from exc
        if not self._model_path:
            raise ProviderNotAvailableError(
                "mediapipe face provider requires 'model_path' (face_landmarker.task)"
            )
        try:
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            raise ProviderNotAvailableError(f"mediapipe failed to load: {exc}") from exc

    async def analyze(self, frame: VideoFrame) -> FaceObservation:
        self.mark_used()
        if self._detector is None:
            raise ProviderNotAvailableError(f"model {self.model_id} not loaded")
        import asyncio

        try:
            result = await asyncio.to_thread(self._analyze_sync, frame)
        except Exception as exc:
            raise ProviderTimeoutError(f"face analysis failed: {exc}") from exc
        return result

    def _analyze_sync(self, frame: VideoFrame) -> FaceObservation:
        import mediapipe as mp
        import numpy as np

        ts = int(frame.timestamp * 1000) if frame.timestamp else int(SystemClock().unix() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        arr = np.frombuffer(frame.rgb, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        result = self._detector.detect_for_video(mp_image, ts)

        obs = FaceObservation(
            face_id="primary",
            timestamp=SystemClock().now_iso(),
            detected=False,
            blendshapes=BlendshapeSet(),
            head_pose=HeadPose(),
            gaze=Gaze(),
            confidence=0.0,
        )
        if not result.face_landmarks:
            return obs
        obs.detected = True
        obs.confidence = 0.8
        if result.face_blendshapes:
            values = {}
            for b in result.face_blendshapes[0]:
                values[b.category_name] = float(b.score)
            obs.blendshapes = BlendshapeSet(values=values)
        if result.facial_transformation_matrixes:
            m = result.facial_transformation_matrixes[0]
            obs.head_pose = self._head_pose_from_matrix(m)
        obs.gaze = self._gaze_from_landmarks(result.face_landmarks[0])
        return obs

    @staticmethod
    def _head_pose_from_matrix(m) -> HeadPose:
        import numpy as np

        mat = np.array([[m[0][0], m[0][1], m[0][2]],
                        [m[1][0], m[1][1], m[1][2]],
                        [m[2][0], m[2][1], m[2][2]]])
        # Standard rotation-matrix to Euler angles (approx).
        pitch = np.arctan2(mat[2, 1], np.sqrt(mat[0, 1] ** 2 + mat[1, 1] ** 2))
        yaw = np.arctan2(-mat[2, 0], mat[2, 2])
        roll = np.arctan2(-mat[0, 1], mat[1, 1])
        return HeadPose(pitch=float(pitch), yaw=float(yaw), roll=float(roll))

    @staticmethod
    def _gaze_from_landmarks(landmarks) -> Gaze:
        """Crude attention proxy: normalized eye-openness + iris position.

        This is a measurable signal, not a claim about what the user attends to.
        """
        lm = {i: (landmark.x, landmark.y) for i, landmark in enumerate(landmarks)}
        left_eye = MediaPipeFaceProvider._eye_openness(lm, [33, 133, 159, 145, 133, 155])
        right_eye = MediaPipeFaceProvider._eye_openness(lm, [362, 263, 386, 374, 380, 362])
        openness = 0.5 * (left_eye + right_eye)
        # iris offset relative to eye corners -> horizontal attention proxy
        x_l = (lm.get(33, (0, 0))[0] + lm.get(133, (0, 0))[0]) / 2.0
        x_r = (lm.get(362, (0, 0))[0] + lm.get(263, (0, 0))[0]) / 2.0
        gaze = max(0.0, min(1.0, openness * 0.6 + (0.5 + x_l - x_r) * 0.4))
        return Gaze(estimated_attention=float(gaze))

    @staticmethod
    def _eye_openness(lm: dict, indices: list[int]) -> float:
        try:
            top = lm[indices[2]]
            bottom = lm[indices[3]]
            corner1 = lm[indices[0]]
            corner2 = lm[indices[1]]
            eye_h = abs(bottom[1] - top[1])
            eye_w = abs(corner2[0] - corner1[0]) or 1e-6
            return min(1.0, max(0.0, (eye_h / eye_w) * 3.0))
        except (KeyError, IndexError):
            return 0.3


class MockFaceProvider(BaseAdapter):
    """Deterministic face observations for tests/demos without a camera.

    Supports a script of states keyed by frame index so simulation tests can
    verify the temporal buffer and state estimator.
    """

    provider_name = "mock"

    def __init__(self, config: dict, model_id: str = "") -> None:
        super().__init__(config, model_id)
        self.frame_index = 0
        self.states: list[dict] = config.get("states", [])
        self.default = config.get(
            "default",
            {"blendshapes": {"mouthSmileLeft": 0.2, "mouthSmileRight": 0.2},
             "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
             "attention": 0.5},
        )

    def _do_load(self) -> None:
        return None

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(name=self.model_id, estimated_ram_mb=0)

    async def analyze(self, frame: VideoFrame) -> FaceObservation:
        self.mark_used()
        state = self.default
        if self.states:
            state = self.states[self.frame_index % len(self.states)]
            self.frame_index += 1
        bs = dict(state.get("blendshapes", {}))
        hp = state.get("head_pose", {})
        return FaceObservation(
            face_id="primary",
            timestamp=SystemClock().now_iso(),
            detected=bool(state.get("detected", True)),
            blendshapes=BlendshapeSet(values=bs),
            head_pose=HeadPose(
                pitch=float(hp.get("pitch", 0.0)),
                yaw=float(hp.get("yaw", 0.0)),
                roll=float(hp.get("roll", 0.0)),
            ),
            gaze=Gaze(estimated_attention=float(state.get("attention", 0.5))),
            confidence=float(state.get("confidence", 0.7)),
        )


class NullFaceProvider(BaseAdapter):
    """No camera available: always 'no face detected'."""

    provider_name = "null"

    def _do_load(self) -> None:
        return None

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(name=self.model_id, estimated_ram_mb=0)

    async def analyze(self, frame: VideoFrame) -> FaceObservation:
        self.mark_used()
        return FaceObservation(face_id="primary", timestamp=SystemClock().now_iso(), detected=False, confidence=0.0)
