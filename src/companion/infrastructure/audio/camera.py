"""Camera capture source built on OpenCV."""

from __future__ import annotations

import logging
import threading

from companion.core.contracts import VideoFrame
from companion.core.errors import CameraUnavailableError

log = logging.getLogger(__name__)


class CameraCaptureSource:
    """Capture RGB frames from a webcam into an asyncio queue on a thread."""

    def __init__(self, device: int = 0, fps: int = 10) -> None:
        self.device = device
        self.fps = fps
        self._cap = None
        self._queue = None
        self._loop = None
        self._thread = None
        self._running = False
        self._width = 320
        self._height = 240

    def _open(self):
        try:
            import cv2
        except ImportError as exc:
            raise CameraUnavailableError(f"opencv not installed: {exc}") from exc
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            cap.release()
            raise CameraUnavailableError(f"cannot open camera device {self.device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return cap

    def start(self, queue, sample_rate: int | None = None) -> None:
        if self._running:
            return
        try:
            self._cap = self._open()
        except CameraUnavailableError as exc:
            log.warning("camera unavailable: %s", exc)
            raise
        self._queue = queue
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="cam-capture")
        self._thread.start()
        log.info("camera capture started (device=%s)", self.device)

    def _capture_loop(self) -> None:
        import cv2

        frame_interval = 1.0 / max(1, self.fps)
        while self._running:
            import time

            t0 = time.monotonic()
            ok, frame = self._cap.read()
            if ok:
                frame = cv2.resize(frame, (self._width, self._height))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                vf = VideoFrame(
                    rgb=rgb.tobytes(),
                    width=self._width,
                    height=self._height,
                    timestamp=time.time(),
                )
                if self._loop is not None and self._queue is not None:
                    try:
                        self._loop.call_soon_threadsafe(self._queue.put_nowait, vf)
                    except RuntimeError:
                        pass
            elapsed = time.monotonic() - t0
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        log.info("camera capture stopped")


class NullCameraSource:
    """No camera available: feeds the face pipeline nothing."""

    def start(self, queue, sample_rate: int | None = None) -> None:
        pass

    def stop(self) -> None:
        pass
