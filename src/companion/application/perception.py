"""Perception application layer.

Perception produces *measurements*, never interpretations:
  - TemporalObservationBuffer: sliding-window statistics over noisy observations.
  - AcousticExtractor: cheap acoustic features from speech frames.
  - UserStateEstimator: combines face + acoustic + text evidence into UserState
    estimates (inferences with confidence, not facts).
  - PerceptionService: drives camera/mic inputs through the providers and the
    event bus.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

from companion.core.clock import Clock, SystemClock
from companion.core.contracts import AudioInput, VideoFrame
from companion.core.events import (
    EVENT_AUDIO_ENDED,
    EVENT_AUDIO_STARTED,
    EVENT_EXPRESSION_OBSERVATION_UPDATED,
    EVENT_FACE_DETECTED,
    EVENT_FACE_LOST,
    EVENT_FACE_OBSERVATION_UPDATED,
    EVENT_GAZE_UPDATED,
    EVENT_HEAD_POSE_UPDATED,
    EVENT_TRANSCRIPT_FINAL,
    EVENT_TRANSCRIPT_PARTIAL,
    EVENT_USER_STATE_UPDATED,
    EventBus,
)
from companion.core.types import ValueEstimate
from companion.domain.state import UserState

log = logging.getLogger(__name__)

DEFAULT_WINDOWS = (0.25, 1.0, 5.0, 30.0)


@dataclass
class WindowStats:
    window: float
    mean: float = 0.0
    variance: float = 0.0
    velocity: float = 0.0    # per-second change
    peak: float = 0.0
    duration: float = 0.0    # seconds above threshold
    frequency: float = 0.0   # activations per minute
    trend: float = 0.0       # -1 declining .. +1 rising

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "mean": self.mean,
            "variance": self.variance,
            "velocity": self.velocity,
            "peak": self.peak,
            "duration": self.duration,
            "frequency": self.frequency,
            "trend": self.trend,
        }


class TemporalObservationBuffer:
    """Sliding-window statistics over scalar observations per metric.

    Metric = e.g. 'smile', 'head_yaw', 'attention'. Each pushed sample keeps its
    timestamp; statistics are computed on demand within the requested windows.
    """

    def __init__(self, clock: Clock | None = None, windows: tuple[float, ...] = DEFAULT_WINDOWS) -> None:
        self._clock = clock or SystemClock()
        self.windows = windows
        self._samples: dict[str, deque[tuple[float, float]]] = {}
        self._threshold: dict[str, float] = {}

    def set_threshold(self, metric: str, threshold: float) -> None:
        self._threshold[metric] = threshold

    def push(self, metric: str, value: float, timestamp: float | None = None) -> None:
        t = timestamp if timestamp is not None else self._clock.monotonic()
        self._samples.setdefault(metric, deque()).append((t, value))
        max_window = max(self.windows)
        q = self._samples[metric]
        while q and t - q[0][0] > max_window + 1.0:
            q.popleft()

    def push_face(self, face_dict: dict) -> None:
        """Push a FaceObservation dict into metric buffers."""
        now = self._clock.monotonic()
        bs = face_dict.get("blendshapes", {})
        smile = 0.5 * (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0))
        brow = bs.get("browInnerUp", 0.0)
        self.push("smile", smile, now)
        self.push("brow_inner_up", brow, now)
        hp = face_dict.get("head_pose", {})
        self.push("head_pitch", hp.get("pitch", 0.0), now)
        self.push("head_yaw", hp.get("yaw", 0.0), now)
        self.push("attention", face_dict.get("gaze", {}).get("estimated_attention", 0.5), now)
        self.push("face_present", 1.0 if face_dict.get("detected", False) else 0.0, now)

    def stats(self, metric: str, window: float) -> WindowStats:
        q = self._samples.get(metric)
        if not q:
            return WindowStats(window=window)
        now = self._clock.monotonic()
        cutoff = now - window
        pts = [(t, v) for t, v in q if t >= cutoff]
        if not pts:
            return WindowStats(window=window)
        values = [v for _, v in pts]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        peak = max(values)
        # velocity: total change over elapsed time
        elapsed = max(1e-6, pts[-1][0] - pts[0][0])
        total_delta = sum(abs(pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
        velocity = total_delta / elapsed
        # trend: linear regression slope normalized
        slope = _lin_slope(pts)
        trend = max(-1.0, min(1.0, slope * window * 2.0))
        # duration above threshold + activation frequency
        thr = self._threshold.get(metric, 0.3)
        active = 0.0
        activations = 0
        in_run = False
        for i, (t, v) in enumerate(pts):
            if v >= thr:
                active += (pts[i + 1][0] - t) if i + 1 < len(pts) else (now - t)
                if not in_run:
                    activations += 1
                    in_run = True
            else:
                in_run = False
        minutes = max(1e-6, (now - pts[0][0]) / 60.0)
        return WindowStats(
            window=window,
            mean=mean,
            variance=var,
            velocity=velocity,
            peak=peak,
            duration=active,
            frequency=activations / minutes,
            trend=trend,
        )

    def all_stats(self, metric: str) -> dict[float, WindowStats]:
        return {w: self.stats(metric, w) for w in self.windows}

    def clear(self, metric: str | None = None) -> None:
        if metric:
            self._samples.pop(metric, None)
        else:
            self._samples.clear()


def _lin_slope(pts: list[tuple[float, float]]) -> float:
    n = len(pts)
    if n < 2:
        return 0.0
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    xm = sum(x) / n
    ym = sum(y) / n
    num = sum((x[i] - xm) * (y[i] - ym) for i in range(n))
    den = sum((x[i] - xm) ** 2 for i in range(n))
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# Acoustic features
# ---------------------------------------------------------------------------

@dataclass
class AcousticFeatures:
    speaking_rate: float = 0.0     # syllables/sec approximation
    pause_frequency: float = 0.0   # pauses per minute
    volume: float = 0.0            # normalized RMS 0..1
    pitch_mean: float = 0.0        # Hz
    pitch_variance: float = 0.0
    energy: float = 0.0            # normalized energy

    def to_dict(self) -> dict:
        return {
            "speaking_rate": self.speaking_rate,
            "pause_frequency": self.pause_frequency,
            "volume": self.volume,
            "pitch_mean": self.pitch_mean,
            "pitch_variance": self.pitch_variance,
            "energy": self.energy,
        }


class AcousticExtractor:
    """Cheap acoustic feature extraction from PCM frames (dependency-free)."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._frame = 400  # 25ms at 16k

    def features(self, samples: bytes, spoken_words: int = 0, speech_seconds: float = 0.0) -> AcousticFeatures:
        import array

        if not samples:
            return AcousticFeatures()
        arr = array.array("h", samples)
        if len(arr) < self._frame:
            return AcousticFeatures()
        n = len(arr)
        rms = math.sqrt(sum(v * v for v in arr) / n) / 32768.0
        energy = rms * rms
        # pitch via autocorrelation on a 25ms window
        pitches: list[float] = []
        frame = arr[: self._frame]
        pitch = _autocorrelation_pitch(frame, self.sample_rate)
        if pitch:
            pitches.append(pitch)
        # coarse pauses: runs of frames with RMS below 1% of max
        max_rms = max((sum(arr[i:i + self._frame][j] ** 2 for j in range(0, self._frame)) / self._frame) ** 0.5
                      for i in range(0, n - self._frame, self._frame)) or 1.0
        pause_count = 0
        for i in range(0, n - self._frame, self._frame):
            seg = arr[i:i + self._frame]
            seg_rms = math.sqrt(sum(v * v for v in seg) / self._frame) / 32768.0
            if seg_rms < 0.01 * max_rms / 32768.0 + 1e-5:
                pause_count += 1
        total_frames = max(1, (n - self._frame) // self._frame)
        pmean = sum(pitches) / len(pitches) if pitches else 0.0
        pvar = sum((p - pmean) ** 2 for p in pitches) / len(pitches) if pitches else 0.0
        return AcousticFeatures(
            speaking_rate=spoken_words / speech_seconds if speech_seconds else 0.0,
            pause_frequency=pause_count / total_frames * (60.0 * self.sample_rate / self._frame),
            volume=min(1.0, rms * 4.0),
            pitch_mean=pmean,
            pitch_variance=pvar,
            energy=min(1.0, energy * 16.0),
        )


def _autocorrelation_pitch(frame, sample_rate: int) -> float:
    n = len(frame)
    if n < 2:
        return 0.0
    center = frame
    best_lag = 0
    best_val = 0.0
    for lag in range(int(sample_rate / 500), min(n // 2, int(sample_rate / 50))):
        s = 0.0
        a = 0.0
        for i in range(n - lag):
            s += center[i] * center[i + lag]
            a += center[i] * center[i]
        if a == 0:
            continue
        val = s / a
        if val > best_val:
            best_val = val
            best_lag = lag
    if best_val < 0.3 or best_lag == 0:
        return 0.0
    return sample_rate / best_lag


# ---------------------------------------------------------------------------
# User state estimator
# ---------------------------------------------------------------------------

CONFUSION_TEXT = ("confused", "what do you mean", "i don't follow", "not sure", "hmm", "unsure", "wait")
FRUSTRATION_TEXT = ("ugh", "annoying", "why won't", "this is hard", "again", "stupid", "frustrat")
INTEREST_TEXT = ("interesting", "tell me more", "cool", "wow", "fascinating", "love that", "nice")


class UserStateEstimator:
    """Fuses face + acoustic + text evidence into conservative UserState.

    The mappings are deliberately cautious: one cue moves a dimension only a
    little, and confidence reflects how many independent cues agree.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock or SystemClock()
        self._buffer = TemporalObservationBuffer(clock=self._clock)
        self.last_state = UserState(timestamp=self._clock.now_iso())

    def ingest_face(self, face_dict: dict) -> None:
        self._buffer.push_face(face_dict)

    def ingest_transcript(self, text: str) -> None:
        self._buffer.push("speech_present", 1.0 if text.strip() else 0.0)
        self._recent_text = text

    def ingest_acoustic(self, features: AcousticFeatures) -> None:
        self._buffer.push("volume", features.volume)
        self._buffer.push("energy", features.energy)
        self._buffer.push("speaking_rate", min(1.0, features.speaking_rate / 6.0))
        self._buffer.push("pitch", min(1.0, features.pitch_mean / 400.0))

    def estimate(self) -> UserState:
        st = UserState(timestamp=self._clock.now_iso())
        b = self._buffer
        s1 = b.stats("attention", 5.0)
        s5 = b.stats("smile", 5.0)
        brow = b.stats("brow_inner_up", 5.0)
        tilt = b.stats("head_yaw", 5.0)
        energy = b.stats("energy", 5.0)
        vol = b.stats("volume", 5.0)
        pitch = b.stats("pitch", 5.0)
        rate = b.stats("speaking_rate", 5.0)
        face = b.stats("face_present", 30.0)
        speech = b.stats("speech_present", 30.0)
        text = getattr(self, "_recent_text", "").lower()

        face_w = min(1.0, face.mean * 3.0)   # how long a face has been present
        speech_w = min(1.0, speech.mean * 3.0)
        has_text = bool(text.strip())

        # --- attention / engagement (face-led) ---
        att_evidence = []
        if face.mean > 0.2:
            att_evidence.append("face_present")
        if s1.mean > 0.4:
            att_evidence.append("gaze_on_camera")
        att = s1.mean if s1.mean else 0.5
        st.set("attention", ValueEstimate(att, min(1.0, 0.35 + 0.4 * face_w + 0.1 * speech_w),
                                          self._clock.now_iso(), tuple(att_evidence)))
        eng = 0.55 * att + 0.3 * min(1.0, s5.mean * 1.5) + 0.15 * speech_w
        st.set("engagement", ValueEstimate(eng, 0.35 + 0.25 * face_w + 0.2 * speech_w,
                                           self._clock.now_iso(),
                                           ("attention", "smile", "speech") if has_text else ("attention", "smile")))

        # --- valence (smile-led, cautious) ---
        val_evidence = []
        if s5.mean > 0.35:
            val_evidence.append("smile_active")
        if pitch.mean > 0.3 and pitch.trend > 0.2:
            val_evidence.append("rising_pitch")
        valence = 0.7 * (2 * s5.mean - 1) + 0.3 * (2 * min(1.0, pitch.mean) - 1)
        if not val_evidence:
            valence = max(-0.2, min(0.2, valence))  # neutral by default
        st.set("valence", ValueEstimate(max(-1.0, min(1.0, valence)), 0.3 + 0.25 * len(val_evidence),
                                        self._clock.now_iso(), tuple(val_evidence)))

        # --- arousal (acoustic-led) ---
        aro_evidence = []
        if vol.mean > 0.4:
            aro_evidence.append("loud_voice")
        if energy.mean > 0.3:
            aro_evidence.append("high_energy")
        if pitch.variance > 0.15:
            aro_evidence.append("pitch_variation")
        arousal = 0.5 * min(1.0, vol.mean * 2.0) + 0.3 * min(1.0, energy.mean * 2.0) + 0.2 * min(1.0, pitch.variance * 3.0)
        st.set("arousal", ValueEstimate(min(1.0, arousal), 0.35 + 0.2 * len(aro_evidence),
                                        self._clock.now_iso(), tuple(aro_evidence)))

        # --- confusion (face + text) ---
        conf_evidence = []
        if brow.mean > 0.4:
            conf_evidence.append("brow_inner_up")
        if abs(tilt.mean) > 0.25:
            conf_evidence.append("head_tilt")
        if rate.mean > 0.5 and speech.mean > 0:
            conf_evidence.append("speech_pause")
        if any(k in text for k in CONFUSION_TEXT):
            conf_evidence.append("confusion_words")
        confusion = 0.4 * max(0.0, brow.mean - 0.3) * 3 + 0.3 * min(1.0, abs(tilt.mean) * 2) + 0.3 * min(1.0, rate.mean)
        if "confusion_words" in conf_evidence:
            confusion = max(confusion, 0.5)
        st.set("confusion", ValueEstimate(min(1.0, confusion), 0.3 + 0.2 * len(conf_evidence),
                                          self._clock.now_iso(), tuple(conf_evidence)))

        # --- frustration (text-led) ---
        frust_evidence = []
        if any(k in text for k in FRUSTRATION_TEXT):
            frust_evidence.append("frustration_words")
        if vol.mean > 0.6 and energy.mean > 0.5:
            frust_evidence.append("loud_high_energy")
        frustration = 0.5 if "frustration_words" in frust_evidence else 0.0
        if "loud_high_energy" in frust_evidence:
            frustration = max(frustration, 0.3)
        st.set("frustration", ValueEstimate(frustration, 0.3 + 0.25 * len(frust_evidence),
                                            self._clock.now_iso(), tuple(frust_evidence)))

        # --- interest (text-led) ---
        interest_evidence = []
        if any(k in text for k in INTEREST_TEXT):
            interest_evidence.append("interest_words")
        if s5.mean > 0.25 and s5.trend > 0.15:
            interest_evidence.append("rising_smile")
        if s1.mean > 0.5:
            interest_evidence.append("engaged_gaze")
        interest = 0.5 if "interest_words" in interest_evidence else 0.3
        if "engaged_gaze" in interest_evidence:
            interest += 0.2
        st.set("interest", ValueEstimate(min(1.0, interest), 0.3 + 0.2 * len(interest_evidence),
                                         self._clock.now_iso(), tuple(interest_evidence)))

        # --- energy / confidence (weak priors) ---
        st.set("energy", ValueEstimate(min(1.0, energy.mean), 0.4, self._clock.now_iso(), tuple(aro_evidence)))
        conf = 0.5 + 0.3 * att - 0.4 * confusion
        st.set("confidence", ValueEstimate(max(0.0, min(1.0, conf)), 0.4, self._clock.now_iso(),
                                           tuple(conf_evidence)))
        st.set("social_engagement", ValueEstimate(eng, 0.5, self._clock.now_iso(),
                                                  tuple(att_evidence)))

        self.last_state = st
        return st


# ---------------------------------------------------------------------------
# Perception service
# ---------------------------------------------------------------------------

@dataclass
class SpeechSession:
    chunks: bytearray = field(default_factory=bytearray)
    started_at: float = 0.0
    partial_text: str = ""


class PerceptionService:
    """Wires camera/mic providers to the event bus with backpressure."""

    def __init__(
        self,
        face_provider=None,
        vad_provider=None,
        stt_provider=None,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        face_fps: int = 10,
        sample_rate: int = 16000,
        transcript_partial_threshold: int = 20,
    ) -> None:
        self.face = face_provider
        self.vad = vad_provider
        self.stt = stt_provider
        self.bus = bus or EventBus(clock=clock)
        self.clock = clock or SystemClock()
        self.face_fps = face_fps
        self.sample_rate = sample_rate
        self.transcript_partial_threshold = transcript_partial_threshold
        self._camera_queue = None
        self.buffer = TemporalObservationBuffer(clock=self.clock)
        self.state_estimator = UserStateEstimator(clock=self.clock)
        self.acoustic = AcousticExtractor(sample_rate=sample_rate)
        self._speech: SpeechSession | None = None
        self._frame_counter = 0
        self._last_face_present: bool | None = None

    # -- face pipeline ---------------------------------------------------

    async def process_face_frame(self, frame: VideoFrame) -> None:
        if self.face is None:
            return
        obs = await self.face.analyze(frame)
        d = obs.to_dict()
        self.buffer.push_face(d)
        self.state_estimator.ingest_face(d)
        if obs.detected and self._last_face_present is not True:
            self.bus.publish(EVENT_FACE_DETECTED, d)
            self._last_face_present = True
        elif not obs.detected and self._last_face_present is not False:
            self.bus.publish(EVENT_FACE_LOST, {"timestamp": self.clock.now_iso()})
            self._last_face_present = False
        if obs.detected:
            self.bus.publish(EVENT_FACE_OBSERVATION_UPDATED, d)
            self.bus.publish(EVENT_EXPRESSION_OBSERVATION_UPDATED, d)
            self.bus.publish(EVENT_GAZE_UPDATED, d)
            self.bus.publish(EVENT_HEAD_POSE_UPDATED, d)
        self._frame_counter += 1

    async def face_tick(self) -> None:
        """Consume frames from a bounded camera queue, dropping stale ones."""
        if self.face is None:
            return
        frame_interval = 1.0 / max(1, self.face_fps)
        now = self.clock.monotonic()
        last = getattr(self, "_last_face_time", 0.0)
        if now - last < frame_interval:
            return
        self._last_face_time = now
        frame = await self._camera_queue.get() if self._camera_queue else None
        if frame is not None:
            await self.process_face_frame(frame)

    def attach_camera(self, queue, provider=None) -> None:
        self._camera_queue = queue
        if provider is not None:
            self.face = provider

    # -- audio pipeline --------------------------------------------------

    async def process_audio_chunk(self, chunk: AudioInput) -> None:
        if self.vad is not None:
            is_speech = await self.vad.is_speech(chunk)
        else:
            is_speech = bool(chunk.samples)
        if is_speech:
            if self._speech is None:
                self._speech = SpeechSession(started_at=self.clock.monotonic())
                self.bus.publish(EVENT_AUDIO_STARTED, {"timestamp": self.clock.now_iso()})
            self._speech.chunks.extend(chunk.samples)
            if len(self._speech.chunks) >= self.transcript_partial_threshold * 2 * self.sample_rate / 10 and self.stt is not None:
                await self._transcribe_current(partial=True)
        else:
            if self._speech is not None:
                await self._finish_speech()

    async def _transcribe_current(self, partial: bool = False) -> None:
        if self.stt is None or self._speech is None:
            return
        audio = AudioInput(samples=bytes(self._speech.chunks), sample_rate=self.sample_rate, source="mic")
        try:
            tr = await self.stt.transcribe(audio)
        except Exception as exc:
            log.warning("STT failed (%s)", exc)
            return
        if partial:
            if tr.text and tr.text != self._speech.partial_text:
                self._speech.partial_text = tr.text
                self.bus.publish(EVENT_TRANSCRIPT_PARTIAL, {"text": tr.text})
        else:
            self.bus.publish(EVENT_TRANSCRIPT_FINAL, {"text": tr.text, "language": tr.language})
            self.state_estimator.ingest_transcript(tr.text)
            self._update_state_event()

    async def _finish_speech(self) -> None:
        if self._speech is None:
            return
        self.bus.publish(EVENT_AUDIO_ENDED, {"timestamp": self.clock.now_iso()})
        await self._transcribe_current(partial=False)
        self._speech = None

    def _update_state_event(self) -> None:
        st = self.state_estimator.estimate()
        self.bus.publish(EVENT_USER_STATE_UPDATED, st.to_dict())

    async def estimate_acoustic(self, chunk: AudioInput, spoken_words: int = 0, speech_seconds: float = 0.0) -> None:
        feats = self.acoustic.features(chunk.samples, spoken_words, speech_seconds)
        self.state_estimator.ingest_acoustic(feats)

    def current_user_state(self) -> UserState:
        return self.state_estimator.estimate()
