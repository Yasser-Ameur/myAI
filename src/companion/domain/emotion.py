"""Emotion and affect domain model.

Affect is an *estimate* with uncertainty, never a fact. The avatar, TTS and
conversation planner consume an AffectVector; perception produces evidence that
a state estimator turns into an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AffectVector:
    """Continuous valence-arousal representation plus discrete blend.

    Dimensions are in [-1, 1] unless documented otherwise.
    """

    valence: float = 0.0    # -1 negative .. +1 positive
    arousal: float = 0.0    # -1 calm .. +1 intense
    dominance: float = 0.0  # -1 submissive .. +1 dominant

    # Discrete contributions (0..1). Kept as hints, not canonical.
    blend: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "blend": dict(self.blend),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AffectVector":
        return cls(
            valence=float(d.get("valence", 0.0)),
            arousal=float(d.get("arousal", 0.0)),
            dominance=float(d.get("dominance", 0.0)),
            blend={str(k): float(v) for k, v in d.get("blend", {}).items()},
        )


@dataclass(frozen=True)
class EmotionEstimate:
    """A contextual emotional/affective estimate with uncertainty."""

    label: str  # machine label, e.g. "engagement", "frustration", "neutral"
    value: float
    confidence: float
    evidence: tuple[str, ...] = ()
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "timestamp": self.timestamp,
        }
