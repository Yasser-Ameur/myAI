"""User state domain model.

Estimates about the user's current contextual state. These are INFERENCES with
confidence and evidence — never irreversible facts. Facial appearance alone can
never produce a claim like "user is sad".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from companion.core.types import ValueEstimate

# Dimensions of the user state model (spec section 9).
USER_STATE_DIMENSIONS = (
    "engagement",
    "attention",
    "energy",
    "arousal",
    "valence",
    "frustration",
    "confusion",
    "confidence",
    "interest",
    "social_engagement",
)

# Optional acoustic evidence dimensions (spec section 10).
ACOUSTIC_DIMENSIONS = (
    "speaking_rate",
    "pause_frequency",
    "volume",
    "pitch_mean",
    "pitch_variance",
    "energy",
)


@dataclass
class UserState:
    """A dynamic snapshot of estimated user state."""

    dimensions: dict[str, ValueEstimate] = field(default_factory=dict)
    timestamp: str = ""
    modality_used: list[str] = field(default_factory=list)

    def get(self, name: str) -> ValueEstimate | None:
        return self.dimensions.get(name)

    def set(self, name: str, estimate: ValueEstimate) -> None:
        self.dimensions[name] = estimate

    def with_modality(self, modality: str) -> "UserState":
        self.modality_used.append(modality)
        return self

    def top_evidence(self, name: str, limit: int = 4) -> list[str]:
        est = self.dimensions.get(name)
        if est is None:
            return []
        return list(est.evidence)[:limit]

    def to_dict(self) -> dict:
        return {
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "timestamp": self.timestamp,
            "modality_used": self.modality_used,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserState":
        return cls(
            dimensions={
                str(k): ValueEstimate.from_dict(v)
                for k, v in d.get("dimensions", {}).items()
            },
            timestamp=str(d.get("timestamp", "")),
            modality_used=list(d.get("modality_used", [])),
        )
