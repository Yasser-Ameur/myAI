"""Shared scalar value types that are layer-agnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

# A dense numeric vector. Floats; providers decide dimensionality.
Vector = list[float]

# Confidence in [0, 1].
Confidence = float

# Importance in [0, 1].
Importance = float

# A stable logical name for a model slot, e.g. "llm.default".
ModelSlot = str


@dataclass(frozen=True)
class ValueEstimate:
    """A scalar estimate with uncertainty and provenance."""

    value: float
    confidence: float
    timestamp: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._clamp()

    def _clamp(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            object.__setattr__(self, "value", min(1.0, max(0.0, self.value)))
        if not (0.0 <= self.confidence <= 1.0):
            object.__setattr__(self, "confidence", min(1.0, max(0.0, self.confidence)))

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ValueEstimate":
        return cls(
            value=float(d.get("value", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            timestamp=str(d.get("timestamp", "")),
            evidence=tuple(d.get("evidence", [])),
        )


ScalarOrVector = Union[float, int, bool, str, list, dict, None]
