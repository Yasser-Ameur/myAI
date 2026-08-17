"""Personality domain model.

Personality is a graph-backed probabilistic model, not a prose paragraph.

Two personalities exist:
  - User personality: learned from PersonalityEvidence with conservative updates.
  - Agent personality: configured identity/values plus mutable internal state.

The fast-moving parts (mood, energy) are explicitly separated from the stable
parts (traits, values). FAST state never overwrites STABLE structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from companion.core.ids import new_evidence_id
from companion.core.types import ValueEstimate


class StabilityClass(str, Enum):
    VERY_STABLE = "very_stable"  # core values, deep traits
    MEDIUM = "medium"            # preferences, habits, communication style, goals
    FAST = "fast"                # mood, energy, attention, frustration, interest


@dataclass
class Trait:
    name: str
    value: float = 0.5
    confidence: float = 0.0
    stability: float = 0.5      # 0..1 how resistant to change
    evidence_count: int = 0
    stability_class: StabilityClass = StabilityClass.VERY_STABLE
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "stability": self.stability,
            "evidence_count": self.evidence_count,
            "stability_class": self.stability_class.value,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trait":
        return cls(
            name=str(d["name"]),
            value=float(d.get("value", 0.5)),
            confidence=float(d.get("confidence", 0.0)),
            stability=float(d.get("stability", 0.5)),
            evidence_count=int(d.get("evidence_count", 0)),
            stability_class=StabilityClass(d.get("stability_class", StabilityClass.VERY_STABLE.value)),
            updated_at=str(d.get("updated_at", "")),
        )


@dataclass
class Value:
    name: str
    importance: float = 0.5
    confidence: float = 0.0
    stability: float = 0.9
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "importance": self.importance,
            "confidence": self.confidence,
            "stability": self.stability,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Value":
        return cls(
            name=str(d["name"]),
            importance=float(d.get("importance", 0.5)),
            confidence=float(d.get("confidence", 0.0)),
            stability=float(d.get("stability", 0.9)),
            updated_at=str(d.get("updated_at", "")),
        )


@dataclass
class Preference:
    name: str
    value: float = 0.5
    confidence: float = 0.0
    stability: float = 0.4
    evidence_count: int = 0
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "stability": self.stability,
            "evidence_count": self.evidence_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Preference":
        return cls(
            name=str(d["name"]),
            value=float(d.get("value", 0.5)),
            confidence=float(d.get("confidence", 0.0)),
            stability=float(d.get("stability", 0.4)),
            evidence_count=int(d.get("evidence_count", 0)),
            updated_at=str(d.get("updated_at", "")),
        )


# Extensible ontology of communication preferences learned from interaction.
COMMUNICATION_PREFERENCES = (
    "preferred_response_length",
    "preferred_tone",
    "preferred_humor",
    "preferred_directness",
    "preferred_technical_depth",
    "preferred_question_frequency",
    "preferred_voice_speed",
    "preferred_avatar_expressiveness",
    "preferred_interruption_style",
)

CORE_VALUES = (
    "autonomy",
    "achievement",
    "curiosity",
    "creativity",
    "security",
    "belonging",
    "mastery",
    "status",
    "truth",
    "novelty",
)

AGENT_PERSONALITY_DIMENSIONS = (
    "warmth",
    "curiosity",
    "humor",
    "confidence",
    "patience",
    "playfulness",
    "seriousness",
    "assertiveness",
    "empathy",
)


@dataclass
class PersonalityEvidence:
    """One observation that may affect a trait/value/preference."""

    id: str = field(default_factory=new_evidence_id)
    target: str = ""                # trait_or_value name
    direction: str = "positive"     # positive | negative | neutral
    strength: float = 0.3
    confidence: float = 0.3
    source_episode: str = ""
    source: str = ""                # user | conversation | document | model_inference
    timestamp: str = ""
    context: str = ""
    kind: str = "statement"         # statement | behavior | preference | fact | inference

    def signed_strength(self) -> float:
        sign = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}.get(self.direction, 0.0)
        return sign * self.strength

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target": self.target,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "source_episode": self.source_episode,
            "source": self.source,
            "timestamp": self.timestamp,
            "context": self.context,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PersonalityEvidence":
        return cls(
            id=str(d.get("id", new_evidence_id())),
            target=str(d.get("target", "")),
            direction=str(d.get("direction", "positive")),
            strength=float(d.get("strength", 0.3)),
            confidence=float(d.get("confidence", 0.3)),
            source_episode=str(d.get("source_episode", "")),
            source=str(d.get("source", "conversation")),
            timestamp=str(d.get("timestamp", "")),
            context=str(d.get("context", "")),
            kind=str(d.get("kind", "statement")),
        )


@dataclass
class Contradiction:
    """Two statements that conflict. Neither is deleted; later evidence resolves."""

    id: str
    statement_a: str
    statement_b: str
    subject: str = ""
    predicate: str = ""
    contexts: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    resolution_status: str = "unresolved"  # unresolved | preference_changed | context_explains | emotional | duplicate

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement_a": self.statement_a,
            "statement_b": self.statement_b,
            "subject": self.subject,
            "predicate": self.predicate,
            "contexts": self.contexts,
            "timestamps": self.timestamps,
            "resolution_status": self.resolution_status,
        }


@dataclass
class PersonalityProfile:
    """Probabilistic, evidence-backed personality representation."""

    traits: dict[str, Trait] = field(default_factory=dict)
    values: dict[str, Value] = field(default_factory=dict)
    preferences: dict[str, Preference] = field(default_factory=dict)
    motivations: dict[str, ValueEstimate] = field(default_factory=dict)
    behavioral_patterns: dict[str, ValueEstimate] = field(default_factory=dict)
    communication_style: dict[str, ValueEstimate] = field(default_factory=dict)
    goals: dict[str, "Goal"] = field(default_factory=dict)
    current_state: dict[str, ValueEstimate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The agent always has a full personality scaffold; dimensions start
        # neutral (0.5) with zero confidence until evidence arrives.
        if not self.traits:
            self.traits = {
                name: Trait(name=name, value=0.5, confidence=0.0,
                            stability=0.6, stability_class=StabilityClass.VERY_STABLE)
                for name in AGENT_PERSONALITY_DIMENSIONS
            }
        if not self.values:
            self.values = {
                name: Value(name=name, importance=0.5, confidence=0.0, stability=0.9)
                for name in CORE_VALUES
            }
        if not self.preferences:
            self.preferences = {
                name: Preference(name=name, value=0.5, confidence=0.0, stability=0.4)
                for name in COMMUNICATION_PREFERENCES
            }

    def to_dict(self) -> dict:
        return {
            "traits": {k: v.to_dict() for k, v in self.traits.items()},
            "values": {k: v.to_dict() for k, v in self.values.items()},
            "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
            "motivations": {k: v.to_dict() for k, v in self.motivations.items()},
            "behavioral_patterns": {k: v.to_dict() for k, v in self.behavioral_patterns.items()},
            "communication_style": {k: v.to_dict() for k, v in self.communication_style.items()},
            "goals": {k: v.to_dict() for k, v in self.goals.items()},
            "current_state": {k: v.to_dict() for k, v in self.current_state.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PersonalityProfile":
        return cls(
            traits={k: Trait.from_dict(v) for k, v in d.get("traits", {}).items()},
            values={k: Value.from_dict(v) for k, v in d.get("values", {}).items()},
            preferences={k: Preference.from_dict(v) for k, v in d.get("preferences", {}).items()},
            motivations={k: ValueEstimate.from_dict(v) for k, v in d.get("motivations", {}).items()},
            behavioral_patterns={k: ValueEstimate.from_dict(v) for k, v in d.get("behavioral_patterns", {}).items()},
            communication_style={k: ValueEstimate.from_dict(v) for k, v in d.get("communication_style", {}).items()},
            goals={k: Goal.from_dict(v) for k, v in d.get("goals", {}).items()},
            current_state={k: ValueEstimate.from_dict(v) for k, v in d.get("current_state", {}).items()},
        )


from companion.domain.graph import Goal  # noqa: E402  (import at end to avoid cycle)
