"""Relationship domain model.

Relationships are first-class objects (not plain graph edges) because they carry
their own longitudinal state: trust, familiarity, valence, interaction history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from companion.core.ids import new_relationship_id


@dataclass
class Relationship:
    id: str = field(default_factory=new_relationship_id)
    subject_id: str = ""      # user entity id
    target_id: str = ""       # the other person's entity id
    type: str = "person"      # person | group | organization
    name: str = ""
    trust: float = 0.5
    familiarity: float = 0.0
    emotional_valence: float = 0.0  # -1 .. +1
    interaction_count: int = 0
    last_interaction: str = ""
    important_events: list[str] = field(default_factory=list)
    confidence: float = 0.3
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "target_id": self.target_id,
            "type": self.type,
            "name": self.name,
            "trust": self.trust,
            "familiarity": self.familiarity,
            "emotional_valence": self.emotional_valence,
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction,
            "important_events": self.important_events,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relationship":
        return cls(
            id=str(d.get("id", new_relationship_id())),
            subject_id=str(d.get("subject_id", "")),
            target_id=str(d.get("target_id", "")),
            type=str(d.get("type", "person")),
            name=str(d.get("name", "")),
            trust=float(d.get("trust", 0.5)),
            familiarity=float(d.get("familiarity", 0.0)),
            emotional_valence=float(d.get("emotional_valence", 0.0)),
            interaction_count=int(d.get("interaction_count", 0)),
            last_interaction=str(d.get("last_interaction", "")),
            important_events=list(d.get("important_events", [])),
            confidence=float(d.get("confidence", 0.3)),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            notes=str(d.get("notes", "")),
        )
