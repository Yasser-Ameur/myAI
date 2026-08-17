"""Relationship engine: maintains Relationship objects over time.

Relationships are first-class objects (trust, familiarity, valence,
interaction history), not plain edges. Updated gradually from episodes.
"""

from __future__ import annotations

import logging

from companion.application.ports import GraphStore
from companion.core.clock import Clock, SystemClock
from companion.core.ids import new_relationship_id
from companion.domain.graph import Entity
from companion.domain.relationship import Relationship

log = logging.getLogger(__name__)


class RelationshipEngine:
    def __init__(self, graph: GraphStore, clock: Clock | None = None) -> None:
        self._graph = graph
        self._clock = clock or SystemClock()

    def ensure_person(self, name: str) -> Entity:
        entity = self._graph.find_entity_by_name(name, type="person")
        if entity is None:
            entity = Entity(
                type="person",
                name=name,
                importance=0.5,
                created_at=self._clock.now_iso(),
                updated_at=self._clock.now_iso(),
            )
            self._graph.upsert_entity(entity)
        return entity

    def note_interaction(self, person_name: str, episode_id: str = "",
                         valence_delta: float = 0.0, trust_delta: float = 0.0,
                         event_note: str = "") -> Relationship:
        person = self.ensure_person(person_name)
        rel = self._graph.get_relationship_for(person.id)
        if rel is None:
            user_id = self._graph.get_system_state("primary_user_entity") or ""
            rel = Relationship(
                id=new_relationship_id(),
                subject_id=user_id,
                target_id=person.id,
                name=person_name,
                created_at=self._clock.now_iso(),
                updated_at=self._clock.now_iso(),
            )
        rel.interaction_count += 1
        rel.familiarity = min(1.0, rel.familiarity + 0.04)
        rel.trust = max(0.0, min(1.0, rel.trust + trust_delta))
        rel.emotional_valence = max(-1.0, min(1.0, rel.emotional_valence + valence_delta))
        rel.last_interaction = self._clock.now_iso()
        rel.confidence = min(1.0, rel.confidence + 0.05)
        if event_note and len(rel.important_events) < 20:
            rel.important_events.append(event_note)
        rel.updated_at = self._clock.now_iso()
        self._graph.upsert_relationship(rel)
        return rel

    def snapshot(self, limit: int = 3) -> list[dict]:
        rels = self._graph.list_relationships()[:limit]
        return [r.to_dict() for r in rels]
