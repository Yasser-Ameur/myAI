"""Application-layer ports (Protocols) that infrastructure implements.

Application code depends on these interfaces only. Infrastructure classes are
injected by the runtime. A NullCognitiveStore provides graceful degradation when
the database is unavailable.
"""

from __future__ import annotations

from typing import Protocol

from companion.core.types import Vector
from companion.domain.graph import (
    Belief,
    Entity,
    Fact,
    Goal,
    KnowledgeChunk,
    Observation,
    Relation,
    Source,
)
from companion.domain.memory import Episode, Memory
from companion.domain.personality import (
    Contradiction,
    PersonalityEvidence,
    PersonalityProfile,
)
from companion.domain.relationship import Relationship


class GraphStore(Protocol):
    # entities
    def upsert_entity(self, entity: Entity) -> None: ...
    def get_entity(self, entity_id: str) -> Entity | None: ...
    def find_entity_by_name(self, name: str, type: str = "") -> Entity | None: ...
    def list_entities(self, type: str = "", limit: int = 100) -> list[Entity]: ...

    # facts
    def add_fact(self, fact: Fact) -> None: ...
    def get_fact(self, fact_id: str) -> Fact | None: ...
    def list_facts(self, subject_id: str = "", predicate: str = "", include_deleted: bool = False) -> list[Fact]: ...
    def list_facts_about_object(self, object_id: str) -> list[Fact]: ...
    def invalidate_fact(self, fact_id: str, valid_to: str) -> None: ...
    def confirm_fact(self, fact_id: str, at: str) -> None: ...
    def set_fact_confidence(self, fact_id: str, confidence: float) -> None: ...
    def delete_fact(self, fact_id: str) -> None: ...

    # relations
    def add_relation(self, relation: Relation) -> None: ...
    def list_relations(self, entity_id: str) -> list[Relation]: ...

    # episodes
    def save_episode(self, episode: Episode) -> None: ...
    def get_episode(self, episode_id: str) -> Episode | None: ...
    def list_episodes(self, limit: int = 50, consolidated_only: bool = False) -> list[Episode]: ...
    def add_turn(self, turn_id: str, episode_id: str, role: str, text: str, timestamp: str, source: str, user_state: dict) -> None: ...

    # observations
    def add_observation(self, observation: Observation) -> None: ...
    def list_observations(self, kind: str = "", episode_id: str = "", limit: int = 500) -> list[Observation]: ...

    # beliefs
    def upsert_belief(self, belief: Belief) -> None: ...
    def list_beliefs(self, target_type: str = "", status: str = "") -> list[Belief]: ...
    def get_belief(self, belief_id: str) -> Belief | None: ...

    # personality
    def load_profile(self) -> PersonalityProfile: ...
    def save_profile(self, profile: PersonalityProfile) -> None: ...
    def add_personality_evidence(self, evidence: PersonalityEvidence) -> None: ...
    def list_personality_evidence(self, target: str = "", limit: int = 200) -> list[PersonalityEvidence]: ...

    # contradictions
    def add_contradiction(self, contradiction: Contradiction) -> None: ...
    def list_contradictions(self, status: str = "") -> list[Contradiction]: ...
    def resolve_contradiction(self, contradiction_id: str, status: str) -> None: ...

    # goals
    def upsert_goal(self, goal: Goal) -> None: ...
    def list_goals(self, status: str = "active") -> list[Goal]: ...

    # relationships
    def upsert_relationship(self, relationship: Relationship) -> None: ...
    def list_relationships(self) -> list[Relationship]: ...
    def get_relationship_for(self, target_id: str) -> Relationship | None: ...

    # sources / knowledge
    def get_or_create_source(self, source: Source) -> Source: ...
    def add_knowledge(self, chunk: KnowledgeChunk) -> None: ...
    def list_knowledge(self, source_type: str = "", limit: int = 100) -> list[KnowledgeChunk]: ...

    # memories
    def add_memory(self, memory: Memory) -> None: ...
    def get_memory(self, memory_id: str) -> Memory | None: ...
    def update_memory_status(self, memory_id: str, status: str) -> None: ...
    def update_memory_access(self, memory_id: str, at: str) -> None: ...
    def set_memory_locked(self, memory_id: str, locked: bool) -> None: ...
    def forget_memory(self, memory_id: str) -> None: ...
    def list_memories(self, status: str = "", type: str = "", limit: int = 200) -> list[Memory]: ...

    # system state
    def get_system_state(self, key: str) -> str | None: ...
    def set_system_state(self, key: str, value: str) -> None: ...


class VectorStorePort(Protocol):
    def upsert(self, vector_id: str, model_id: str, vector: Vector, owner_type: str, owner_id: str) -> None: ...
    def remove(self, vector_id: str) -> None: ...
    def search(self, model_id: str, query: Vector, top_k: int) -> list: ...
    def clear_namespace(self, model_id: str) -> None: ...
    def count(self, model_id: str) -> int: ...


class NullGraphStore:
    """Degraded stateless store used when the database is unavailable.

    The conversation continues; nothing is persisted until the store recovers.
    List/query methods return empty collections so callers degrade gracefully.
    """

    def __init__(self, reason: str = "degraded") -> None:
        self.reason = reason

    def __getattr__(self, name: str):
        def _noop(*args, **kwargs):
            return None

        return _noop

    def list_episodes(self, limit: int = 50, consolidated_only: bool = False) -> list[Episode]:
        return []

    def list_memories(self, status: str = "", type: str = "", limit: int = 200) -> list[Memory]:
        return []

    def list_facts(self, subject_id: str = "", predicate: str = "", include_deleted: bool = False) -> list[Fact]:
        return []

    def list_facts_about_object(self, object_id: str) -> list[Fact]:
        return []

    def list_relations(self, entity_id: str) -> list[Relation]:
        return []

    def list_entities(self, type: str = "", limit: int = 100) -> list[Entity]:
        return []

    def list_beliefs(self, target_type: str = "", status: str = "") -> list[Belief]:
        return []

    def list_personality_evidence(self, target: str = "", limit: int = 200) -> list[PersonalityEvidence]:
        return []

    def list_contradictions(self, status: str = "") -> list[Contradiction]:
        return []

    def list_goals(self, status: str = "active") -> list[Goal]:
        return []

    def list_relationships(self) -> list[Relationship]:
        return []

    def list_knowledge(self, source_type: str = "", limit: int = 100) -> list[KnowledgeChunk]:
        return []

    def load_profile(self) -> PersonalityProfile:
        return PersonalityProfile()

    def get_entity(self, entity_id: str) -> Entity | None:
        return None

    def find_entity_by_name(self, name: str, type: str = "") -> Entity | None:
        return None

    def get_fact(self, fact_id: str) -> Fact | None:
        return None

    def get_episode(self, episode_id: str) -> Episode | None:
        return None

    def get_memory(self, memory_id: str) -> Memory | None:
        return None

    def get_system_state(self, key: str) -> str | None:
        return None

    def set_system_state(self, key: str, value: str) -> None:
        return None
