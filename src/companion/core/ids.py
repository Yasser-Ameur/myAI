"""Typed identifiers. IDs are opaque strings generated locally, dependency-free."""

from __future__ import annotations

import secrets
import time
from typing import NewType

EntityId = NewType("EntityId", str)
FactId = NewType("FactId", str)
EpisodeId = NewType("EpisodeId", str)
MemoryId = NewType("MemoryId", str)
ObservationId = NewType("ObservationId", str)
BeliefId = NewType("BeliefId", str)
EvidenceId = NewType("EvidenceId", str)
RelationId = NewType("RelationId", str)
GoalId = NewType("GoalId", str)
RelationshipId = NewType("RelationshipId", str)
StateId = NewType("StateId", str)
SourceId = NewType("SourceId", str)
EmbeddingId = NewType("EmbeddingId", str)
ContradictionId = NewType("ContradictionId", str)
TurnId = NewType("TurnId", str)
SessionId = NewType("SessionId", str)

_ALLOWED_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"


def _prefix(name: str) -> str:
    return name[:1]


def new_id(prefix: str = "") -> str:
    """Create a sortable, random, timestamp-prefixed id.

    Format: <prefix>_<seconds_since_epoch_hex>_<8 random chars>.
    """
    secs = int(time.time())
    rnd = "".join(secrets.choice(_ALLOWED_CHARS) for _ in range(8))
    if prefix:
        return f"{prefix[:12]}_{secs:x}_{rnd}"
    return f"{secs:x}_{rnd}"


def new_entity_id() -> EntityId:
    return EntityId(new_id("ent"))


def new_fact_id() -> FactId:
    return FactId(new_id("fac"))


def new_episode_id() -> EpisodeId:
    return EpisodeId(new_id("ep"))


def new_memory_id() -> MemoryId:
    return MemoryId(new_id("mem"))


def new_observation_id() -> ObservationId:
    return ObservationId(new_id("obs"))


def new_belief_id() -> BeliefId:
    return BeliefId(new_id("bel"))


def new_evidence_id() -> EvidenceId:
    return EvidenceId(new_id("evi"))


def new_relation_id() -> RelationId:
    return RelationId(new_id("rel"))


def new_goal_id() -> GoalId:
    return GoalId(new_id("goal"))


def new_relationship_id() -> RelationshipId:
    return RelationshipId(new_id("rs"))


def new_state_id() -> StateId:
    return StateId(new_id("st"))


def new_source_id() -> SourceId:
    return SourceId(new_id("src"))


def new_embedding_id() -> EmbeddingId:
    return EmbeddingId(new_id("vec"))


def new_contradiction_id() -> ContradictionId:
    return ContradictionId(new_id("con"))


def new_turn_id() -> TurnId:
    return TurnId(new_id("turn"))


def new_session_id() -> SessionId:
    return SessionId(new_id("sess"))
