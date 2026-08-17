"""Memory domain model: memory records and episodes.

Memories have an explicit lifecycle (RAW -> CANDIDATE -> VALIDATED -> ACTIVE
-> DECAYING -> ARCHIVED). Decay influences retrieval, it does not delete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from companion.core.ids import new_episode_id, new_memory_id


class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    GOAL = "goal"
    PERSONALITY = "personality"
    WORLD_KNOWLEDGE = "world_knowledge"


class MemoryStatus(str, Enum):
    RAW = "raw"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    ACTIVE = "active"
    DECAYING = "decaying"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"  # explicitly removed by the user


@dataclass
class Memory:
    id: str = field(default_factory=new_memory_id)
    type: MemoryType = MemoryType.SEMANTIC
    content: str = ""
    importance: float = 0.3
    confidence: float = 0.5
    status: MemoryStatus = MemoryStatus.CANDIDATE
    created_at: str = ""
    updated_at: str = ""
    accessed_at: str = ""
    retrieval_count: int = 0
    source_episode_id: str = ""
    embedding_id: str = ""
    locked: bool = False
    meta: dict = field(default_factory=dict)  # provenance, entities, etc.

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "importance": self.importance,
            "confidence": self.confidence,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "retrieval_count": self.retrieval_count,
            "source_episode_id": self.source_episode_id,
            "embedding_id": self.embedding_id,
            "locked": self.locked,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        return cls(
            id=str(d.get("id", new_memory_id())),
            type=MemoryType(d.get("type", MemoryType.SEMANTIC.value)),
            content=str(d.get("content", "")),
            importance=float(d.get("importance", 0.3)),
            confidence=float(d.get("confidence", 0.5)),
            status=MemoryStatus(d.get("status", MemoryStatus.CANDIDATE.value)),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            accessed_at=str(d.get("accessed_at", "")),
            retrieval_count=int(d.get("retrieval_count", 0)),
            source_episode_id=str(d.get("source_episode_id", "")),
            embedding_id=str(d.get("embedding_id", "")),
            locked=bool(d.get("locked", False)),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class Episode:
    """A meaningful conversational interaction, stored verbatim-ish."""

    id: str = field(default_factory=new_episode_id)
    started_at: str = ""
    ended_at: str = ""
    transcript: list[dict] = field(default_factory=list)  # [{role, text, timestamp}]
    participants: list[str] = field(default_factory=list)
    user_state_before: dict = field(default_factory=dict)
    user_state_after: dict = field(default_factory=dict)
    assistant_state: dict = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    outcome: str = ""
    importance: float = 0.3
    is_consolidated: bool = False
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "transcript": self.transcript,
            "participants": self.participants,
            "user_state_before": self.user_state_before,
            "user_state_after": self.user_state_after,
            "assistant_state": self.assistant_state,
            "topics": self.topics,
            "entities": self.entities,
            "actions": self.actions,
            "outcome": self.outcome,
            "importance": self.importance,
            "is_consolidated": self.is_consolidated,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(
            id=str(d.get("id", new_episode_id())),
            started_at=str(d.get("started_at", "")),
            ended_at=str(d.get("ended_at", "")),
            transcript=list(d.get("transcript", [])),
            participants=list(d.get("participants", [])),
            user_state_before=dict(d.get("user_state_before", {})),
            user_state_after=dict(d.get("user_state_after", {})),
            assistant_state=dict(d.get("assistant_state", {})),
            topics=list(d.get("topics", [])),
            entities=list(d.get("entities", [])),
            actions=list(d.get("actions", [])),
            outcome=str(d.get("outcome", "")),
            importance=float(d.get("importance", 0.3)),
            is_consolidated=bool(d.get("is_consolidated", False)),
            summary=str(d.get("summary", "")),
        )
