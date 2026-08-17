"""Cognitive graph domain model: entities, facts, relations, observations,
beliefs, goals, sources and knowledge chunks.

All facts carry temporal metadata (valid_from/valid_to), confidence and
provenance. Historical information is never destroyed silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from companion.core.ids import (
    new_belief_id,
    new_entity_id,
    new_fact_id,
    new_relation_id,
    new_source_id,
)


class SourceType(str, Enum):
    USER = "user"
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    WEB = "web"
    MODEL_INFERENCE = "model_inference"
    SYSTEM = "system"


@dataclass
class Source:
    id: str = field(default_factory=new_source_id)
    type: SourceType = SourceType.CONVERSATION
    name: str = ""
    uri: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "uri": self.uri,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        return cls(
            id=str(d.get("id", new_source_id())),
            type=SourceType(d.get("type", SourceType.CONVERSATION.value)),
            name=str(d.get("name", "")),
            uri=str(d.get("uri", "")),
            created_at=str(d.get("created_at", "")),
        )


@dataclass
class Entity:
    id: str = field(default_factory=new_entity_id)
    type: str = "thing"  # person | project | place | concept | thing | ...
    name: str = ""
    summary: str = ""
    confidence: float = 0.5
    importance: float = 0.3
    created_at: str = ""
    updated_at: str = ""
    valid_from: str = ""
    valid_to: str = ""  # empty = still valid
    is_deleted: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "summary": self.summary,
            "confidence": self.confidence,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            id=str(d.get("id", new_entity_id())),
            type=str(d.get("type", "thing")),
            name=str(d.get("name", "")),
            summary=str(d.get("summary", "")),
            confidence=float(d.get("confidence", 0.5)),
            importance=float(d.get("importance", 0.3)),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            valid_from=str(d.get("valid_from", "")),
            valid_to=str(d.get("valid_to", "")),
            is_deleted=bool(d.get("is_deleted", False)),
        )


@dataclass
class Fact:
    id: str = field(default_factory=new_fact_id)
    subject_id: str = ""
    predicate: str = ""
    object_id: str | None = None
    value: str | None = None  # literal object when there is no object entity
    confidence: float = 0.5
    importance: float = 0.3
    created_at: str = ""
    valid_from: str = ""
    valid_to: str = ""
    source_episode_id: str = ""
    source_id: str = ""
    last_confirmed_at: str = ""
    embedding_id: str = ""
    is_deleted: bool = False
    provenance: str = SourceType.CONVERSATION.value

    def display(self) -> str:
        obj = self.object_id or self.value or ""
        return f"{self.subject_id} --[{self.predicate}]--> {obj}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "value": self.value,
            "confidence": self.confidence,
            "importance": self.importance,
            "created_at": self.created_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "source_episode_id": self.source_episode_id,
            "source_id": self.source_id,
            "last_confirmed_at": self.last_confirmed_at,
            "embedding_id": self.embedding_id,
            "is_deleted": self.is_deleted,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(
            id=str(d.get("id", new_fact_id())),
            subject_id=str(d.get("subject_id", "")),
            predicate=str(d.get("predicate", "")),
            object_id=d.get("object_id"),
            value=d.get("value"),
            confidence=float(d.get("confidence", 0.5)),
            importance=float(d.get("importance", 0.3)),
            created_at=str(d.get("created_at", "")),
            valid_from=str(d.get("valid_from", "")),
            valid_to=str(d.get("valid_to", "")),
            source_episode_id=str(d.get("source_episode_id", "")),
            source_id=str(d.get("source_id", "")),
            last_confirmed_at=str(d.get("last_confirmed_at", "")),
            embedding_id=str(d.get("embedding_id", "")),
            is_deleted=bool(d.get("is_deleted", False)),
            provenance=str(d.get("provenance", SourceType.CONVERSATION.value)),
        )


@dataclass
class Relation:
    id: str = field(default_factory=new_relation_id)
    type: str = ""  # knows | works_with | parent_of | member_of | ...
    subject_id: str = ""
    target_id: str = ""
    properties: dict = field(default_factory=dict)
    confidence: float = 0.5
    created_at: str = ""
    valid_from: str = ""
    valid_to: str = ""
    is_deleted: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "subject_id": self.subject_id,
            "target_id": self.target_id,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        return cls(
            id=str(d.get("id", new_relation_id())),
            type=str(d.get("type", "")),
            subject_id=str(d.get("subject_id", "")),
            target_id=str(d.get("target_id", "")),
            properties=dict(d.get("properties", {})),
            confidence=float(d.get("confidence", 0.5)),
            created_at=str(d.get("created_at", "")),
            valid_from=str(d.get("valid_from", "")),
            valid_to=str(d.get("valid_to", "")),
            is_deleted=bool(d.get("is_deleted", False)),
        )


@dataclass
class Observation:
    """A raw recorded observation, kept as evidence for later reasoning."""

    id: str = ""
    kind: str = ""  # user_state | facial | acoustic | behavioral | statement | outcome
    payload: dict = field(default_factory=dict)
    episode_id: str = ""
    timestamp: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "payload": self.payload,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }


@dataclass
class Belief:
    """A structured belief about the user/world with evidence and confidence."""

    id: str = field(default_factory=new_belief_id)
    target_type: str = ""      # trait | preference | relationship | fact | communication_style
    target_name: str = ""      # e.g. "curiosity"
    predicate: str = ""        # e.g. "has", "likes"
    value: dict = field(default_factory=dict)
    confidence: float = 0.5
    evidence: list[dict] = field(default_factory=list)
    status: str = "active"     # active | conflicting | superseded | locked | forgotten
    created_at: str = ""
    updated_at: str = ""
    importance: float = 0.3

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target_type": self.target_type,
            "target_name": self.target_name,
            "predicate": self.predicate,
            "value": self.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Belief":
        return cls(
            id=str(d.get("id", new_belief_id())),
            target_type=str(d.get("target_type", "")),
            target_name=str(d.get("target_name", "")),
            predicate=str(d.get("predicate", "")),
            value=dict(d.get("value", {})),
            confidence=float(d.get("confidence", 0.5)),
            evidence=list(d.get("evidence", [])),
            status=str(d.get("status", "active")),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            importance=float(d.get("importance", 0.3)),
        )


@dataclass
class Goal:
    id: str = ""
    name: str = ""
    description: str = ""
    status: str = "active"  # active | paused | completed | abandoned
    priority: float = 0.5
    progress: float = 0.0
    confidence: float = 0.5
    source_episode_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "progress": self.progress,
            "confidence": self.confidence,
            "source_episode_id": self.source_episode_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Goal":
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            status=str(d.get("status", "active")),
            priority=float(d.get("priority", 0.5)),
            progress=float(d.get("progress", 0.0)),
            confidence=float(d.get("confidence", 0.5)),
            source_episode_id=str(d.get("source_episode_id", "")),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
        )


@dataclass
class KnowledgeChunk:
    """External/local knowledge, distinguishable from personal belief."""

    id: str = ""
    source_id: str = ""
    source_type: SourceType = SourceType.DOCUMENT
    content: str = ""
    title: str = ""
    confidence: float = 0.5
    embedding_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "content": self.content,
            "title": self.title,
            "confidence": self.confidence,
            "embedding_id": self.embedding_id,
            "created_at": self.created_at,
        }
