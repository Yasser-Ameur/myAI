"""Knowledge domain: external knowledge is distinguishable from personal belief.

The agent must know whether "user says X", "document says X" and "I inferred X"
are different claims. Provenance lives on every knowledge record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from companion.domain.graph import KnowledgeChunk, Source

# Canonical knowledge categories.
KNOWLEDGE_CATEGORIES = ("user", "documents", "projects", "websites", "world")


@dataclass
class KnowledgeRecord:
    """A single sourced knowledge claim."""

    id: str = ""
    chunk: KnowledgeChunk = field(default_factory=KnowledgeChunk)
    category: str = "world"
    source: Source = field(default_factory=Source)
    tags: list[str] = field(default_factory=list)
    importance: float = 0.3

    @property
    def provenance_kind(self) -> str:
        return self.source.type.value

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "chunk": self.chunk.to_dict(),
            "category": self.category,
            "source": self.source.to_dict(),
            "tags": self.tags,
            "importance": self.importance,
        }
