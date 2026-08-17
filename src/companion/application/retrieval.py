"""HybridRetriever: lexical + semantic + graph + recency + importance +
confidence + relationship relevance, with reranking.

Retrieval modes: recent | semantic | entity | relationship | temporal | goal |
personality | episodic. The conversation orchestrator selects the strategy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from companion.application.ports import GraphStore, VectorStorePort
from companion.core.clock import Clock, SystemClock
from companion.core.types import Vector
from companion.domain.graph import Fact
from companion.domain.memory import Memory

log = logging.getLogger(__name__)


@dataclass
class RetrievedMemory:
    source_type: str      # memory | fact | episode | goal | relationship | personality | knowledge
    id: str
    content: str
    score: float = 0.0
    importance: float = 0.0
    confidence: float = 0.0
    timestamp: str = ""
    scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "importance": self.importance,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "scores": self.scores,
        }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w']+", text.lower()))


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Informative-word matching: common English function words carry no memory
# signal, so lexical relevance is scored on the remaining query tokens only.
_STOPWORDS = frozenset(
    "a an the is are was were be been being am i you he she it we they do does did have has had "
    "me my mine your yours our ours his her its their theirs about of on in at for with by from to "
    "and or but if then than so as not no nor can could would should may might must what which who whom "
    "whose when where why how this that these those there here tell me us him them all any both each "
    "more most other some such only own same very just also too".split()
)


def _lexical(query_tokens: set[str], content_tokens: set[str]) -> float:
    informative = query_tokens - _STOPWORDS
    if not informative:
        return 0.0
    return len(informative & content_tokens) / len(informative)


class HybridRetriever:
    def __init__(
        self,
        graph: GraphStore,
        vector_store: VectorStorePort,
        embeddings=None,
        clock: Clock | None = None,
        embedding_model_id: str = "default",
        recency_halflife_days: float = 30.0,
        weights: dict | None = None,
    ) -> None:
        self._graph = graph
        self._vectors = vector_store
        self._embeddings = embeddings
        self._clock = clock or SystemClock()
        self._embedding_model_id = embedding_model_id
        self._halflife = recency_halflife_days
        self.weights = weights or {
            "lexical": 0.25,
            "semantic": 0.30,
            "graph": 0.10,
            "recency": 0.10,
            "importance": 0.15,
            "confidence": 0.10,
        }

    # -- public entry ----------------------------------------------------

    async def retrieve(self, query: str, mode: str = "auto", top_k: int = 8,
                       min_score: float = 0.0) -> list[RetrievedMemory]:
        mode = mode or "auto"
        candidates: list[RetrievedMemory] = []

        if mode in ("auto", "recent"):
            candidates.extend(self.recent(top_k * 2))
        if mode in ("auto", "semantic"):
            candidates.extend(await self.semantic(query, top_k * 2))
        if mode in ("auto", "entity"):
            candidates.extend(self.entity(query, top_k))
        if mode == "temporal":
            candidates.extend(self.recent(top_k * 2))
        if mode == "goal":
            candidates.extend(self.goals(top_k))
        if mode == "personality":
            candidates.extend(self.personality(top_k))
        if mode == "episodic":
            candidates.extend(self.episodic(top_k))
        if mode == "relationship":
            candidates.extend(self.relationships(top_k))

        if mode in ("auto", "semantic", "recent", "entity", "temporal"):
            query_tokens = _tokens(query)
            for c in candidates:
                if c.source_type in ("memory", "fact", "episode", "knowledge"):
                    self._hybrid_score(c, query, query_tokens)

        # dedup by id, keep best
        best: dict[str, RetrievedMemory] = {}
        for c in candidates:
            key = f"{c.source_type}:{c.id}"
            if key not in best or c.score > best[key].score:
                best[key] = c
        results = sorted(best.values(), key=lambda c: -c.score)[:top_k]
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]
        return results

    # -- mode-specific sources -------------------------------------------

    def recent(self, limit: int = 8) -> list[RetrievedMemory]:
        out: list[RetrievedMemory] = []
        for mem in self._graph.list_memories(limit=limit):
            out.append(_memory_hit(mem))
        for ep in self._graph.list_episodes(limit=max(3, limit // 2)):
            out.append(RetrievedMemory(
                source_type="episode", id=ep.id, content=ep.summary or _episode_preview(ep),
                importance=ep.importance, confidence=0.6, timestamp=ep.ended_at or ep.started_at,
            ))
        return out

    async def semantic(self, query: str, top_k: int = 8) -> list[RetrievedMemory]:
        if self._embeddings is None or self._vectors is None:
            return []
        try:
            vectors: list[Vector] = await self._embeddings.embed([query])
        except Exception as exc:
            log.warning("embedding failed during retrieval: %s", exc)
            return []
        try:
            hits = self._vectors.search(self._embedding_model_id, vectors[0], top_k)
        except Exception as exc:
            log.warning("vector search failed: %s", exc)
            return []
        out: list[RetrievedMemory] = []
        for hit in hits:
            if hit.owner_type == "memory":
                mem = self._graph.get_memory(hit.owner_id)
                if mem:
                    r = _memory_hit(mem)
                    r.scores["semantic"] = hit.score
                    out.append(r)
            elif hit.owner_type == "fact":
                fact = self._graph.get_fact(hit.owner_id)
                if fact:
                    r = _fact_hit(fact)
                    r.scores["semantic"] = hit.score
                    out.append(r)
        return out

    def entity(self, query: str, top_k: int = 8) -> list[RetrievedMemory]:
        tokens = _tokens(query)
        out: list[RetrievedMemory] = []
        for entity in self._graph.list_entities(limit=200):
            name_tokens = _tokens(entity.name)
            sim = _overlap(tokens, name_tokens)
            if sim < 0.4 and entity.name.lower() not in query.lower():
                continue
            for fact in self._graph.list_facts(entity.id)[:3]:
                out.append(_fact_hit(fact))
            for fact in self._graph.list_facts_about_object(entity.id)[:3]:
                out.append(_fact_hit(fact))
        return out

    def goals(self, top_k: int = 8) -> list[RetrievedMemory]:
        out: list[RetrievedMemory] = []
        for goal in self._graph.list_goals(status="active"):
            out.append(RetrievedMemory(
                source_type="goal", id=goal.id, content=f"goal: {goal.name} — {goal.description}",
                importance=goal.priority, confidence=goal.confidence, timestamp=goal.updated_at,
            ))
        return out[:top_k]

    def personality(self, top_k: int = 8) -> list[RetrievedMemory]:
        out: list[RetrievedMemory] = []
        for belief in self._graph.list_beliefs(target_type="personality", status="active"):
            val = belief.value.get("value", belief.value)
            out.append(RetrievedMemory(
                source_type="personality", id=belief.id,
                content=f"{belief.target_name} {belief.predicate} {val} (conf {belief.confidence:.2f})",
                importance=belief.importance, confidence=belief.confidence, timestamp=belief.updated_at,
            ))
        return out[:top_k]

    def episodic(self, top_k: int = 8) -> list[RetrievedMemory]:
        out: list[RetrievedMemory] = []
        for ep in self._graph.list_episodes(limit=top_k):
            out.append(RetrievedMemory(
                source_type="episode", id=ep.id, content=ep.summary or _episode_preview(ep),
                importance=ep.importance, confidence=0.6, timestamp=ep.ended_at or ep.started_at,
            ))
        return out

    def relationships(self, top_k: int = 8) -> list[RetrievedMemory]:
        out: list[RetrievedMemory] = []
        for rel in self._graph.list_relationships():
            out.append(RetrievedMemory(
                source_type="relationship", id=rel.id,
                content=f"relationship with {rel.name}: trust {rel.trust:.2f}, familiarity {rel.familiarity:.2f}, "
                        f"valence {rel.emotional_valence:+.2f}, {rel.interaction_count} interactions",
                importance=0.6, confidence=rel.confidence, timestamp=rel.last_interaction,
            ))
        return out[:top_k]

    def knowledge(self, query: str, top_k: int = 4) -> list[RetrievedMemory]:
        out: list[RetrievedMemory] = []
        for chunk in self._graph.list_knowledge(limit=top_k * 5):
            if _overlap(_tokens(query), _tokens(chunk.content)) > 0.25 or query.lower() in chunk.content.lower():
                out.append(RetrievedMemory(
                    source_type="knowledge", id=chunk.id, content=chunk.content,
                    importance=0.4, confidence=chunk.confidence, timestamp=chunk.created_at,
                ))
        return out[:top_k]

    # -- scoring ----------------------------------------------------------

    def _hybrid_score(self, hit: RetrievedMemory, query: str, query_tokens: set[str]) -> None:
        content_tokens = _tokens(hit.content)
        lexical = _lexical(query_tokens, content_tokens)
        semantic = hit.scores.get("semantic", 0.35 if self._embeddings is None else 0.0)
        graph = hit.scores.get("graph", 0.0)
        recency = _recency(hit.timestamp, self._halflife)
        w = self.weights
        hit.score = (
            w["lexical"] * lexical
            + w["semantic"] * semantic
            + w["graph"] * graph
            + w["recency"] * recency
            + w["importance"] * hit.importance
            + w["confidence"] * hit.confidence
        )
        hit.scores = {
            "lexical": lexical, "semantic": semantic, "graph": graph,
            "recency": recency, "importance": hit.importance, "confidence": hit.confidence,
        }


def _memory_hit(mem: Memory) -> RetrievedMemory:
    return RetrievedMemory(
        source_type="memory", id=mem.id, content=mem.content,
        importance=mem.importance, confidence=mem.confidence,
        timestamp=mem.created_at,
    )


def _fact_hit(fact: Fact) -> RetrievedMemory:
    obj = fact.object_id or fact.value or ""
    return RetrievedMemory(
        source_type="fact", id=fact.id,
        content=f"{fact.subject_id} {fact.predicate} {obj}",
        importance=fact.importance, confidence=fact.confidence, timestamp=fact.created_at,
    )


def _episode_preview(ep) -> str:
    texts = [t.get("text", "") for t in ep.transcript[:3]]
    return " | ".join(texts)[:300]


def _recency(timestamp: str, halflife_days: float) -> float:
    if not timestamp:
        return 0.5
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        now = datetime.now(dt.tzinfo)
        days = max(0.0, (now - dt).total_seconds() / 86400.0)
        return 2.0 ** (-days / halflife_days)
    except (ValueError, TypeError):
        return 0.5
