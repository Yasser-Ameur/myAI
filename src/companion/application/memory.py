"""Memory pipeline and service.

Episode -> candidate memories -> dedup -> contradiction detection -> confidence
scoring -> graph update. Memories follow an explicit lifecycle; decay affects
retrieval, not deletion. User-facing forget/correct/lock act on real records.
"""

from __future__ import annotations

import logging
import re

from companion.application.extraction import ExtractedMemory, StructuredExtractor
from companion.application.personality import PersonalityEngine
from companion.application.ports import GraphStore, VectorStorePort
from companion.application.relationship import RelationshipEngine
from companion.core.clock import Clock, SystemClock
from companion.core.ids import (
    new_fact_id,
    new_goal_id,
    new_memory_id,
    new_observation_id,
)
from companion.core.types import Vector
from companion.domain.graph import Entity, Fact, Goal, Observation, Source, SourceType
from companion.domain.memory import Episode, Memory, MemoryStatus, MemoryType

log = logging.getLogger(__name__)

DUPLICATE_THRESHOLD = 0.8


class MemoryPipeline:
    def __init__(
        self,
        graph: GraphStore,
        vector_store: VectorStorePort,
        embeddings,
        clock: Clock | None = None,
        extractor: StructuredExtractor | None = None,
        personality: PersonalityEngine | None = None,
        relationships: RelationshipEngine | None = None,
        embedding_model_id: str = "default",
        embedding_dimension: int = 0,
    ) -> None:
        self._graph = graph
        self._vectors = vector_store
        self._embeddings = embeddings
        self._clock = clock or SystemClock()
        self._extractor = extractor or StructuredExtractor()
        self._personality = personality or PersonalityEngine(graph, clock)
        self._relationships = relationships or RelationshipEngine(graph, clock)
        self._embedding_model_id = embedding_model_id
        self._embedding_dim = embedding_dimension
        self.stats = {"episodes_processed": 0, "memories_created": 0, "memories_reinforced": 0,
                      "contradictions_found": 0}

    async def process_episode(self, episode: Episode) -> None:
        transcript = self._episode_to_text(episode)
        if not transcript.strip():
            return
        extraction = await self._extractor.extract(transcript, episode.id)

        # Hallucination defence. A model handed a transcript will happily
        # extract its OWN answers as facts about the user — an observed
        # failure, not a hypothetical one. Anything the model proposes must be
        # traceable to something the user actually said, or it is dropped.
        if extraction.method == "llm":
            user_text = self._user_text(episode)
            kept, rejected = [], []
            for mem in extraction.memories:
                (kept if _grounded_in(mem, user_text) else rejected).append(mem)
            if rejected:
                log.info("dropped %d ungrounded extracted memories: %s",
                         len(rejected), [m.content[:60] for m in rejected])
                self.stats["ungrounded_rejected"] = (
                    self.stats.get("ungrounded_rejected", 0) + len(rejected))
            extraction.memories = kept

        for mem in extraction.memories:
            await self._ingest_memory(episode, mem)

        for ev in extraction.personality_evidence:
            self._personality.apply_evidence(ev)

        for goal_dict in extraction.goals:
            self._upsert_goal(goal_dict, episode.id)

        for rel in extraction.relationships:
            try:
                self._relationships.note_interaction(
                    person_name=str(rel.get("person", "")),
                    episode_id=episode.id,
                    valence_delta=float(rel.get("valence_delta", 0.0)),
                    event_note=str(rel.get("note", "")),
                )
            except Exception as exc:
                log.warning("relationship update failed: %s", exc)

        self._personality.learn_communication_preference(
            "\n".join(t["text"] for t in episode.transcript if t.get("role") == "user"),
            episode.id,
        )

        episode.is_consolidated = True
        episode.summary = self._summarize(episode)
        self._graph.save_episode(episode)
        self.stats["episodes_processed"] += 1

    # -- ingestion --------------------------------------------------------

    async def _ingest_memory(self, episode: Episode, mem: ExtractedMemory) -> None:
        if not mem.content:
            return
        existing = self._dedup(mem.content)
        if existing is not None:
            # reinforce instead of duplicating
            existing.importance = min(1.0, existing.importance + 0.05)
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.updated_at = self._clock.now_iso()
            self._graph.add_memory(existing)
            self.stats["memories_reinforced"] += 1
            return

        memory = Memory(
            id=new_memory_id(),
            type=_memory_type(mem.type),
            content=mem.content,
            importance=min(1.0, max(0.0, mem.importance)),
            confidence=min(1.0, max(0.0, mem.confidence)),
            status=MemoryStatus.CANDIDATE,
            created_at=self._clock.now_iso(),
            updated_at=self._clock.now_iso(),
            source_episode_id=episode.id,
            meta={"subject": mem.subject, "predicate": mem.predicate, "object": mem.object},
        )
        self._graph.add_memory(memory)
        await self._stamp_embedding(memory)
        self._upsert_graph_objects(episode, mem, memory.id)
        self.stats["memories_created"] += 1

    async def _stamp_embedding(self, memory: Memory) -> None:
        if self._embeddings is None or self._vectors is None:
            return
        try:
            vectors: list[Vector] = await self._embeddings.embed([memory.content])
            vec_id = f"mem:{memory.id}"
            self._vectors.upsert(vec_id, self._embedding_model_id, vectors[0], "memory", memory.id)
            memory.embedding_id = vec_id
            memory.status = MemoryStatus.VALIDATED
            self._graph.add_memory(memory)
        except Exception as exc:
            log.warning("embedding failed for memory %s: %s", memory.id, exc)

    def _upsert_graph_objects(self, episode: Episode, mem: ExtractedMemory, memory_id: str) -> None:
        user_id = self._graph.get_system_state("primary_user_entity") or "user"
        subject_id = user_id
        object_id: str | None = None
        if mem.subject and mem.subject != "user":
            subject_id = self._entity_for(mem.subject).id
        if mem.object:
            object_id = self._entity_for(mem.object).id

        source = Source(
            type=SourceType.CONVERSATION,
            name=f"episode:{episode.id}",
            uri=episode.id,
            created_at=self._clock.now_iso(),
        )
        source = self._graph.get_or_create_source(source)

        fact = Fact(
            id=new_fact_id(),
            subject_id=subject_id,
            predicate=mem.predicate or "mentioned",
            object_id=object_id,
            value=None if object_id else mem.object,
            confidence=mem.confidence,
            importance=mem.importance,
            created_at=self._clock.now_iso(),
            valid_from=self._clock.now_iso(),
            source_episode_id=episode.id,
            source_id=source.id,
            last_confirmed_at=self._clock.now_iso(),
            embedding_id=f"mem:{memory_id}",
            provenance="conversation",
        )
        self._graph.add_fact(fact)
        self._graph.add_observation(
            Observation(
                id=new_observation_id(),
                kind="memory_extraction",
                payload={"memory_id": memory_id, "fact_id": fact.id, "content": mem.content},
                episode_id=episode.id,
                timestamp=self._clock.now_iso(),
                confidence=mem.confidence,
            )
        )

    def _entity_for(self, name: str) -> Entity:
        entity = self._graph.find_entity_by_name(name)
        if entity is None:
            entity = Entity(
                type=_guess_entity_type(name),
                name=name,
                created_at=self._clock.now_iso(),
                updated_at=self._clock.now_iso(),
                importance=0.4,
            )
            self._graph.upsert_entity(entity)
        return entity

    def _upsert_goal(self, goal_dict: dict, episode_id: str) -> None:
        name = str(goal_dict.get("name", "")).strip()
        if not name:
            return
        existing = self._graph.list_goals(status="active")
        for g in existing:
            if g.name.lower() == name.lower():
                g.description = str(goal_dict.get("description", g.description))
                g.progress = min(1.0, g.progress + 0.05)
                g.updated_at = self._clock.now_iso()
                self._graph.upsert_goal(g)
                return
        goal = Goal(
            id=new_goal_id(),
            name=name,
            description=str(goal_dict.get("description", "")),
            status="active",
            priority=0.6,
            confidence=0.5,
            source_episode_id=episode_id,
            created_at=self._clock.now_iso(),
            updated_at=self._clock.now_iso(),
        )
        self._graph.upsert_goal(goal)

    # -- dedup / scoring ---------------------------------------------------

    def _dedup(self, content: str) -> Memory | None:
        candidates = self._graph.list_memories(
            status="", limit=500
        )
        norm = _normalize(content)
        best: Memory | None = None
        best_score = 0.0
        for mem in candidates:
            if mem.status in (MemoryStatus.FORGOTTEN.value, MemoryStatus.ARCHIVED.value):
                continue
            score = _jaccard(norm, _normalize(mem.content))
            if score > best_score:
                best_score = score
                best = mem
        if best_score >= DUPLICATE_THRESHOLD:
            return best
        return None

    def _summarize(self, episode: Episode) -> str:
        texts = [t["text"] for t in episode.transcript if t.get("role") == "user"]
        if not texts:
            return ""
        joined = " ".join(texts)
        return joined[:400]

    @staticmethod
    def _episode_to_text(episode: Episode) -> str:
        return "\n".join(
            f"{t.get('role', 'user')}: {t.get('text', '')}" for t in episode.transcript
        )

    @staticmethod
    def _user_text(episode: Episode) -> str:
        return " ".join(t.get("text", "") for t in episode.transcript
                        if t.get("role") == "user").lower()


class MemoryService:
    """Manages the current episode and exposes memory lifecycle commands."""

    def __init__(self, graph: GraphStore, pipeline: MemoryPipeline,
                 clock: Clock | None = None) -> None:
        self._graph = graph
        self._pipeline = pipeline
        self._clock = clock or SystemClock()
        self._current: Episode | None = None

    def begin_episode(self, user_state_before: dict | None = None) -> Episode:
        ep = Episode(
            started_at=self._clock.now_iso(),
            user_state_before=user_state_before or {},
            participants=["user", "assistant"],
        )
        self._graph.save_episode(ep)
        self._current = ep
        return ep

    def current_episode(self) -> Episode | None:
        return self._current

    def append_turn(self, role: str, text: str, source: str = "text",
                    user_state: dict | None = None) -> None:
        if self._current is None:
            self.begin_episode()
        ep = self._current
        ep.transcript.append(
            {"role": role, "text": text, "timestamp": self._clock.now_iso()}
        )
        self._graph.add_turn(
            turn_id=f"turn:{ep.id}:{len(ep.transcript)}",
            episode_id=ep.id,
            role=role,
            text=text,
            timestamp=self._clock.now_iso(),
            source=source,
            user_state=user_state or {},
        )
        self._graph.save_episode(ep)

    async def close_episode(self, user_state_after: dict | None = None) -> Episode | None:
        ep = self._current
        if ep is None:
            return None
        ep.ended_at = self._clock.now_iso()
        ep.user_state_after = user_state_after or {}
        ep.importance = self._estimate_importance(ep)
        self._graph.save_episode(ep)
        self._current = None
        await self._pipeline.process_episode(ep)
        return ep

    @staticmethod
    def _estimate_importance(ep: Episode) -> float:
        n = len(ep.transcript)
        base = min(1.0, n / 12.0)
        if any("wants to study" in t.get("text", "") or "working on" in t.get("text", "")
               for t in ep.transcript):
            base = max(base, 0.7)
        return base

    # -- lifecycle commands ------------------------------------------------

    def validate_memory(self, memory_id: str) -> None:
        self._graph.update_memory_status(memory_id, MemoryStatus.VALIDATED.value)

    def activate_memory(self, memory_id: str) -> None:
        self._graph.update_memory_status(memory_id, MemoryStatus.ACTIVE.value)

    def archive_memory(self, memory_id: str) -> None:
        self._graph.update_memory_status(memory_id, MemoryStatus.ARCHIVED.value)

    def forget_memory(self, memory_id: str) -> None:
        self._graph.forget_memory(memory_id)

    def lock_memory(self, memory_id: str, locked: bool) -> None:
        self._graph.set_memory_locked(memory_id, locked)

    def correct_memory(self, memory_id: str, new_content: str) -> None:
        mem = self._graph.get_memory(memory_id)
        if mem is None:
            return
        # Preserve history: old record archived, new record created.
        self._graph.update_memory_status(memory_id, MemoryStatus.ARCHIVED.value)
        corrected = Memory(
            id=new_memory_id(),
            type=mem.type,
            content=new_content,
            importance=mem.importance,
            confidence=min(1.0, mem.confidence + 0.1),
            status=MemoryStatus.VALIDATED,
            created_at=self._clock.now_iso(),
            updated_at=self._clock.now_iso(),
            source_episode_id=mem.source_episode_id,
            meta={**mem.meta, "corrected_from": memory_id},
        )
        self._graph.add_memory(corrected)

    def stats(self) -> dict:
        memories = self._graph.list_memories(limit=100000)
        counts: dict[str, int] = {}
        for m in memories:
            counts[m.status] = counts.get(m.status, 0) + 1
        episodes = self._graph.list_episodes(limit=100000)
        return {
            "memories": counts,
            "total_memories": len(memories),
            "episodes": len(episodes),
            "pipeline": self._pipeline.stats,
        }


# Function words carry no evidence, so they are ignored when checking whether
# an extracted claim is actually supported by what the user said.
_GROUNDING_STOPWORDS = frozenset(
    "a an the is are was were be been being am i you he she it we they do does did "
    "have has had my your our their me him her them this that these those and or but "
    "if then so as not no of on in at for with by from to about user".split()
)


def _grounded_in(mem: ExtractedMemory, user_text: str) -> bool:
    """Is this extracted claim traceable to the user's own words?

    Requires that the specific content of the claim — its object, or failing
    that a majority of its informative tokens — appears in what the user
    actually said. Deliberately lexical: a cheap, deterministic check is the
    right tool for rejecting confabulation, and it cannot itself hallucinate.
    """
    if not user_text:
        return False
    obj = (mem.object or "").strip().lower()
    if obj:
        return obj in user_text
    tokens = [t for t in re.findall(r"[\w']+", (mem.content or "").lower())
              if t not in _GROUNDING_STOPWORDS]
    if not tokens:
        return False
    present = sum(1 for t in tokens if t in user_text)
    return present / len(tokens) >= 0.6


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _jaccard(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _memory_type(raw: str) -> MemoryType:
    mapping = {
        "semantic": MemoryType.SEMANTIC,
        "episodic": MemoryType.EPISODIC,
        "procedural": MemoryType.PROCEDURAL,
        "preference": MemoryType.PREFERENCE,
        "relationship": MemoryType.RELATIONSHIP,
        "goal": MemoryType.GOAL,
        "personality": MemoryType.PERSONALITY,
        "world_knowledge": MemoryType.WORLD_KNOWLEDGE,
    }
    return mapping.get(raw.lower(), MemoryType.SEMANTIC)


def _guess_entity_type(name: str) -> str:
    lowered = name.lower()
    if any(k in lowered for k in ("project", "app", "system", "tool", "software")):
        return "project"
    if any(k in lowered for k in ("france", "paris", "city", "country", "mountain")):
        return "place"
    return "thing"
