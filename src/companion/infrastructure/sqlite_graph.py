"""CognitiveGraph: the SQLite-backed temporal knowledge graph repository.

Implements GraphStore (see application/ports.py). All state lives in one SQLite
database. Facts are temporal: invalidation sets valid_to instead of deleting.
"""

from __future__ import annotations

import logging

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
    Preference,
    Trait,
    Value,
)
from companion.domain.relationship import Relationship
from companion.infrastructure.sqlite_schema import SCHEMA_SQL, SCHEMA_VERSION
from companion.infrastructure.storage import SqliteStorage, decode_json, encode_json

log = logging.getLogger(__name__)


def _row_to_entity(row) -> Entity | None:
    if row is None:
        return None
    return Entity.from_dict(dict(row))


def _row_to_fact(row) -> Fact | None:
    if row is None:
        return None
    d = dict(row)
    d["is_deleted"] = bool(d.get("is_deleted"))
    return Fact.from_dict(d)


class CognitiveGraph:
    def __init__(self, storage: SqliteStorage) -> None:
        self._storage = storage
        self._storage.ensure_schema(SCHEMA_SQL, SCHEMA_VERSION)
        self._boot()

    def _boot(self) -> None:
        if self.get_system_state("primary_user_entity") is None:
            user = Entity(
                type="person",
                name="user",
                summary="The primary user of this companion.",
                confidence=1.0,
                importance=0.9,
            )
            self.upsert_entity(user)
            self.set_system_state("primary_user_entity", user.id)

    # ---------------- entities ----------------

    def upsert_entity(self, entity: Entity) -> None:
        self._storage.execute(
            "INSERT INTO entities(id, type, name, summary, confidence, importance, created_at, updated_at, "
            "valid_from, valid_to, is_deleted) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "type=excluded.type, name=excluded.name, summary=excluded.summary, "
            "confidence=excluded.confidence, importance=excluded.importance, "
            "updated_at=excluded.updated_at, valid_from=excluded.valid_from, "
            "valid_to=excluded.valid_to, is_deleted=excluded.is_deleted",
            (
                entity.id,
                entity.type,
                entity.name,
                entity.summary,
                entity.confidence,
                entity.importance,
                entity.created_at,
                entity.updated_at,
                entity.valid_from,
                entity.valid_to,
                1 if entity.is_deleted else 0,
            ),
        )

    def get_entity(self, entity_id: str) -> Entity | None:
        return _row_to_entity(
            self._storage.query_one(
                "SELECT * FROM entities WHERE id=? AND is_deleted=0", (entity_id,)
            )
        )

    def find_entity_by_name(self, name: str, type: str = "") -> Entity | None:
        if type:
            row = self._storage.query_one(
                "SELECT * FROM entities WHERE lower(name)=lower(?) AND type=? AND is_deleted=0 ORDER BY importance DESC LIMIT 1",
                (name, type),
            )
        else:
            row = self._storage.query_one(
                "SELECT * FROM entities WHERE lower(name)=lower(?) AND is_deleted=0 ORDER BY importance DESC LIMIT 1",
                (name,),
            )
        return _row_to_entity(row)

    def list_entities(self, type: str = "", limit: int = 100) -> list[Entity]:
        if type:
            rows = self._storage.query(
                "SELECT * FROM entities WHERE type=? AND is_deleted=0 ORDER BY importance DESC LIMIT ?",
                (type, limit),
            )
        else:
            rows = self._storage.query(
                "SELECT * FROM entities WHERE is_deleted=0 ORDER BY importance DESC LIMIT ?",
                (limit,),
            )
        return [_row_to_entity(r) for r in rows if r is not None]

    def primary_user_entity_id(self) -> str:
        return self.get_system_state("primary_user_entity") or ""

    # ---------------- facts ----------------

    def add_fact(self, fact: Fact) -> None:
        self._storage.execute(
            "INSERT INTO facts(id, subject_id, predicate, object_id, value, confidence, importance, "
            "created_at, valid_from, valid_to, source_episode_id, source_id, last_confirmed_at, "
            "embedding_id, is_deleted, provenance) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                fact.id,
                fact.subject_id,
                fact.predicate,
                fact.object_id,
                fact.value,
                fact.confidence,
                fact.importance,
                fact.created_at,
                fact.valid_from,
                fact.valid_to,
                fact.source_episode_id,
                fact.source_id,
                fact.last_confirmed_at,
                fact.embedding_id,
                1 if fact.is_deleted else 0,
                fact.provenance,
            ),
        )

    def get_fact(self, fact_id: str) -> Fact | None:
        return _row_to_fact(self._storage.query_one("SELECT * FROM facts WHERE id=?", (fact_id,)))

    def list_facts(self, subject_id: str = "", predicate: str = "", include_deleted: bool = False) -> list[Fact]:
        sql = "SELECT * FROM facts"
        params: list = []
        clauses: list[str] = []
        if subject_id:
            clauses.append("subject_id=?")
            params.append(subject_id)
        if predicate:
            clauses.append("predicate=?")
            params.append(predicate)
        if not include_deleted:
            clauses.append("is_deleted=0")
            clauses.append("(valid_to='' OR valid_to > ?)")
            params.append(self._now())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        rows = self._storage.query(sql + " ORDER BY created_at DESC", tuple(params))
        return [f for f in (_row_to_fact(r) for r in rows) if f is not None]

    def list_facts_about_object(self, object_id: str) -> list[Fact]:
        rows = self._storage.query(
            "SELECT * FROM facts WHERE object_id=? AND is_deleted=0 AND (valid_to='' OR valid_to > ?) ORDER BY created_at DESC",
            (object_id, self._now()),
        )
        return [f for f in (_row_to_fact(r) for r in rows) if f is not None]

    def invalidate_fact(self, fact_id: str, valid_to: str) -> None:
        self._storage.execute("UPDATE facts SET valid_to=? WHERE id=?", (valid_to, fact_id))

    def confirm_fact(self, fact_id: str, at: str) -> None:
        self._storage.execute(
            "UPDATE facts SET last_confirmed_at=? WHERE id=?", (at, fact_id)
        )

    def set_fact_confidence(self, fact_id: str, confidence: float) -> None:
        self._storage.execute(
            "UPDATE facts SET confidence=? WHERE id=?",
            (max(0.0, min(1.0, float(confidence))), fact_id),
        )

    def delete_fact(self, fact_id: str) -> None:
        self._storage.execute("UPDATE facts SET is_deleted=1 WHERE id=?", (fact_id,))

    # ---------------- relations ----------------

    def add_relation(self, relation: Relation) -> None:
        self._storage.execute(
            "INSERT INTO relations(id, type, subject_id, target_id, properties, confidence, "
            "created_at, valid_from, valid_to, is_deleted) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                relation.id,
                relation.type,
                relation.subject_id,
                relation.target_id,
                encode_json(relation.properties),
                relation.confidence,
                relation.created_at,
                relation.valid_from,
                relation.valid_to,
                1 if relation.is_deleted else 0,
            ),
        )

    def list_relations(self, entity_id: str) -> list[Relation]:
        rows = self._storage.query(
            "SELECT * FROM relations WHERE (subject_id=? OR target_id=?) AND is_deleted=0",
            (entity_id, entity_id),
        )
        result: list[Relation] = []
        for r in rows:
            d = dict(r)
            d["properties"] = decode_json(d.get("properties"), {})
            d["is_deleted"] = bool(d.get("is_deleted"))
            result.append(Relation.from_dict(d))
        return result

    # ---------------- episodes ----------------

    def save_episode(self, episode: Episode) -> None:
        self._storage.execute(
            "INSERT INTO episodes(id, started_at, ended_at, transcript, participants, "
            "user_state_before, user_state_after, assistant_state, topics, entities, actions, "
            "outcome, importance, is_consolidated, summary) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "ended_at=excluded.ended_at, transcript=excluded.transcript, "
            "user_state_after=excluded.user_state_after, assistant_state=excluded.assistant_state, "
            "topics=excluded.topics, entities=excluded.entities, actions=excluded.actions, "
            "outcome=excluded.outcome, importance=excluded.importance, "
            "is_consolidated=excluded.is_consolidated, summary=excluded.summary",
            (
                episode.id,
                episode.started_at,
                episode.ended_at,
                encode_json(episode.transcript),
                encode_json(episode.participants),
                encode_json(episode.user_state_before),
                encode_json(episode.user_state_after),
                encode_json(episode.assistant_state),
                encode_json(episode.topics),
                encode_json(episode.entities),
                encode_json(episode.actions),
                episode.outcome,
                episode.importance,
                1 if episode.is_consolidated else 0,
                episode.summary,
            ),
        )

    def get_episode(self, episode_id: str) -> Episode | None:
        row = self._storage.query_one("SELECT * FROM episodes WHERE id=?", (episode_id,))
        return self._episode_from_row(row)

    def list_episodes(self, limit: int = 50, consolidated_only: bool = False) -> list[Episode]:
        sql = "SELECT * FROM episodes"
        params: list = []
        if consolidated_only:
            sql += " WHERE is_consolidated=1"
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        rows = self._storage.query(sql, tuple(params))
        return [e for e in (self._episode_from_row(r) for r in rows) if e is not None]

    def _episode_from_row(self, row) -> Episode | None:
        if row is None:
            return None
        d = dict(row)
        for key in ("transcript", "participants", "user_state_before", "user_state_after",
                    "assistant_state", "topics", "entities", "actions"):
            d[key] = decode_json(d.get(key), default_for(key))
        d["is_consolidated"] = bool(d.get("is_consolidated"))
        return Episode.from_dict(d)

    def add_turn(self, turn_id: str, episode_id: str, role: str, text: str, timestamp: str,
                 source: str = "text", user_state: dict | None = None) -> None:
        self._storage.execute(
            "INSERT INTO turns(id, episode_id, role, text, timestamp, source, user_state) VALUES(?,?,?,?,?,?,?)",
            (turn_id, episode_id, role, text, timestamp, source, encode_json(user_state or {})),
        )

    # ---------------- observations ----------------

    def add_observation(self, observation: Observation) -> None:
        self._storage.execute(
            "INSERT INTO observations(id, kind, payload, episode_id, timestamp, confidence) VALUES(?,?,?,?,?,?)",
            (
                observation.id,
                observation.kind,
                encode_json(observation.payload),
                observation.episode_id,
                observation.timestamp,
                observation.confidence,
            ),
        )

    def list_observations(self, kind: str = "", episode_id: str = "",
                          limit: int = 500) -> list[Observation]:
        """Read back recorded evidence.

        Observations were previously written but never queryable, which made
        provenance unauditable: the chain existed in the database and nothing
        could follow it.
        """
        sql = "SELECT * FROM observations"
        clauses: list[str] = []
        params: list = []
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        if episode_id:
            clauses.append("episode_id=?")
            params.append(episode_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        params.append(limit)
        rows = self._storage.query(sql + " ORDER BY timestamp DESC LIMIT ?", tuple(params))
        out: list[Observation] = []
        for row in rows:
            d = dict(row)
            out.append(Observation(
                id=str(d.get("id", "")),
                kind=str(d.get("kind", "")),
                payload=decode_json(d.get("payload"), {}),
                episode_id=str(d.get("episode_id", "")),
                timestamp=str(d.get("timestamp", "")),
                confidence=float(d.get("confidence", 0.5) or 0.5),
            ))
        return out

    # ---------------- beliefs ----------------

    def upsert_belief(self, belief: Belief) -> None:
        self._storage.execute(
            "INSERT INTO beliefs(id, target_type, target_name, predicate, value, confidence, evidence, "
            "status, created_at, updated_at, importance) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "value=excluded.value, confidence=excluded.confidence, evidence=excluded.evidence, "
            "status=excluded.status, updated_at=excluded.updated_at, importance=excluded.importance",
            (
                belief.id,
                belief.target_type,
                belief.target_name,
                belief.predicate,
                encode_json(belief.value),
                belief.confidence,
                encode_json(belief.evidence),
                belief.status,
                belief.created_at,
                belief.updated_at,
                belief.importance,
            ),
        )

    def list_beliefs(self, target_type: str = "", status: str = "") -> list[Belief]:
        sql = "SELECT * FROM beliefs"
        clauses: list[str] = []
        params: list = []
        if target_type:
            clauses.append("target_type=?")
            params.append(target_type)
        if status:
            clauses.append("status=?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC"
        rows = self._storage.query(sql, tuple(params))
        result: list[Belief] = []
        for r in rows:
            d = dict(r)
            d["value"] = decode_json(d.get("value"), {})
            d["evidence"] = decode_json(d.get("evidence"), [])
            result.append(Belief.from_dict(d))
        return result

    def get_belief(self, belief_id: str) -> Belief | None:
        row = self._storage.query_one("SELECT * FROM beliefs WHERE id=?", (belief_id,))
        if row is None:
            return None
        d = dict(row)
        d["value"] = decode_json(d.get("value"), {})
        d["evidence"] = decode_json(d.get("evidence"), [])
        return Belief.from_dict(d)

    # ---------------- personality ----------------

    def load_profile(self) -> PersonalityProfile:
        profile = PersonalityProfile()
        for row in self._storage.query("SELECT * FROM personality_traits"):
            profile.traits[str(row["name"])] = Trait.from_dict(dict(row))
        for row in self._storage.query("SELECT * FROM personality_values"):
            profile.values[str(row["name"])] = Value.from_dict(dict(row))
        for row in self._storage.query("SELECT * FROM personality_preferences"):
            profile.preferences[str(row["name"])] = Preference.from_dict(dict(row))
        return profile

    def save_profile(self, profile: PersonalityProfile) -> None:
        with self._storage.transaction():
            for name, trait in profile.traits.items():
                self._storage.execute(
                    "INSERT INTO personality_traits(name, value, confidence, stability, evidence_count, stability_class, updated_at) "
                    "VALUES(?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                    "value=excluded.value, confidence=excluded.confidence, stability=excluded.stability, "
                    "evidence_count=excluded.evidence_count, stability_class=excluded.stability_class, updated_at=excluded.updated_at",
                    (name, trait.value, trait.confidence, trait.stability, trait.evidence_count,
                     trait.stability_class.value, trait.updated_at),
                )
            for name, value in profile.values.items():
                self._storage.execute(
                    "INSERT INTO personality_values(name, importance, confidence, stability, updated_at) "
                    "VALUES(?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                    "importance=excluded.importance, confidence=excluded.confidence, stability=excluded.stability, updated_at=excluded.updated_at",
                    (name, value.importance, value.confidence, value.stability, value.updated_at),
                )
            for name, pref in profile.preferences.items():
                self._storage.execute(
                    "INSERT INTO personality_preferences(name, value, confidence, stability, evidence_count, updated_at) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                    "value=excluded.value, confidence=excluded.confidence, stability=excluded.stability, "
                    "evidence_count=excluded.evidence_count, updated_at=excluded.updated_at",
                    (name, pref.value, pref.confidence, pref.stability, pref.evidence_count, pref.updated_at),
                )

    def add_personality_evidence(self, evidence: PersonalityEvidence) -> None:
        self._storage.execute(
            "INSERT INTO personality_evidence(id, target, direction, strength, confidence, source_episode, "
            "source, timestamp, context, kind) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                evidence.id,
                evidence.target,
                evidence.direction,
                evidence.strength,
                evidence.confidence,
                evidence.source_episode,
                evidence.source,
                evidence.timestamp,
                evidence.context,
                evidence.kind,
            ),
        )

    def list_personality_evidence(self, target: str = "", limit: int = 200) -> list[PersonalityEvidence]:
        if target:
            rows = self._storage.query(
                "SELECT * FROM personality_evidence WHERE target=? ORDER BY timestamp DESC LIMIT ?",
                (target, limit),
            )
        else:
            rows = self._storage.query(
                "SELECT * FROM personality_evidence ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
        return [PersonalityEvidence.from_dict(dict(r)) for r in rows]

    # ---------------- contradictions ----------------

    def add_contradiction(self, contradiction: Contradiction) -> None:
        self._storage.execute(
            "INSERT INTO contradictions(id, statement_a, statement_b, subject, predicate, contexts, timestamps, resolution_status) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                contradiction.id,
                contradiction.statement_a,
                contradiction.statement_b,
                contradiction.subject,
                contradiction.predicate,
                encode_json(contradiction.contexts),
                encode_json(contradiction.timestamps),
                contradiction.resolution_status,
            ),
        )

    def list_contradictions(self, status: str = "") -> list[Contradiction]:
        sql = "SELECT * FROM contradictions"
        params: list = []
        if status:
            sql += " WHERE resolution_status=?"
            params.append(status)
        rows = self._storage.query(sql + " ORDER BY timestamps DESC", tuple(params))
        result: list[Contradiction] = []
        for r in rows:
            d = dict(r)
            d["contexts"] = decode_json(d.get("contexts"), [])
            d["timestamps"] = decode_json(d.get("timestamps"), [])
            result.append(Contradiction(**d))
        return result

    def resolve_contradiction(self, contradiction_id: str, status: str) -> None:
        self._storage.execute(
            "UPDATE contradictions SET resolution_status=? WHERE id=?", (status, contradiction_id)
        )

    # ---------------- goals ----------------

    def upsert_goal(self, goal: Goal) -> None:
        self._storage.execute(
            "INSERT INTO goals(id, name, description, status, priority, progress, confidence, source_episode_id, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "name=excluded.name, description=excluded.description, status=excluded.status, "
            "priority=excluded.priority, progress=excluded.progress, confidence=excluded.confidence, "
            "updated_at=excluded.updated_at",
            (
                goal.id, goal.name, goal.description, goal.status, goal.priority, goal.progress,
                goal.confidence, goal.source_episode_id, goal.created_at, goal.updated_at,
            ),
        )

    def list_goals(self, status: str = "active") -> list[Goal]:
        sql = "SELECT * FROM goals"
        params: list = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        rows = self._storage.query(sql + " ORDER BY priority DESC", tuple(params))
        return [Goal.from_dict(dict(r)) for r in rows]

    # ---------------- relationships ----------------

    def upsert_relationship(self, relationship: Relationship) -> None:
        self._storage.execute(
            "INSERT INTO relationships(id, subject_id, target_id, type, name, trust, familiarity, "
            "emotional_valence, interaction_count, last_interaction, important_events, confidence, created_at, updated_at, notes) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "trust=excluded.trust, familiarity=excluded.familiarity, emotional_valence=excluded.emotional_valence, "
            "interaction_count=excluded.interaction_count, last_interaction=excluded.last_interaction, "
            "important_events=excluded.important_events, confidence=excluded.confidence, "
            "updated_at=excluded.updated_at, notes=excluded.notes",
            (
                relationship.id, relationship.subject_id, relationship.target_id, relationship.type,
                relationship.name, relationship.trust, relationship.familiarity,
                relationship.emotional_valence, relationship.interaction_count,
                relationship.last_interaction, encode_json(relationship.important_events),
                relationship.confidence, relationship.created_at, relationship.updated_at,
                relationship.notes,
            ),
        )

    def list_relationships(self) -> list[Relationship]:
        rows = self._storage.query("SELECT * FROM relationships ORDER BY familiarity DESC")
        result: list[Relationship] = []
        for r in rows:
            d = dict(r)
            d["important_events"] = decode_json(d.get("important_events"), [])
            result.append(Relationship.from_dict(d))
        return result

    def get_relationship_for(self, target_id: str) -> Relationship | None:
        row = self._storage.query_one(
            "SELECT * FROM relationships WHERE target_id=? ORDER BY familiarity DESC LIMIT 1",
            (target_id,),
        )
        if row is None:
            return None
        d = dict(row)
        d["important_events"] = decode_json(d.get("important_events"), [])
        return Relationship.from_dict(d)

    # ---------------- sources / knowledge ----------------

    def get_or_create_source(self, source: Source) -> Source:
        if source.id:
            existing = self._storage.query_one("SELECT * FROM sources WHERE id=?", (source.id,))
            if existing is not None:
                return Source.from_dict(dict(existing))
        row = self._storage.query_one(
            "SELECT * FROM sources WHERE uri=? AND type=?", (source.uri, source.type.value)
        )
        if row is not None:
            return Source.from_dict(dict(row))
        self._storage.execute(
            "INSERT INTO sources(id, type, name, uri, created_at) VALUES(?,?,?,?,?)",
            (source.id, source.type.value, source.name, source.uri, source.created_at),
        )
        return source

    def add_knowledge(self, chunk: KnowledgeChunk) -> None:
        self._storage.execute(
            "INSERT INTO knowledge(id, source_id, source_type, content, title, confidence, embedding_id, created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                chunk.id, chunk.source_id, chunk.source_type.value, chunk.content,
                chunk.title, chunk.confidence, chunk.embedding_id, chunk.created_at,
            ),
        )

    def list_knowledge(self, source_type: str = "", limit: int = 100) -> list[KnowledgeChunk]:
        sql = "SELECT * FROM knowledge"
        params: list = []
        if source_type:
            sql += " WHERE source_type=?"
            params.append(source_type)
        rows = self._storage.query(sql + " ORDER BY created_at DESC LIMIT ?", tuple(params + [limit]))
        result: list[KnowledgeChunk] = []
        for r in rows:
            d = dict(r)
            d["source_type"] = d.get("source_type", "document")
            result.append(KnowledgeChunk(**d))
        return result

    # ---------------- memories ----------------

    def add_memory(self, memory: Memory) -> None:
        self._storage.execute(
            "INSERT INTO memories(id, type, content, importance, confidence, status, created_at, updated_at, "
            "accessed_at, retrieval_count, source_episode_id, embedding_id, locked, meta) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "importance=excluded.importance, confidence=excluded.confidence, status=excluded.status, "
            "updated_at=excluded.updated_at, meta=excluded.meta",
            (
                memory.id, memory.type.value, memory.content, memory.importance, memory.confidence,
                memory.status.value, memory.created_at, memory.updated_at, memory.accessed_at,
                memory.retrieval_count, memory.source_episode_id, memory.embedding_id,
                1 if memory.locked else 0, encode_json(memory.meta),
            ),
        )

    def get_memory(self, memory_id: str) -> Memory | None:
        row = self._storage.query_one("SELECT * FROM memories WHERE id=?", (memory_id,))
        if row is None:
            return None
        d = dict(row)
        d["locked"] = bool(d.get("locked"))
        d["meta"] = decode_json(d.get("meta"), {})
        return Memory.from_dict(d)

    def update_memory_status(self, memory_id: str, status: str) -> None:
        self._storage.execute(
            "UPDATE memories SET status=?, updated_at=? WHERE id=?",
            (status, self._now(), memory_id),
        )

    def update_memory_access(self, memory_id: str, at: str) -> None:
        self._storage.execute(
            "UPDATE memories SET accessed_at=?, retrieval_count=retrieval_count+1 WHERE id=?",
            (at, memory_id),
        )

    def set_memory_locked(self, memory_id: str, locked: bool) -> None:
        self._storage.execute(
            "UPDATE memories SET locked=? WHERE id=?", (1 if locked else 0, memory_id)
        )

    def forget_memory(self, memory_id: str) -> None:
        self._storage.execute(
            "UPDATE memories SET status='forgotten', updated_at=? WHERE id=?",
            (self._now(), memory_id),
        )

    def list_memories(self, status: str = "", type: str = "", limit: int = 200) -> list[Memory]:
        sql = "SELECT * FROM memories"
        clauses: list[str] = []
        params: list = []
        if status:
            clauses.append("status=?")
            params.append(status)
        else:
            clauses.append("status != 'forgotten'")
        if type:
            clauses.append("type=?")
            params.append(type)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY importance DESC LIMIT ?"
        rows = self._storage.query(sql, tuple(params + [limit]))
        result: list[Memory] = []
        for r in rows:
            d = dict(r)
            d["locked"] = bool(d.get("locked"))
            d["meta"] = decode_json(d.get("meta"), {})
            result.append(Memory.from_dict(d))
        return result

    # ---------------- system state ----------------

    def get_system_state(self, key: str) -> str | None:
        row = self._storage.query_one("SELECT value FROM system_state WHERE key=?", (key,))
        return str(row["value"]) if row is not None else None

    def set_system_state(self, key: str, value: str) -> None:
        self._storage.execute(
            "INSERT INTO system_state(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, self._now()),
        )

    @staticmethod
    def _now() -> str:
        from companion.core.clock import SystemClock

        return SystemClock().now_iso()

    def storage_handle(self) -> SqliteStorage:
        return self._storage


def default_for(key: str):
    if key in ("transcript", "participants", "topics", "entities", "actions"):
        return []
    return {}
