"""FactWriter: the single place that writes durable facts to the graph.

Every durable claim in the system goes through here so that four properties
hold uniformly, instead of being re-implemented (and forgotten) per caller:

* **Supersession, not overwrite.** Asserting a new value for a slot that
  already has one closes the old fact (``valid_to = now``) and inserts a new
  row. History stays queryable — "what *used* to be true" is a real query.
* **Provenance.** Who said it, in which episode, from which source class, and
  with what authority. An inference never silently becomes an observation.
* **Confirmation.** Re-asserting the same value does not duplicate the fact;
  it stamps ``last_confirmed_at`` and nudges confidence up.
* **Evidence.** Every assertion also records an Observation, so
  ``memory why`` can reconstruct the chain later.

A "slot" is the (subject, predicate) pair. Predicates that name a category
encode it — ``favorite:color`` rather than ``has_favorite`` — so that setting a
favorite colour does not invalidate a favorite food.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from companion.application.ports import GraphStore
from companion.core.clock import Clock, SystemClock
from companion.core.ids import new_fact_id, new_observation_id
from companion.domain.graph import Entity, Fact, Observation, Source, SourceType

log = logging.getLogger(__name__)


# Authority ranking for competing claims about the same slot. A hedged guess
# must never overwrite something the user stated outright.
AUTHORITY_EXPLICIT_USER = 1.0      # "your name is Jarvis"
AUTHORITY_USER_STATEMENT = 0.85    # "my favourite colour is purple"
AUTHORITY_PERSISTED = 0.7          # already-validated memory
AUTHORITY_INFERENCE = 0.45         # model-inferred, needs corroboration
AUTHORITY_HEDGED = 0.25            # "I think your name might be Bob"


@dataclass
class FactAssertion:
    """Result of a write, so callers can report what actually happened."""

    fact: Fact | None
    created: bool = False
    confirmed: bool = False
    superseded: list[str] = None  # ids of facts closed by this assertion

    def __post_init__(self) -> None:
        if self.superseded is None:
            self.superseded = []

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact.id if self.fact else "",
            "created": self.created,
            "confirmed": self.confirmed,
            "superseded": list(self.superseded),
        }


class FactWriter:
    def __init__(self, graph: GraphStore, clock: Clock | None = None) -> None:
        self._graph = graph
        self._clock = clock or SystemClock()

    # -- entities ---------------------------------------------------------

    def entity_id(self, key: str, *, name: str, type: str, summary: str = "") -> str:
        """Resolve a canonical entity id held in system_state, creating it once.

        Used for the two singleton entities the cognitive system pivots on:
        ``primary_user_entity`` and ``agent_self_entity``.
        """
        existing = self._graph.get_system_state(key)
        if existing:
            if self._graph.get_entity(existing) is not None:
                return existing
            log.warning("system_state %s points at a missing entity; recreating", key)
        entity = Entity(
            type=type,
            name=name,
            summary=summary,
            confidence=1.0,
            importance=0.95,
            created_at=self._clock.now_iso(),
            updated_at=self._clock.now_iso(),
            valid_from=self._clock.now_iso(),
        )
        self._graph.upsert_entity(entity)
        self._graph.set_system_state(key, entity.id)
        return entity.id

    def resolve_object_entity(self, name: str, type: str = "thing") -> str:
        """Find or create the entity a fact points at."""
        name = (name or "").strip()
        if not name:
            return ""
        found = self._graph.find_entity_by_name(name)
        if found is not None:
            return found.id
        entity = Entity(
            type=type,
            name=name,
            confidence=0.6,
            importance=0.4,
            created_at=self._clock.now_iso(),
            updated_at=self._clock.now_iso(),
            valid_from=self._clock.now_iso(),
        )
        self._graph.upsert_entity(entity)
        return entity.id

    # -- assertions -------------------------------------------------------

    def assert_fact(
        self,
        *,
        subject_id: str,
        predicate: str,
        value: str = "",
        object_id: str | None = None,
        confidence: float = 0.6,
        importance: float = 0.4,
        authority: float = AUTHORITY_USER_STATEMENT,
        provenance: str = SourceType.CONVERSATION.value,
        source_episode_id: str = "",
        evidence_text: str = "",
        supersede: bool = True,
        permanent: bool = False,
    ) -> FactAssertion:
        """Assert that ``subject predicate value`` holds as of now.

        Returns what happened rather than raising, because a refused write
        (lower authority than the incumbent) is a normal outcome, not an error.
        """
        if not subject_id or not predicate:
            return FactAssertion(fact=None)
        now = self._clock.now_iso()
        value = (value or "").strip()

        incumbents = [
            f for f in self._graph.list_facts(subject_id=subject_id, predicate=predicate)
            if not f.valid_to
        ]

        # Same value already on record -> confirm, do not duplicate.
        for fact in incumbents:
            if _same_value(fact, value, object_id):
                self._graph.confirm_fact(fact.id, now)
                fact.last_confirmed_at = now
                # Independent re-statement is corroboration: move confidence a
                # fraction of the way toward certainty, never all the way.
                raised = fact.confidence + (1.0 - fact.confidence) * 0.25 * confidence
                fact.confidence = min(0.99, max(fact.confidence, raised))
                self._graph.set_fact_confidence(fact.id, fact.confidence)
                self._record_evidence(fact.id, evidence_text, source_episode_id,
                                      confidence, kind="fact_confirmed")
                return FactAssertion(fact=fact, confirmed=True)

        # A weaker claim must not displace a stronger incumbent.
        if incumbents and supersede:
            strongest = max(_authority_of(f) for f in incumbents)
            if authority < strongest:
                log.info(
                    "refusing to supersede %s/%s: authority %.2f < incumbent %.2f",
                    subject_id, predicate, authority, strongest,
                )
                self._record_evidence("", evidence_text, source_episode_id, confidence,
                                      kind="fact_rejected_low_authority")
                return FactAssertion(fact=None)

        superseded: list[str] = []
        if supersede:
            for fact in incumbents:
                self._graph.invalidate_fact(fact.id, now)
                superseded.append(fact.id)

        source = self._graph.get_or_create_source(
            Source(type=_source_type(provenance),
                   name=f"episode:{source_episode_id}" if source_episode_id else provenance,
                   uri=source_episode_id, created_at=now)
        )
        fact = Fact(
            id=new_fact_id(),
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            value=None if object_id else (value or None),
            confidence=max(0.0, min(1.0, confidence)),
            importance=max(0.0, min(1.0, 1.0 if permanent else importance)),
            created_at=now,
            valid_from=now,
            valid_to="",
            source_episode_id=source_episode_id,
            source_id=getattr(source, "id", "") or "",
            last_confirmed_at=now,
            provenance=provenance,
        )
        self._graph.add_fact(fact)
        self._record_evidence(fact.id, evidence_text, source_episode_id, confidence,
                              kind="fact_asserted", superseded=superseded)
        log.info("fact asserted %s -%s-> %s (conf %.2f, authority %.2f, superseded %d)",
                 subject_id, predicate, value or object_id, confidence, authority,
                 len(superseded))
        return FactAssertion(fact=fact, created=True, superseded=superseded)

    def retract(self, subject_id: str, predicate: str, *, evidence_text: str = "",
                source_episode_id: str = "") -> list[str]:
        """Close every active fact in a slot without asserting a replacement.

        Used for "I don't like X anymore" where no new value is supplied. The
        rows stay in the database with a ``valid_to``; they are no longer true,
        but they remain part of the record.
        """
        now = self._clock.now_iso()
        closed: list[str] = []
        for fact in self._graph.list_facts(subject_id=subject_id, predicate=predicate):
            if fact.valid_to:
                continue
            self._graph.invalidate_fact(fact.id, now)
            closed.append(fact.id)
            self._record_evidence(fact.id, evidence_text, source_episode_id, 0.9,
                                  kind="fact_retracted")
        return closed

    # -- queries ----------------------------------------------------------

    def current(self, subject_id: str, predicate: str) -> Fact | None:
        facts = [f for f in self._graph.list_facts(subject_id=subject_id, predicate=predicate)
                 if not f.valid_to]
        if not facts:
            return None
        return max(facts, key=lambda f: (f.confidence, f.created_at))

    def history(self, subject_id: str, predicate: str) -> list[Fact]:
        """Every value this slot has held, newest first, including closed ones."""
        facts = self._graph.list_facts(subject_id=subject_id, predicate=predicate,
                                       include_deleted=True)
        return sorted(facts, key=lambda f: f.created_at, reverse=True)

    def previous(self, subject_id: str, predicate: str) -> Fact | None:
        """The most recently superseded value, for 'what used to be...' queries."""
        closed = [f for f in self.history(subject_id, predicate) if f.valid_to]
        if not closed:
            return None
        return max(closed, key=lambda f: f.valid_to)

    # -- internals --------------------------------------------------------

    def _record_evidence(self, fact_id: str, text: str, episode_id: str,
                         confidence: float, kind: str,
                         superseded: list[str] | None = None) -> None:
        try:
            self._graph.add_observation(
                Observation(
                    id=new_observation_id(),
                    kind=kind,
                    payload={
                        "fact_id": fact_id,
                        "utterance": text[:400],
                        "superseded": superseded or [],
                    },
                    episode_id=episode_id,
                    timestamp=self._clock.now_iso(),
                    confidence=confidence,
                )
            )
        except Exception as exc:  # evidence must never break the write
            log.warning("could not record fact evidence: %s", exc)


def _same_value(fact: Fact, value: str, object_id: str | None) -> bool:
    if object_id:
        return fact.object_id == object_id
    return (fact.value or "").strip().lower() == (value or "").strip().lower()


def _authority_of(fact: Fact) -> float:
    """Recover the authority of a stored fact from its provenance."""
    return {
        "explicit_user_statement": AUTHORITY_EXPLICIT_USER,
        SourceType.USER.value: AUTHORITY_USER_STATEMENT,
        SourceType.CONVERSATION.value: AUTHORITY_USER_STATEMENT,
        SourceType.DOCUMENT.value: AUTHORITY_INFERENCE,
        SourceType.MODEL_INFERENCE.value: AUTHORITY_INFERENCE,
        "hedged": AUTHORITY_HEDGED,
        SourceType.SYSTEM.value: AUTHORITY_PERSISTED,
    }.get(fact.provenance, AUTHORITY_INFERENCE)


def _source_type(provenance: str) -> SourceType:
    try:
        return SourceType(provenance)
    except ValueError:
        return SourceType.CONVERSATION
