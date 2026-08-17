"""Turn-level salient-fact commit.

The original pipeline extracted memories only when an episode was *closed*.
That made durability hostage to a graceful shutdown: Ctrl+C, a crash, or a
power cut lost everything said since the session began, and the acceptance
tests ("tell it something, exit, restart, ask") could never pass.

So durability is split in two:

* **This module — synchronous, deterministic, cheap.** On every user turn,
  explicit high-salience statements (identities, preferences, corrections,
  goals, "remember that ...") are written to the graph immediately. No model
  call, single-digit milliseconds, safe to run inline on the turn.
* **Episode consolidation — LLM-driven, background.** Richer extraction,
  dedup, contradiction analysis and summarisation, which may take seconds and
  is allowed to be interrupted.

The rule extractor here is intentionally narrow. It only fires on statements
whose meaning is unambiguous in the surface form, because anything it writes
is committed with high authority. Everything subtler is left to consolidation,
where it lands with lower confidence and is subject to review.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from companion.application.facts import (
    AUTHORITY_EXPLICIT_USER,
    AUTHORITY_USER_STATEMENT,
    FactWriter,
)
from companion.application.identity import (
    SelfModelService,
    detect_identity_statement,
)
from companion.core.clock import Clock, SystemClock
from companion.core.ids import new_goal_id, new_memory_id
from companion.domain.graph import Goal
from companion.domain.memory import Memory, MemoryStatus, MemoryType

log = logging.getLogger(__name__)


@dataclass
class SalientFact:
    predicate: str
    value: str = ""
    object_name: str = ""
    kind: str = "semantic"        # semantic | preference | goal | note
    confidence: float = 0.8
    importance: float = 0.5
    authority: float = AUTHORITY_USER_STATEMENT
    supersede: bool = True
    utterance: str = ""

    def slot(self) -> str:
        return self.predicate


@dataclass
class TurnCommitResult:
    facts: list = field(default_factory=list)
    goals: list = field(default_factory=list)
    memories: list = field(default_factory=list)
    identity: dict | None = None
    retracted: list = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.facts or self.goals or self.memories or self.identity or self.retracted)

    def to_dict(self) -> dict:
        return {
            "facts": [f.to_dict() if hasattr(f, "to_dict") else f for f in self.facts],
            "goals": list(self.goals),
            "memories": list(self.memories),
            "identity": self.identity,
            "retracted": list(self.retracted),
        }


_SENTENCE = re.compile(r"[^.!?;\n]+[.!?;]?")

# "my favourite <category> is <value>"
_FAVOURITE = re.compile(
    r"\bmy\s+(?:favou?rite|preferred)\s+([a-z][a-z ]{1,24}?)\s+(?:is|are|=)\s+(.+)",
    re.IGNORECASE,
)
# "it's blue now" / "now it's blue" -> refers to the slot last discussed
_SLOT_UPDATE = re.compile(
    r"^\s*(?:and\s+)?(?:it'?s|it\s+is|now\s+it'?s|make\s+it|change\s+it\s+to)\s+"
    r"(.{1,40}?)\s*(?:now)?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_NEGATED_PREFERENCE = re.compile(
    r"\bi\s+(?:don'?t|do\s+not|no\s+longer)\s+(?:like|love|enjoy|prefer)\s+(.+?)"
    r"(?:\s+any\s?more)?[.!]?$",
    re.IGNORECASE,
)
_STOPPED_LIKING = re.compile(
    r"\bi\s+(?:stopped|quit|gave\s+up)\s+(?:liking|loving|using|doing)\s+(.+)",
    re.IGNORECASE,
)
_LIKES = re.compile(r"\bi\s+(?:really\s+)?(?:like|love|enjoy)\s+(.+)", re.IGNORECASE)
_DISLIKES = re.compile(r"\bi\s+(?:really\s+)?(?:hate|dislike|can'?t\s+stand)\s+(.+)",
                       re.IGNORECASE)
_PREFERS = re.compile(r"\bi\s+prefer\s+(.+)", re.IGNORECASE)
_WORKS_ON = re.compile(
    r"\bi\s?(?:'m|\s+am)?\s*(?:currently\s+)?(?:working\s+on|building|developing)\s+"
    r"(?:a\s+|an\s+|the\s+)?(?:project\s+(?:called|named)\s+)?(.+)",
    re.IGNORECASE,
)
_LIVES_IN = re.compile(r"\bi\s+(?:live|am\s+based)\s+in\s+(.+)", re.IGNORECASE)
_REMEMBER = re.compile(
    r"\b(?:remember|note|keep\s+in\s+mind|don'?t\s+forget)\s+(?:that\s+)?(.+)",
    re.IGNORECASE,
)
_WANTS = re.compile(
    r"\bi\s+(?:want|need|would\s+like|plan|intend)\s+to\s+(.+)", re.IGNORECASE,
)
_FRUSTRATION = re.compile(
    r"\bi(?:'ve|\s+have)?\s+(?:been\s+)?(?:feel(?:ing)?\s+)?"
    r"(frustrated|annoyed|stuck|blocked|confused|tired|exhausted)\s+"
    r"(?:with|by|about|on)\s+(.+)",
    re.IGNORECASE,
)

_FORGET = re.compile(
    r"\b(?:forget|erase|delete)\s+(?:that\s+|what\s+i\s+said\s+about\s+)?(.+)",
    re.IGNORECASE,
)

_TRAILING = " .!?,;:\"'"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(_TRAILING)).strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text or "") if s.strip()]


class SalientExtractor:
    """Deterministic extraction of unambiguous, high-authority statements."""

    def extract(self, text: str, last_slot: str = "") -> tuple[list[SalientFact], list[str]]:
        """Return (facts, retracted_slots) for one user utterance."""
        facts: list[SalientFact] = []
        retract: list[str] = []
        for sentence in _sentences(text):
            s = sentence.strip()
            lowered = s.lower()
            if lowered.endswith("?"):
                continue  # questions assert nothing

            m = _FAVOURITE.search(s)
            if m:
                category = _clean(m.group(1)).lower().replace(" ", "_")
                value = _clean(m.group(2))
                if category and value:
                    facts.append(SalientFact(
                        predicate=f"favorite:{category}", value=value, kind="preference",
                        confidence=0.95, importance=0.6, utterance=s,
                    ))
                continue

            m = _NEGATED_PREFERENCE.search(s) or _STOPPED_LIKING.search(s)
            if m:
                obj = _clean(m.group(1))
                if obj:
                    facts.append(SalientFact(
                        predicate=f"opinion:{obj.lower()}", value="dislikes",
                        kind="preference", confidence=0.9, importance=0.55,
                        authority=AUTHORITY_EXPLICIT_USER, utterance=s,
                    ))
                    # If the thing they now reject is the current value of the
                    # slot we were just discussing, that slot is no longer true.
                    if last_slot:
                        retract.append(f"{last_slot}|{obj.lower()}")
                continue

            m = _SLOT_UPDATE.match(s)
            if m and last_slot:
                value = _clean(m.group(1))
                if value and value.lower() not in ("it", "that", "this"):
                    facts.append(SalientFact(
                        predicate=last_slot, value=value, kind="preference",
                        confidence=0.9, importance=0.6, utterance=s,
                    ))
                continue

            m = _FRUSTRATION.search(s)
            if m:
                feeling, about = m.group(1).lower(), _clean(m.group(2))
                if about:
                    facts.append(SalientFact(
                        predicate=f"experience:{feeling}", value=about, kind="note",
                        confidence=0.85, importance=0.55, supersede=False, utterance=s,
                    ))
                continue

            m = _REMEMBER.search(s)
            if m:
                content = _clean(m.group(1))
                if content:
                    facts.append(SalientFact(
                        predicate="explicit_memory", value=content, kind="note",
                        confidence=0.95, importance=0.8, supersede=False,
                        authority=AUTHORITY_EXPLICIT_USER, utterance=s,
                    ))
                continue

            m = _WORKS_ON.search(s)
            if m:
                obj = _clean(m.group(1))
                if obj:
                    facts.append(SalientFact(
                        predicate="works_on", object_name=obj, kind="semantic",
                        confidence=0.9, importance=0.7, supersede=False, utterance=s,
                    ))
                continue

            m = _WANTS.search(s)
            if m:
                obj = _clean(m.group(1))
                if obj:
                    facts.append(SalientFact(
                        predicate="goal", value=obj, kind="goal",
                        confidence=0.85, importance=0.65, supersede=False, utterance=s,
                    ))
                continue

            m = _LIVES_IN.search(s)
            if m:
                obj = _clean(m.group(1))
                if obj:
                    facts.append(SalientFact(
                        predicate="lives_in", value=obj, kind="semantic",
                        confidence=0.9, importance=0.6, utterance=s,
                    ))
                continue

            m = _PREFERS.search(s)
            if m:
                obj = _clean(m.group(1))
                if obj:
                    facts.append(SalientFact(
                        predicate=f"prefers:{obj.lower()[:40]}", value="yes",
                        kind="preference", confidence=0.85, importance=0.55, utterance=s,
                    ))
                continue

            m = _DISLIKES.search(s)
            if m:
                obj = _clean(m.group(1))
                if obj:
                    facts.append(SalientFact(
                        predicate=f"opinion:{obj.lower()}", value="dislikes",
                        kind="preference", confidence=0.85, importance=0.5, utterance=s,
                    ))
                continue

            m = _LIKES.search(s)
            if m:
                obj = _clean(m.group(1))
                if obj:
                    facts.append(SalientFact(
                        predicate=f"opinion:{obj.lower()}", value="likes",
                        kind="preference", confidence=0.85, importance=0.5, utterance=s,
                    ))
                continue
        return facts, retract


class TurnCommitter:
    """Writes salient facts from a single user turn straight to the graph."""

    def __init__(self, graph, self_model: SelfModelService,
                 writer: FactWriter | None = None, clock: Clock | None = None,
                 extractor: SalientExtractor | None = None) -> None:
        self._graph = graph
        self._self = self_model
        self._clock = clock or SystemClock()
        self._writer = writer or FactWriter(graph, self._clock)
        self._extractor = extractor or SalientExtractor()
        self._last_slot = ""

    def reset_context(self) -> None:
        self._last_slot = ""

    def commit(self, text: str, episode_id: str = "") -> TurnCommitResult:
        result = TurnCommitResult()
        if not (text or "").strip():
            return result

        statement = detect_identity_statement(text)
        if statement is not None:
            outcome = self._self.apply_statement(statement, episode_id=episode_id)
            outcome["target"] = statement.target
            outcome["hedged"] = statement.hedged
            result.identity = outcome

        try:
            user_id = self._self.user_entity_id()
        except Exception as exc:
            log.warning("cannot resolve user entity, skipping turn commit: %s", exc)
            return result

        # Resolve what "it" refers to before extracting. In-process memory of
        # the last slot is not enough: "I don't like purple anymore. It's blue
        # now." usually arrives in a *later session* than the statement it
        # corrects, so the referent is recovered from the graph by value.
        last_slot = self._referent_slot(user_id, text) or self._last_slot \
            or self._recover_last_slot(user_id)
        facts, retractions = self._extractor.extract(text, last_slot=last_slot)
        if last_slot:
            self._last_slot = last_slot
        for spec in retractions:
            slot, obj = spec.split("|", 1)
            current = self._writer.current(user_id, slot)
            if current is not None and (current.value or "").strip().lower() == obj:
                closed = self._writer.retract(user_id, slot, evidence_text=text,
                                              source_episode_id=episode_id)
                result.retracted.extend(closed)

        for fact in facts:
            if fact.kind == "goal":
                result.goals.append(self._commit_goal(fact, episode_id))
                continue
            if fact.predicate == "explicit_memory":
                result.memories.append(self._commit_memory(fact, episode_id))
                continue
            object_id = None
            if fact.object_name:
                object_id = self._writer.resolve_object_entity(fact.object_name)
            assertion = self._writer.assert_fact(
                subject_id=user_id,
                predicate=fact.predicate,
                value=fact.value,
                object_id=object_id,
                confidence=fact.confidence,
                importance=fact.importance,
                authority=fact.authority,
                provenance="explicit_user_statement"
                if fact.authority >= AUTHORITY_EXPLICIT_USER else "conversation",
                source_episode_id=episode_id,
                evidence_text=fact.utterance,
                supersede=fact.supersede,
            )
            if assertion.fact is not None:
                result.facts.append(assertion)
                # Also mirror durable preferences into the memory table so the
                # existing retrieval paths (lexical/semantic) can see them.
                self._mirror_memory(fact, episode_id, assertion)
                if fact.supersede and fact.predicate.startswith(("favorite:", "opinion:")):
                    self._last_slot = fact.predicate
        return result

    _SUPERSEDABLE = ("favorite:", "prefers:", "lives_in")

    def _referent_slot(self, user_id: str, text: str) -> str:
        """Find the slot the utterance is talking about, by matching values.

        "I don't like purple anymore" names *purple*, not the slot. If purple
        is the current value of ``favorite:color``, that is the slot being
        corrected — and this works no matter how long ago it was set.
        """
        lowered = f" {(text or '').lower()} "
        best = ""
        best_len = 0
        try:
            facts = self._graph.list_facts(subject_id=user_id) or []
        except Exception:
            return ""
        for fact in facts:
            if fact.valid_to or not fact.value:
                continue
            if not fact.predicate.startswith(self._SUPERSEDABLE):
                continue
            value = fact.value.strip().lower()
            if value and re.search(rf"\b{re.escape(value)}\b", lowered):
                if len(value) > best_len:
                    best, best_len = fact.predicate, len(value)
        return best

    def _recover_last_slot(self, user_id: str) -> str:
        """The most recently set supersedable slot, for bare 'it's X now'."""
        try:
            facts = [f for f in (self._graph.list_facts(subject_id=user_id) or [])
                     if not f.valid_to and f.predicate.startswith(self._SUPERSEDABLE)]
        except Exception:
            return ""
        if not facts:
            return ""
        return max(facts, key=lambda f: f.created_at).predicate

    def _commit_goal(self, fact: SalientFact, episode_id: str) -> str:
        name = fact.value[:120]
        for existing in self._graph.list_goals(status="active") or []:
            if existing.name.lower() == name.lower():
                return existing.id
        goal = Goal(
            id=new_goal_id(), name=name, description=fact.utterance,
            status="active", priority=0.6, progress=0.0, confidence=fact.confidence,
            source_episode_id=episode_id,
            created_at=self._clock.now_iso(), updated_at=self._clock.now_iso(),
        )
        self._graph.upsert_goal(goal)
        return goal.id

    def _commit_memory(self, fact: SalientFact, episode_id: str) -> str:
        memory = Memory(
            id=new_memory_id(),
            type=MemoryType.SEMANTIC,
            content=fact.value,
            importance=fact.importance,
            confidence=fact.confidence,
            status=MemoryStatus.VALIDATED,
            created_at=self._clock.now_iso(),
            updated_at=self._clock.now_iso(),
            source_episode_id=episode_id,
            meta={"source": "explicit_user_request", "utterance": fact.utterance},
        )
        self._graph.add_memory(memory)
        return memory.id

    def _mirror_memory(self, fact: SalientFact, episode_id: str, assertion) -> None:
        """Keep a readable memory row alongside the structured fact.

        The fact is the source of truth; this row exists so lexical and
        semantic retrieval — which search prose — can surface it before any
        embedding job has run.
        """
        content = _readable(fact)
        if not content:
            return
        memory = Memory(
            id=new_memory_id(),
            type=MemoryType.PREFERENCE if fact.kind == "preference" else MemoryType.SEMANTIC,
            content=content,
            importance=fact.importance,
            confidence=fact.confidence,
            status=MemoryStatus.VALIDATED,
            created_at=self._clock.now_iso(),
            updated_at=self._clock.now_iso(),
            source_episode_id=episode_id,
            meta={"fact_id": assertion.fact.id if assertion.fact else "",
                  "predicate": fact.predicate},
        )
        self._graph.add_memory(memory)
        # Superseded facts leave stale mirrors behind; archive them so the old
        # value cannot be retrieved as if it were still true.
        for old_id in assertion.superseded:
            for mem in self._graph.list_memories(limit=500) or []:
                if mem.meta.get("fact_id") == old_id:
                    self._graph.update_memory_status(mem.id, MemoryStatus.ARCHIVED.value)


def _readable(fact: SalientFact) -> str:
    """Render a fact as the sentence a human would use to state it."""
    predicate, value = fact.predicate, fact.value
    if predicate.startswith("favorite:"):
        return f"user's favorite {predicate.split(':', 1)[1].replace('_', ' ')} is {value}"
    if predicate.startswith("opinion:"):
        obj = predicate.split(":", 1)[1]
        return f"user {'likes' if value == 'likes' else 'dislikes'} {obj}"
    if predicate.startswith("prefers:"):
        return f"user prefers {predicate.split(':', 1)[1]}"
    if predicate.startswith("experience:"):
        return f"user has been {predicate.split(':', 1)[1]} with {value}"
    if predicate == "works_on":
        return f"user is working on {fact.object_name}"
    if predicate == "lives_in":
        return f"user lives in {value}"
    return f"user {predicate.replace('_', ' ')} {value}".strip()
