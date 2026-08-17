"""Identity: who the agent is, who the user is, and how that survives restart.

Configuration is a *bootstrap*, not an identity. The moment the user says
"your name is Jarvis" that becomes a durable fact on the ``self`` entity, and
from then on the persisted name outranks whatever the YAML says. Renaming
closes the old fact instead of erasing it, so "what was your previous name?"
is answerable.

Identity authority, highest first (see docs/identity.md):

1. an explicit user statement in this turn
2. a persisted, validated identity fact
3. the configured identity
4. the built-in default

Hedged guesses ("I think your name might be Bob") are recorded as low-authority
evidence and deliberately do **not** displace a name the user stated outright.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from companion.application.facts import (
    AUTHORITY_EXPLICIT_USER,
    AUTHORITY_HEDGED,
    FactWriter,
)
from companion.application.ports import GraphStore
from companion.core.clock import Clock, SystemClock
from companion.domain.agent import AgentIdentity

log = logging.getLogger(__name__)

SELF_ENTITY_KEY = "agent_self_entity"
USER_ENTITY_KEY = "primary_user_entity"

PREDICATE_NAME = "self:name"
PREDICATE_USER_NAME = "user:name"
PREDICATE_PERSONA = "self:persona"

DEFAULT_AGENT_NAME = "Companion"

# Words that follow "your name is" without being a name.
_NOT_A_NAME = frozenset("""
not very nice the a an still always what who it that this my your our his her their
different same important irrelevant unknown secret changed wrong right cool great
good bad weird funny silly beautiful long short here there now then just really
kind sort type actually probably maybe perhaps unclear obvious hard easy
""".split())

# Hedges that downgrade a claim to a guess.
_HEDGE = re.compile(
    r"\b(i think|i believe|i guess|i suppose|maybe|perhaps|probably|possibly|"
    r"might be|may be|could be|isn'?t it|wasn'?t it|i wonder|not sure|"
    r"if i remember|correct me)\b",
    re.IGNORECASE,
)

# Question forms must never be read as assignments.
_QUESTION = re.compile(
    r"^\s*(what|who|whats|what'?s|do you|are you|is your|was your|tell me)\b.*\?*\s*$",
    re.IGNORECASE,
)

_NAME = r"([A-Za-z][A-Za-z0-9'\-]{0,23}(?:\s+[A-Z][A-Za-z0-9'\-]{0,23}){0,2})"

# Assignments of the AGENT's name.
_AGENT_NAME_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"\byour\s+(?:new\s+)?name\s+is\s+(?:now\s+)?{_NAME}",
        rf"\byou\s+are\s+(?:now\s+)?called\s+{_NAME}",
        rf"\byou'?re\s+(?:now\s+)?called\s+{_NAME}",
        rf"\bi'?(?:ll|m going to|will)\s+call\s+you\s+{_NAME}",
        rf"\blet'?s\s+call\s+you\s+{_NAME}",
        rf"\bchange\s+your\s+name\s+to\s+{_NAME}",
        rf"\brename\s+yourself\s+(?:to\s+)?{_NAME}",
        rf"\bfrom\s+now\s+on\s+you\s+are\s+{_NAME}",
        rf"\byour\s+name\s+will\s+be\s+{_NAME}",
        rf"\bi\s+(?:hereby\s+)?name\s+you\s+{_NAME}",
        rf"\byou\s+are\s+{_NAME}\s+now\b",
        # Hedged forms are matched deliberately so they can be recorded as weak
        # evidence rather than vanishing: _HEDGE marks them non-authoritative.
        rf"\byour\s+name\s+(?:might|may|could|must)\s+be\s+{_NAME}",
        rf"\byour\s+name\s+was\s+{_NAME}",
    )
)

# "You are Jarvis." — a bare copula assignment. Accepted only when the
# complement is capitalised and is not an ordinary adjective, so "you are
# helpful" and "you are working on it" are not read as renames.
# The pronoun matches case-insensitively; the complement must be capitalised,
# so the capture group stays case-sensitive.
_BARE_COPULA = re.compile(
    r"(?i:\byou(?:'re|\s+are)\s+)([A-Z][A-Za-z0-9'\-]{1,23}(?:\s+[A-Z][A-Za-z0-9'\-]{1,23}){0,2})\b"
)

_NOT_A_NAME_COMPLEMENT = frozenset("""
Helpful Amazing Great Good Bad Right Wrong Correct Incorrect Funny Smart Stupid
Awesome Nice Kind Cool Welcome Sure Ok Okay Here There Ready Able Free Busy
Alive Real Fake Human Robot Ai An A The My So Very Too Not Still Always Never
Probably Just Being Doing Going Talking Working Slow Fast Useless Useful Wonderful
Terrible Brilliant Confused Confusing Annoying Repeating Hallucinating Broken
""".split())

# Assignments of the USER's name.
_USER_NAME_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"\bmy\s+name\s+is\s+(?:now\s+)?{_NAME}",
        rf"\bi'?m\s+called\s+{_NAME}",
        rf"\bi\s+am\s+called\s+{_NAME}",
        rf"\bcall\s+me\s+{_NAME}",
        rf"\byou\s+can\s+call\s+me\s+{_NAME}",
    )
)

# Questions the identity layer can answer from the graph without the LLM.
_ASK_AGENT_NAME = re.compile(
    r"\b(what'?s?\s+(is\s+)?your\s+name|who\s+are\s+you|what\s+(are\s+you|do\s+i)\s+call\s+you)\b",
    re.IGNORECASE,
)
_ASK_PREVIOUS_AGENT_NAME = re.compile(
    r"\b(previous|old|former|earlier|last)\s+name\b|"
    r"\bwhat\s+(was|were)\s+your\s+(previous|old|former|earlier|last)?\s*name\b|"
    r"\bwhat\s+did\s+i\s+(used\s+to\s+)?call\s+you\b",
    re.IGNORECASE,
)


@dataclass
class IdentityStatement:
    """A detected claim about someone's name."""

    target: str          # "agent" | "user"
    name: str
    hedged: bool = False
    utterance: str = ""

    @property
    def authority(self) -> float:
        return AUTHORITY_HEDGED if self.hedged else AUTHORITY_EXPLICIT_USER

    @property
    def confidence(self) -> float:
        return 0.35 if self.hedged else 1.0

    @property
    def provenance(self) -> str:
        return "hedged" if self.hedged else "explicit_user_statement"


def detect_identity_statement(text: str) -> IdentityStatement | None:
    """Find a name assignment in a user utterance, or return None.

    Deliberately conservative: questions are never assignments, and a hedged
    sentence yields a *hedged* statement rather than nothing, so the caller can
    record it as weak evidence without letting it rewrite a stated identity.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    hedged = bool(_HEDGE.search(raw))
    is_question = bool(_QUESTION.match(raw)) or raw.endswith("?")

    for pattern in _AGENT_NAME_PATTERNS:
        m = pattern.search(raw)
        if m:
            name = _clean_name(m.group(1))
            if not name:
                continue
            if is_question and not hedged:
                # "is your name Jarvis?" asks, it does not assign.
                return None
            return IdentityStatement(target="agent", name=name, hedged=hedged or is_question,
                                     utterance=raw)
    for pattern in _USER_NAME_PATTERNS:
        m = pattern.search(raw)
        if m:
            name = _clean_name(m.group(1))
            if not name:
                continue
            if is_question and not hedged:
                return None
            return IdentityStatement(target="user", name=name, hedged=hedged or is_question,
                                     utterance=raw)

    m = _BARE_COPULA.search(raw)
    if m and not is_question:
        candidate = m.group(1).strip()
        head = candidate.split()[0]
        if head not in _NOT_A_NAME_COMPLEMENT:
            name = _clean_name(candidate)
            if name:
                return IdentityStatement(target="agent", name=name, hedged=hedged,
                                         utterance=raw)
    return None


def asks_agent_name(text: str) -> bool:
    return bool(_ASK_AGENT_NAME.search(text or "")) and not _ASK_PREVIOUS_AGENT_NAME.search(text or "")


def asks_previous_agent_name(text: str) -> bool:
    t = text or ""
    return bool(_ASK_PREVIOUS_AGENT_NAME.search(t)) and "name" in t.lower()


def _display_name(raw: str) -> str:
    """Present a configured name the way a name is written."""
    name = (raw or "").strip()
    if not name:
        return DEFAULT_AGENT_NAME
    return name if name[:1].isupper() else name[:1].upper() + name[1:]


def _clean_name(raw: str) -> str:
    name = (raw or "").strip().strip(".,!?;:\"'")
    # Keep only the leading name-ish run; "Jarvis and nothing else" -> "Jarvis".
    parts = []
    for word in name.split():
        w = word.strip(".,!?;:\"'")
        if not w or w.lower() in _NOT_A_NAME:
            break
        parts.append(w)
        if len(parts) == 3:
            break
    if not parts:
        return ""
    cleaned = " ".join(parts)
    if len(cleaned) > 40 or not re.match(r"^[A-Za-z][A-Za-z0-9'\- ]*$", cleaned):
        return ""
    return cleaned[:1].upper() + cleaned[1:]


@dataclass
class SelfModel:
    """The agent's persistent model of itself.

    Distinguishes what was *learned* (name, persona, how it was named) from
    what was merely *configured*, and from what is *live* (capabilities come
    from the skill registry, never from stale storage).
    """

    identity: AgentIdentity = field(default_factory=AgentIdentity)
    name_source: str = "default"       # explicit_user_statement | persisted | config | default
    named_at: str = ""
    previous_names: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "identity": self.identity.to_dict(),
            "name_source": self.name_source,
            "named_at": self.named_at,
            "previous_names": list(self.previous_names),
            "capabilities": list(self.capabilities),
            "limitations": list(self.limitations),
        }


class SelfModelService:
    """Owns the agent's identity: reconstruction, renaming, and history."""

    def __init__(self, graph: GraphStore, clock: Clock | None = None,
                 configured_name: str = "", writer: FactWriter | None = None) -> None:
        self._graph = graph
        self._clock = clock or SystemClock()
        self._configured_name = (configured_name or "").strip()
        self._writer = writer or FactWriter(graph, self._clock)
        self._model = SelfModel()
        self._capability_provider = None

    # -- wiring -----------------------------------------------------------

    def set_capability_provider(self, provider) -> None:
        """Capabilities are read live from the skill registry, not stored."""
        self._capability_provider = provider

    def self_entity_id(self) -> str:
        return self._writer.entity_id(
            SELF_ENTITY_KEY, name="self", type="agent",
            summary="The companion itself.",
        )

    def user_entity_id(self) -> str:
        return self._writer.entity_id(
            USER_ENTITY_KEY, name="user", type="person",
            summary="The primary user of this companion.",
        )

    # -- reconstruction ---------------------------------------------------

    def load(self) -> SelfModel:
        """Rebuild the self-model from persistent storage.

        This is what makes a restart a *continuation*: the name the user gave
        the agent three sessions ago is recovered here, before any LLM call.
        """
        identity = AgentIdentity()
        name_source = "default"
        named_at = ""
        previous: list[str] = []
        try:
            subject = self.self_entity_id()
            current = self._writer.current(subject, PREDICATE_NAME)
            if current is not None and (current.value or "").strip():
                identity.name = current.value.strip()
                name_source = "persisted"
                named_at = current.valid_from or current.created_at
            elif self._configured_name:
                identity.name = _display_name(self._configured_name)
                name_source = "config"
            else:
                identity.name = DEFAULT_AGENT_NAME
            persona = self._writer.current(subject, PREDICATE_PERSONA)
            if persona is not None and persona.value:
                identity.persona = persona.value
            previous = [f.value for f in self._writer.history(subject, PREDICATE_NAME)
                        if f.valid_to and f.value]
        except Exception as exc:
            # A storage failure must not leave the agent nameless.
            log.warning("could not load persisted identity, using config: %s", exc)
            identity.name = self._configured_name or DEFAULT_AGENT_NAME
            name_source = "config" if self._configured_name else "default"
        self._model = SelfModel(
            identity=identity, name_source=name_source, named_at=named_at,
            previous_names=previous,
        )
        return self._model

    def model(self) -> SelfModel:
        model = self._model
        if self._capability_provider is not None:
            try:
                model.capabilities = list(self._capability_provider())
            except Exception as exc:
                log.warning("capability provider failed: %s", exc)
        return model

    @property
    def name(self) -> str:
        return self._model.identity.name or DEFAULT_AGENT_NAME

    # -- mutation ---------------------------------------------------------

    def apply_statement(self, statement: IdentityStatement, episode_id: str = "") -> dict:
        """Apply a detected identity statement, honouring authority rules."""
        if statement.target == "agent":
            return self.set_name(
                statement.name, authority=statement.authority,
                confidence=statement.confidence, provenance=statement.provenance,
                episode_id=episode_id, evidence_text=statement.utterance,
            )
        return self.set_user_name(
            statement.name, authority=statement.authority,
            confidence=statement.confidence, provenance=statement.provenance,
            episode_id=episode_id, evidence_text=statement.utterance,
        )

    def set_name(self, name: str, *, authority: float = AUTHORITY_EXPLICIT_USER,
                 confidence: float = 1.0, provenance: str = "explicit_user_statement",
                 episode_id: str = "", evidence_text: str = "") -> dict:
        name = _clean_name(name)
        if not name:
            return {"changed": False, "reason": "invalid name"}
        previous = self.name
        assertion = self._writer.assert_fact(
            subject_id=self.self_entity_id(),
            predicate=PREDICATE_NAME,
            value=name,
            confidence=confidence,
            importance=1.0,
            authority=authority,
            provenance=provenance,
            source_episode_id=episode_id,
            evidence_text=evidence_text,
            permanent=True,
        )
        if assertion.fact is None:
            log.info("identity change to %r refused (authority %.2f)", name, authority)
            return {"changed": False, "reason": "lower authority than existing identity",
                    "name": previous}
        if assertion.created:
            self._model.identity.name = name
            self._model.name_source = provenance
            self._model.named_at = assertion.fact.valid_from
            if previous and previous != name and previous not in self._model.previous_names:
                self._model.previous_names.insert(0, previous)
        return {"changed": bool(assertion.created), "name": name, "previous": previous,
                "superseded": assertion.superseded}

    def set_user_name(self, name: str, *, authority: float = AUTHORITY_EXPLICIT_USER,
                      confidence: float = 1.0, provenance: str = "explicit_user_statement",
                      episode_id: str = "", evidence_text: str = "") -> dict:
        name = _clean_name(name)
        if not name:
            return {"changed": False, "reason": "invalid name"}
        assertion = self._writer.assert_fact(
            subject_id=self.user_entity_id(),
            predicate=PREDICATE_USER_NAME,
            value=name,
            confidence=confidence,
            importance=0.95,
            authority=authority,
            provenance=provenance,
            source_episode_id=episode_id,
            evidence_text=evidence_text,
            permanent=True,
        )
        return {"changed": bool(assertion.created), "name": name,
                "superseded": assertion.superseded}

    # -- queries ----------------------------------------------------------

    def user_name(self) -> str:
        try:
            fact = self._writer.current(self.user_entity_id(), PREDICATE_USER_NAME)
        except Exception:
            return ""
        return (fact.value or "").strip() if fact else ""

    def previous_name(self) -> str:
        try:
            fact = self._writer.previous(self.self_entity_id(), PREDICATE_NAME)
        except Exception:
            return ""
        return (fact.value or "").strip() if fact else ""

    def name_history(self) -> list[dict]:
        try:
            facts = self._writer.history(self.self_entity_id(), PREDICATE_NAME)
        except Exception:
            return []
        return [
            {"name": f.value, "from": f.valid_from, "to": f.valid_to or "",
             "current": not f.valid_to, "provenance": f.provenance,
             "source_episode": f.source_episode_id}
            for f in facts if f.value
        ]

    def describe(self) -> str:
        """The AGENT IDENTITY context section — grounded, never invented."""
        model = self.model()
        lines = [f"My name is {self.name}."]
        if model.name_source in ("persisted", "explicit_user_statement") and model.named_at:
            # Phrasing matters: the agent remembers being named, it does not
            # claim to have been born with the name.
            lines.append(f"I remember that you named me {self.name}"
                         f" ({model.named_at[:10]}).")
        elif model.name_source == "config":
            lines.append("That name came from my configuration, not from you.")
        if model.previous_names:
            lines.append("You previously called me " + ", ".join(model.previous_names[:3]) + ".")
        user_name = self.user_name()
        if user_name:
            lines.append(f"I am talking with {user_name}.")
        lines.append("I am a local, private AI companion. I am not human and do not "
                     "pretend to be.")
        if model.capabilities:
            lines.append("Skills I can actually run: " + ", ".join(sorted(model.capabilities)) + ".")
        return "\n".join(lines)
