"""Structured extraction from episodes.

Preference: LLM produces strict JSON (validated into dataclasses). Fallback:
a deterministic rule-based extractor so the pipeline works and is testable
without model weights. Never parse arbitrary prose when structure is possible.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from companion.core.contracts import GenerationRequest
from companion.core.errors import ValidationError
from companion.domain.personality import PersonalityEvidence

log = logging.getLogger(__name__)

MEMORY_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {"type": "string"},
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "importance": {"type": "number"},
                    "confidence": {"type": "number"},
                    "temporal": {"type": "string"},
                },
                "required": ["content"],
            },
        },
        "personality_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "direction": {"type": "string"},
                    "strength": {"type": "number"},
                    "confidence": {"type": "number"},
                    "context": {"type": "string"},
                },
                "required": ["target"],
            },
        },
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "person": {"type": "string"},
                    "note": {"type": "string"},
                    "valence_delta": {"type": "number"},
                },
                "required": ["person"],
            },
        },
    },
    "required": ["memories"],
}

# Roughly 1.5k tokens of transcript: enough to cover a session's recent turns
# without letting prompt-processing time grow without bound.
MAX_EXTRACTION_CHARS = 6000

EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable memories from a conversation. "
    "Follow the instructions: separate observations from inferences; never infer a stable "
    "personality trait from a single event; never claim an emotion as fact; attach confidence "
    "for each memory; prefer context-specific statements over global claims. "
    "Output ONLY valid JSON matching the provided schema."
)


@dataclass
class ExtractedMemory:
    content: str
    type: str = "semantic"
    subject: str = ""
    predicate: str = ""
    object: str = ""
    importance: float = 0.3
    confidence: float = 0.5
    temporal: str = ""

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "type": self.type,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "importance": self.importance,
            "confidence": self.confidence,
            "temporal": self.temporal,
        }


@dataclass
class ExtractionResult:
    memories: list[ExtractedMemory] = field(default_factory=list)
    personality_evidence: list[PersonalityEvidence] = field(default_factory=list)
    goals: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    method: str = "none"

    def to_dict(self) -> dict:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "personality_evidence": [e.to_dict() for e in self.personality_evidence],
            "goals": self.goals,
            "relationships": self.relationships,
            "method": self.method,
        }


class StructuredExtractor:
    """LLM-based extraction with validated JSON + rule-based fallback."""

    def __init__(self, llm=None, router=None) -> None:
        self._llm = llm
        self._router = router
        self._rule = RuleBasedExtractor()

    async def extract(self, episode_transcript: str, episode_id: str = "") -> ExtractionResult:
        if self._llm is not None:
            try:
                result = await self._llm_extract(episode_transcript, episode_id)
                if result.memories or result.personality_evidence:
                    result.method = "llm"
                    return result
            except Exception as exc:
                log.warning("LLM extraction failed, falling back to rules: %s", exc)
        result = self._rule.extract(episode_transcript, episode_id)
        result.method = "rules"
        return result

    async def _llm_extract(self, transcript: str, episode_id: str) -> ExtractionResult:
        provider = self._llm if self._llm else None
        if provider is None:
            raise ValidationError("no LLM available for extraction")
        # Consolidation runs on a CPU-bound local model, so both ends of the
        # request are bounded: a long transcript costs prompt-processing time
        # and a large max_tokens costs generation time. Extraction that never
        # finishes is worth less than extraction that covers recent turns.
        clipped = transcript[-MAX_EXTRACTION_CHARS:]
        req = GenerationRequest(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            prompt=f"<transcript>\n{clipped}\n</transcript>\n\nReturn the extraction JSON.",
            json_schema=MEMORY_EXTRACTION_SCHEMA,
            temperature=0.1,
            max_tokens=512,
        )
        resp = await provider.generate(req)
        return self._parse(resp.text, episode_id)

    @staticmethod
    def _parse(text: str, episode_id: str) -> ExtractionResult:
        cleaned = _strip_json_fence(text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"LLM returned invalid JSON: {exc}") from exc
        return _result_from_dict(data, episode_id)


class RuleBasedExtractor:
    """Deterministic, conservative extraction (no weights needed)."""

    # subject/user -> predicate/object patterns on user turns.
    _PATTERNS: list[tuple[str, str]] = [
        (r"\bi (?:really )?love (?:the )?(.+)", ("likes", "love")),
        (r"\bi (?:really )?like (?:the )?(.+)", ("likes", "like")),
        (r"\bi (?:am|'m) (?:currently )?working on (.+)", ("works_on", None)),
        (r"\bi work (?:on|in|at) (.+)", ("works_in", None)),
        (r"\bi (?:am|'m) working (?:hard )?on (.+)", ("works_on", None)),
        (r"\bi (?:am|'m) building (.+)", ("building", None)),
        (r"\bmy project (?:involves|is about|focuses on|is) (.+)", ("project_involves", None)),
        (r"\bi (?:am|'m) studying (.+)", ("studies", None)),
        (r"\bi study (.+)", ("studies", None)),
        (r"\bi want to study (.+)", ("wants_to_study", "goal")),
        (r"\bi (?:want to|wants to) become (.+)", ("wants_to_become", "goal")),
        (r"\bi (?:want|would like) to learn (.+)", ("wants_to_learn", "goal")),
        (r"\bi (?:hate|dislike) (.+)", ("dislikes", "hate")),
        (r"\bi prefer (.+)", ("prefers", "prefer")),
        (r"\bmy name is ([A-Za-z]+)", ("has_name", None)),
        (r"\bmy favorite (.+?) is (.+)", ("has_favorite", None)),
        (r"\bi (?:am|'m) (?:from|based in) (.+)", ("located_in", None)),
        (r"\bi (?:have|got) a (.+?) (?:called|named) (.+)", ("owns", "has")),
    ]

    # family/acquaintance relationship patterns: "my X wants to study Y".
    _RELATIONSHIP_PATTERNS: list[tuple[str, str]] = [
        (r"\bmy (son|daughter|brother|sister|wife|husband|partner|friend|mom|mother|dad|father|grandma|grandpa|cousin|nephew|niece) (?:wants to|would like to|is going to) study (.+)",
         "wants_to_study"),
        (r"\bmy (son|daughter|brother|sister|wife|husband|partner|friend|mom|mother|dad|father|grandma|grandpa|cousin|nephew|niece) (?:works|is working) (?:at|in|for) (.+)",
         "works_in"),
    ]

    _PERSONALITY_PATTERNS: list[tuple[str, tuple[str, str, float]]] = [
        (r"\b(i love|i enjoy|i like) exploring|new things|novel|try out", ("novelty_seeking", "positive", 0.4)),
        (r"\b(why|how|what if|interesting|tell me more)", ("curiosity", "positive", 0.35)),
        (r"\b(i hate|i dislike|i can't stand) group work|groups|teams", ("prefers_solo", "positive", 0.4)),
        (r"\bi enjoy working in groups|team player|i like teams", ("prefers_groups", "positive", 0.4)),
        (r"\bi plan|i'm planning|i decided", ("planning_oriented", "positive", 0.3)),
        (r"\b(i am|i'm) (?:very )?(nervous|anxious|worried|stressed) about", ("stress_proneness", "positive", 0.35)),
    ]

    def extract(self, transcript: str, episode_id: str = "") -> ExtractionResult:
        result = ExtractionResult()
        for turn in _split_turns(transcript):
            if turn["role"] == "user":
                self._extract_user_turn(turn["text"], result, episode_id)
        return result

    def _extract_user_turn(self, text: str, result: ExtractionResult, episode_id: str) -> None:
        cleaned = re.sub(r"\s+", " ", text.strip().lower())
        for pattern, (predicate, kind) in self._PATTERNS:
            m = re.search(pattern, cleaned)
            if not m:
                continue
            obj = m.group(1).strip(" .")
            if not obj:
                continue
            mem_type = _infer_type(predicate, kind)
            importance = (
                0.4
                if predicate in ("works_in", "works_on", "building", "project_involves",
                                 "wants_to_study", "wants_to_learn", "wants_to_become", "studies")
                else 0.35
            )
            result.memories.append(
                ExtractedMemory(
                    content=f"user {predicate} {obj}",
                    type=mem_type,
                    subject="user",
                    predicate=predicate,
                    object=obj,
                    importance=importance,
                    confidence=0.6,
                )
            )
            if predicate in ("wants_to_study", "wants_to_learn", "wants_to_become") and kind == "goal":
                result.goals.append(
                    {"name": f"{predicate.replace('_', ' ')} {obj}", "description": f"user expressed intent to {predicate} {obj}"}
                )
            if predicate in ("likes", "dislikes", "prefers"):
                direction = "positive" if predicate in ("likes", "prefers") else "negative"
                if predicate == "likes":
                    direction = "positive" if kind != "hate" else "negative"
                result.personality_evidence.append(
                    PersonalityEvidence(
                        target=_pref_target(obj),
                        direction=direction,
                        strength=0.35,
                        confidence=0.4,
                        source_episode=episode_id,
                        source="conversation",
                        context=f"user said '{text[:100]}'",
                        kind="preference",
                    )
                )
        for pattern, predicate in self._RELATIONSHIP_PATTERNS:
            m = re.search(pattern, cleaned)
            if not m:
                continue
            person = m.group(1).strip().lower()
            obj = m.group(2).strip(" .")
            if not obj:
                continue
            result.relationships.append(
                {
                    "person": person,
                    "note": f"{person} {predicate} {obj}",
                    "valence_delta": 0.0,
                }
            )
            result.memories.append(
                ExtractedMemory(
                    content=f"{person} {predicate} {obj}",
                    type="semantic",
                    subject=person,
                    predicate=predicate,
                    object=obj,
                    importance=0.35,
                    confidence=0.6,
                )
            )
        for pattern, (trait, direction, strength) in self._PERSONALITY_PATTERNS:
            if re.search(pattern, cleaned):
                result.personality_evidence.append(
                    PersonalityEvidence(
                        target=trait,
                        direction=direction,
                        strength=strength,
                        confidence=0.35,
                        source_episode=episode_id,
                        source="conversation",
                        context=f"user said '{text[:100]}'",
                        kind="statement",
                    )
                )


def _pref_target(obj: str) -> str:
    return f"likes:{obj}"


def _infer_type(predicate: str, kind: str | None) -> str:
    if kind == "goal":
        return "goal"
    if predicate in ("likes", "dislikes", "prefers", "has_favorite"):
        return "preference"
    if predicate == "has_name":
        return "semantic"
    return "semantic"


def _split_turns(transcript: str) -> list[dict]:
    """Parse the transcript string into role/text turns.

    Accepts our canonical episode transcript format: 'user: ...\nassistant: ...'
    """
    turns: list[dict] = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(user|assistant|agent):\s*(.*)$", line, re.IGNORECASE)
        if m:
            role = "user" if m.group(1).lower() in ("user",) else "assistant"
            turns.append({"role": role, "text": m.group(2)})
        elif turns:
            turns[-1]["text"] += " " + line
        else:
            turns.append({"role": "user", "text": line})
    return turns


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _result_from_dict(data: dict, episode_id: str) -> ExtractionResult:
    result = ExtractionResult(raw=data)
    for m in data.get("memories", []):
        result.memories.append(
            ExtractedMemory(
                content=str(m.get("content", "")),
                type=str(m.get("type", "semantic")),
                subject=str(m.get("subject", "")),
                predicate=str(m.get("predicate", "")),
                object=str(m.get("object", "")),
                importance=float(m.get("importance", 0.3)),
                confidence=float(m.get("confidence", 0.5)),
                temporal=str(m.get("temporal", "")),
            )
        )
    for e in data.get("personality_evidence", []):
        result.personality_evidence.append(
            PersonalityEvidence(
                target=str(e.get("target", "")),
                direction=str(e.get("direction", "positive")),
                strength=float(e.get("strength", 0.3)),
                confidence=float(e.get("confidence", 0.3)),
                source_episode=episode_id,
                source="conversation",
                context=str(e.get("context", "")),
                kind="statement",
            )
        )
    result.goals = [dict(g) for g in data.get("goals", [])]
    result.relationships = [dict(r) for r in data.get("relationships", [])]
    return result
