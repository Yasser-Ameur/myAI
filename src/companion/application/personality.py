"""Personality engine.

Applies PersonalityEvidence to the graph-backed profile with deliberately
conservative updates. A single statement never radically changes a stable
trait. Contradictions are recorded, never resolved by deleting either side.
Communication preferences are learned from interaction behavior.
"""

from __future__ import annotations

import logging

from companion.application.ports import GraphStore
from companion.core.clock import Clock, SystemClock
from companion.core.ids import new_contradiction_id
from companion.domain.personality import (
    COMMUNICATION_PREFERENCES,
    Contradiction,
    PersonalityEvidence,
    PersonalityProfile,
    StabilityClass,
    Trait,
)

log = logging.getLogger(__name__)

UPDATE_RATES = {"conservative": 0.03, "balanced": 0.08, "responsive": 0.15}

# Simple keyword signals for communication-preference learning when the LLM
# extractor is unavailable.
_COMMUNICATION_SIGNALS = {
    "preferred_response_length": {
        "short": ("concise", "short", "brief", "just the answer", "keep it short", "tl;dr", "too long", "stop explaining"),
        "long": ("tell me more", "in detail", "explain more", "elaborate", "go deeper", "full explanation"),
    },
    "preferred_directness": {
        "direct": ("get to the point", "just tell me", "no fluff", "directly", "straight", "yes or no"),
        "indirect": ("maybe", "perhaps", "could you", "might", "softly"),
    },
    "preferred_humor": {
        "low": ("stop joking", "be serious", "no jokes", "not funny"),
        "high": ("joke", "funny", "make me laugh", "humor", "haha"),
    },
    "preferred_question_frequency": {
        "low": ("stop asking questions", "no more questions", "don't ask"),
        "high": ("ask me", "quiz me", "ask questions", "what do you think"),
    },
    "preferred_technical_depth": {
        "low": ("simpler terms", "explain simply", "too technical", "plain language"),
        "high": ("more technical", "in depth", "under the hood", "implementation details"),
    },
}


class PersonalityEngine:
    def __init__(self, graph: GraphStore, clock: Clock | None = None,
                 update_mode: str = "conservative", decay_days: int = 180) -> None:
        self._graph = graph
        self._clock = clock or SystemClock()
        self.update_mode = update_mode
        self._rate = UPDATE_RATES.get(update_mode, 0.05)
        self._decay_days = decay_days
        self._profile = PersonalityProfile()

    def load(self) -> "PersonalityEngine":
        self._profile = self._graph.load_profile()
        return self

    def profile(self) -> PersonalityProfile:
        return self._profile

    # -- evidence application --------------------------------------------

    def apply_evidence(self, evidence: PersonalityEvidence) -> None:
        target = evidence.target.strip()
        if not target:
            return
        # A target may exist in multiple scopes (e.g. "curiosity" is both a
        # core value and an agent trait); update every scope it belongs to.
        updated = False
        if target in self._profile.values:
            self._apply_value(target, evidence)
            updated = True
        if target in self._profile.traits:
            self._apply_trait(target, evidence)
            updated = True
        if target in self._profile.preferences:
            self._apply_preference(target, evidence)
            updated = True
        if not updated:
            self._create_trait(target, evidence)
        self._graph.add_personality_evidence(evidence)
        self._check_contradiction(evidence)
        self._graph.save_profile(self._profile)

    def _apply_trait(self, name: str, evidence: PersonalityEvidence) -> None:
        t = self._profile.traits[name]
        t.value = _shift_value(t.value, evidence, self._rate * t.stability)
        t.confidence = _shift_confidence(t.confidence, t.value, evidence)
        t.evidence_count += 1
        # consistency -> slowly more stable; contradiction -> slightly less
        if _agrees(t.value, evidence):
            t.stability = min(0.95, t.stability + (1 - t.stability) * 0.015)
        else:
            t.stability = max(0.4, t.stability - 0.02)
        t.updated_at = self._clock.now_iso()

    def _apply_value(self, name: str, evidence: PersonalityEvidence) -> None:
        v = self._profile.values[name]
        # values barely move
        v.importance = _shift_value(v.importance, evidence, self._rate * 0.5 * v.stability)
        v.confidence = _shift_confidence(v.confidence, v.importance, evidence)
        v.updated_at = self._clock.now_iso()

    def _apply_preference(self, name: str, evidence: PersonalityEvidence) -> None:
        p = self._profile.preferences[name]
        # preferences are mid-stability: faster than values, slower than state
        p.value = _shift_value(p.value, evidence, self._rate * 2.0 * (1 - p.stability))
        p.confidence = _shift_confidence(p.confidence, p.value, evidence)
        p.evidence_count += 1
        if _agrees(p.value, evidence):
            p.stability = min(0.85, p.stability + (1 - p.stability) * 0.03)
        else:
            p.stability = max(0.2, p.stability - 0.03)
        p.updated_at = self._clock.now_iso()

    def _create_trait(self, name: str, evidence: PersonalityEvidence) -> None:
        stability_class = (
            StabilityClass.MEDIUM if name in COMMUNICATION_PREFERENCES or name in self._profile.preferences
            else StabilityClass.VERY_STABLE
        )
        t = Trait(
            name=name,
            value=0.5 + evidence.signed_strength() * 0.1,
            confidence=evidence.confidence * 0.4,
            stability=0.5 if stability_class == StabilityClass.MEDIUM else 0.6,
            evidence_count=1,
            stability_class=stability_class,
            updated_at=self._clock.now_iso(),
        )
        self._profile.traits[name] = t

    # -- contradiction handling ------------------------------------------

    def _check_contradiction(self, evidence: PersonalityEvidence) -> None:
        """If new evidence conflicts strongly with an existing belief, record it."""
        if evidence.kind != "statement":
            return
        existing = self._graph.list_beliefs(target_type="personality")
        for belief in existing:
            if belief.target_name != evidence.target:
                continue
            belief_value = float(belief.value.get("value", 0.5))
            direction = belief_value >= 0.6
            new_direction = evidence.signed_strength() >= 0.0
            if direction != new_direction and belief.confidence > 0.5:
                con = Contradiction(
                    id=new_contradiction_id(),
                    statement_a=f"{evidence.target}: {'high' if direction else 'low'} (confidence {belief.confidence:.2f})",
                    statement_b=f"user says evidence sign {evidence.signed_strength():+.2f} ({evidence.context or evidence.source})",
                    subject=evidence.source_episode,
                    predicate=evidence.target,
                    contexts=[belief.updated_at, evidence.timestamp],
                    timestamps=[belief.updated_at, evidence.timestamp],
                )
                self._graph.add_contradiction(con)
                belief.status = "conflicting"
                self._graph.upsert_belief(belief)
                log.info("recorded contradiction on %s", evidence.target)
                return

    # -- communication preference learning --------------------------------

    def learn_communication_preference(self, text: str, episode_id: str = "") -> list[PersonalityEvidence]:
        lowered = text.lower()
        produced: list[PersonalityEvidence] = []
        target_map = {
            ("preferred_response_length", "short"): ("negative", 0.5),
            ("preferred_response_length", "long"): ("positive", 0.5),
            ("preferred_directness", "direct"): ("positive", 0.5),
            ("preferred_directness", "indirect"): ("negative", 0.5),
            ("preferred_technical_depth", "low"): ("negative", 0.4),
            ("preferred_technical_depth", "high"): ("positive", 0.4),
            ("preferred_humor", "low"): ("negative", 0.4),
            ("preferred_humor", "high"): ("positive", 0.4),
            ("preferred_question_frequency", "low"): ("negative", 0.4),
            ("preferred_question_frequency", "high"): ("positive", 0.4),
        }
        for pref, signals in _COMMUNICATION_SIGNALS.items():
            for direction, phrases in signals.items():
                if not any(ph in lowered for ph in phrases):
                    continue
                dir2, strength = target_map.get((pref, direction), ("positive", 0.4))
                ev = PersonalityEvidence(
                    target=pref,
                    direction=dir2,
                    strength=strength,
                    confidence=0.35,
                    source_episode=episode_id,
                    source="conversation",
                    timestamp=self._clock.now_iso(),
                    context=f"behavioral signal: user said '{text[:80]}'",
                    kind="preference",
                )
                self.apply_evidence(ev)
                produced.append(ev)
        return produced

    # -- snapshot for context ---------------------------------------------

    def snapshot(self, max_traits: int = 6, max_prefs: int = 6) -> dict:
        """A compact, token-cheap summary for the context builder."""
        traits = sorted(
            self._profile.traits.values(), key=lambda t: -t.confidence * t.evidence_count
        )[:max_traits]
        prefs = sorted(
            self._profile.preferences.values(), key=lambda p: -p.confidence
        )[:max_prefs]
        return {
            "traits": [t.to_dict() for t in traits],
            "values": [v.to_dict() for v in list(self._profile.values.values())[:4]],
            "preferences": [p.to_dict() for p in prefs],
        }

    def evidence_for(self, target: str) -> list[PersonalityEvidence]:
        return self._graph.list_personality_evidence(target)

    def reset(self) -> None:
        for t in self._profile.traits.values():
            t.value = 0.5
            t.confidence = 0.0
            t.evidence_count = 0
            t.updated_at = self._clock.now_iso()
        for p in self._profile.preferences.values():
            p.value = 0.5
            p.confidence = 0.0
            p.evidence_count = 0
            p.updated_at = self._clock.now_iso()
        for v in self._profile.values.values():
            v.importance = 0.5
            v.confidence = 0.0
            v.updated_at = self._clock.now_iso()
        self._graph.save_profile(self._profile)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _shift_value(current: float, evidence: PersonalityEvidence, weight: float) -> float:
    delta = evidence.signed_strength() * evidence.confidence * weight * 0.5
    return max(0.0, min(1.0, current + delta))


def _agrees(value: float, evidence: PersonalityEvidence) -> bool:
    return (value >= 0.5) == (evidence.signed_strength() >= 0.0)


def _shift_confidence(current: float, value: float, evidence: PersonalityEvidence) -> float:
    if _agrees(value, evidence):
        return min(0.95, current + (1 - current) * 0.12 * evidence.confidence)
    return max(0.0, current - current * 0.15 * evidence.confidence)
