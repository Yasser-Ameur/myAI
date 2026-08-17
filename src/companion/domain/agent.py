"""Agent domain model: identity, values, internal state and the agent's own memory.

The agent has its own (separate) memory so it can reason about what it said,
believed and did — without this ever overwriting the user model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from companion.domain.personality import AGENT_PERSONALITY_DIMENSIONS, PersonalityProfile, Trait


@dataclass
class AgentIdentity:
    name: str = "Companion"
    description: str = "A local, private, curious companion."
    languages: list[str] = field(default_factory=lambda: ["en", "fr"])
    persona: str = ""
    avatar_id: str = "default"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "languages": self.languages,
            "persona": self.persona,
            "avatar_id": self.avatar_id,
        }


@dataclass
class AgentValues:
    """Non-negotiable agent values."""

    honesty: float = 0.95
    user_privacy: float = 1.0
    user_autonomy: float = 0.9
    epistemic_humility: float = 0.85  # does not overstate certainty
    helpfulness: float = 0.9
    safety: float = 0.95

    def to_dict(self) -> dict:
        return {k: float(getattr(self, k)) for k in self.__dataclass_fields__}


@dataclass
class AgentState:
    """Fast-moving internal state of the agent itself."""

    current_emotion: str = "neutral"
    energy: float = 0.6
    mood: float = 0.5            # valence of internal mood
    focus: str = ""
    speaking: bool = False
    listening: bool = False
    last_response_at: str = ""
    mode: str = "active"         # active | consolidating | degraded

    def to_dict(self) -> dict:
        return {
            "current_emotion": self.current_emotion,
            "energy": self.energy,
            "mood": self.mood,
            "focus": self.focus,
            "speaking": self.speaking,
            "listening": self.listening,
            "last_response_at": self.last_response_at,
            "mode": self.mode,
        }


@dataclass
class AgentBelief:
    """What the agent believes and why (own epistemic memory)."""

    id: str = ""
    claim: str = ""
    reason: str = ""
    formed_at: str = ""
    confidence: float = 0.5
    status: str = "active"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "claim": self.claim,
            "reason": self.reason,
            "formed_at": self.formed_at,
            "confidence": self.confidence,
            "status": self.status,
        }


@dataclass
class AgentObservation:
    """What the agent observed."""

    id: str = ""
    content: str = ""
    observed_at: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "content": self.content, "observed_at": self.observed_at, "source": self.source}


@dataclass
class AgentAction:
    """What the agent did."""

    id: str = ""
    action: str = ""
    rationale: str = ""
    performed_at: str = ""
    outcome: str = ""  # linked to an Outcome

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action": self.action,
            "rationale": self.rationale,
            "performed_at": self.performed_at,
            "outcome": self.outcome,
        }


@dataclass
class Outcome:
    """What happened afterward."""

    id: str = ""
    action_id: str = ""
    result: str = ""
    user_feedback: str = ""  # positive | neutral | negative | none
    observed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_id": self.action_id,
            "result": self.result,
            "user_feedback": self.user_feedback,
            "observed_at": self.observed_at,
        }


@dataclass
class AgentPersonality:
    """Configured agent personality (the agent is not a pure mirror of the user)."""

    profile: PersonalityProfile = field(
        default_factory=lambda: PersonalityProfile(
            traits={
                d: Trait(name=d, value=0.6, confidence=1.0, evidence_count=1)
                for d in AGENT_PERSONALITY_DIMENSIONS
            }
        )
    )

    def to_dict(self) -> dict:
        return self.profile.to_dict()
