"""Conversation domain model.

The LLM never directly controls voice/avatar. A ResponsePlan is produced first
and consumed independently by the LLM (text), TTS (prosody) and avatar
(expression/gaze/motion). This keeps all output modalities synchronized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    CHAT = "chat"
    ANSWER = "answer"
    QUESTION = "question"
    CLARIFY = "clarify"
    GREETING = "greeting"
    FAREWELL = "farewell"
    COMMAND = "command"
    SMALLTALK = "smalltalk"
    TASK = "task"
    EXPRESSION_OF_NEED = "expression_of_need"


class Tone(str, Enum):
    CALM = "calm"
    WARM = "warm"
    PLAYFUL = "playful"
    SUPPORTIVE = "supportive"
    SERIOUS = "serious"
    FORMAL = "formal"
    ENCOURAGING = "encouraging"
    NEUTRAL = "neutral"


@dataclass
class ConversationTurn:
    role: str          # user | assistant
    text: str
    timestamp: str = ""
    source: str = "text"  # text | speech | api
    user_state: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "text": self.text,
            "timestamp": self.timestamp,
            "source": self.source,
            "user_state": self.user_state,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationTurn":
        return cls(
            role=str(d.get("role", "")),
            text=str(d.get("text", "")),
            timestamp=str(d.get("timestamp", "")),
            source=str(d.get("source", "text")),
            user_state=dict(d.get("user_state", {})),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class ResponsePlan:
    """Intermediate plan that drives text, prosody and expression together."""

    intent: Intent = Intent.CHAT
    tone: Tone = Tone.NEUTRAL
    warmth: float = 0.5
    humor: float = 0.1
    verbosity: float = 0.4
    ask_followup: bool = False
    emotion: str = "neutral"
    confidence: float = 0.7
    affect: dict = field(default_factory=dict)          # AffectVector dict
    gaze_target: str = "user"                           # user | away | shared
    animation_intent: str = "talk"                      # talk | nod | listen | smile | think
    speech_speed: float = 1.0
    reasoning_used: bool = False
    retrieved: list[str] = field(default_factory=list)  # memory ids used

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "tone": self.tone.value,
            "warmth": self.warmth,
            "humor": self.humor,
            "verbosity": self.verbosity,
            "ask_followup": self.ask_followup,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "affect": self.affect,
            "gaze_target": self.gaze_target,
            "animation_intent": self.animation_intent,
            "speech_speed": self.speech_speed,
            "reasoning_used": self.reasoning_used,
            "retrieved": self.retrieved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResponsePlan":
        intent = Intent.CHAT
        try:
            intent = Intent(d.get("intent", Intent.CHAT.value))
        except ValueError:
            intent = Intent.CHAT
        tone = Tone.NEUTRAL
        try:
            tone = Tone(d.get("tone", Tone.NEUTRAL.value))
        except ValueError:
            tone = Tone.NEUTRAL
        return cls(
            intent=intent,
            tone=tone,
            warmth=float(d.get("warmth", 0.5)),
            humor=float(d.get("humor", 0.1)),
            verbosity=float(d.get("verbosity", 0.4)),
            ask_followup=bool(d.get("ask_followup", False)),
            emotion=str(d.get("emotion", "neutral")),
            confidence=float(d.get("confidence", 0.7)),
            affect=dict(d.get("affect", {})),
            gaze_target=str(d.get("gaze_target", "user")),
            animation_intent=str(d.get("animation_intent", "talk")),
            speech_speed=float(d.get("speech_speed", 1.0)),
            reasoning_used=bool(d.get("reasoning_used", False)),
            retrieved=list(d.get("retrieved", [])),
        )
