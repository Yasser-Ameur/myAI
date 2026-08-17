"""Conversation application layer.

ContextAssembler builds token-bounded context sections. ResponsePlanner produces
a ResponsePlan that drives text + TTS prosody + avatar simultaneously. The LLM
never controls the face directly. Interruption and backchannel policies live
here as small, testable components.
"""

from __future__ import annotations

import logging
import re

from companion.application.ports import GraphStore
from companion.application.retrieval import HybridRetriever, RetrievedMemory
from companion.core.clock import Clock, SystemClock
from companion.core.contracts import GenerationRequest, LanguageModel
from companion.core.events import (
    EVENT_MEMORY_COMMITTED,
    EVENT_RESPONSE_COMPLETE,
    EVENT_RESPONSE_PLAN_CREATED,
    EVENT_RESPONSE_TOKEN_GENERATED,
    EVENT_RETRIEVAL_COMPLETE,
    EventBus,
)
from companion.domain.agent import AgentState
from companion.domain.conversation import Intent, ResponsePlan, Tone
from companion.domain.personality import PersonalityProfile
from companion.domain.state import UserState
from companion.infrastructure.models.router import TaskRouter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------

def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _agent_identity(agent_profile: PersonalityProfile | None,
                    agent_state: AgentState | None,
                    self_model=None) -> str:
    # The agent's own name is the first thing in its context. Before this the
    # section never mentioned it, so the companion could not answer "what is
    # your name?" even when the name was configured.
    if self_model is not None:
        lines = [self_model.describe()]
    else:
        lines = ["I am a local, private AI companion."]
    if agent_profile:
        for t in sorted(agent_profile.traits.values(), key=lambda x: -x.confidence * x.evidence_count)[:3]:
            if t.confidence > 0.2:
                lines.append(f"trait {t.name}: {t.value:.2f}")
        for v in list(agent_profile.values.values())[:3]:
            if v.confidence > 0.2:
                lines.append(f"value {v.name}: {v.importance:.2f}")
    if agent_state:
        lines.append(
            f"current emotion: {agent_state.current_emotion}, "
            f"energy: {agent_state.energy:.2f}, mood: {agent_state.mood:.2f}"
        )
    return "\n".join(lines)


def _user_profile_section(profile: PersonalityProfile | None,
                          user_state: UserState | None = None) -> str:
    lines = []
    if user_state:
        state_lines = []
        for dim, est in user_state.dimensions.items():
            if est.confidence >= 0.3:
                state_lines.append(f"{dim}: {est.value:.2f} (conf {est.confidence:.2f})")
        if state_lines:
            lines.append("current state: " + ", ".join(state_lines))
    if profile:
        for t in sorted(profile.traits.values(), key=lambda x: -x.confidence * x.evidence_count)[:5]:
            if t.confidence > 0.2:
                lines.append(f"trait {t.name}: {t.value:.2f} (conf {t.confidence:.2f}, {t.evidence_count} ev)")
        for p in list(profile.preferences.values())[:5]:
            if p.confidence > 0.2:
                lines.append(f"preference {p.name}: {p.value:.2f} (conf {p.confidence:.2f})")
        for v in list(profile.values.values())[:3]:
            if v.confidence > 0.2:
                lines.append(f"value {v.name}: {v.importance:.2f}")
    return "\n".join(lines)


def _recent_conversation_section(recent_turns: list[dict] | None) -> str:
    if not recent_turns:
        return ""
    lines = []
    for turn in recent_turns[-10:]:
        role = "user" if turn.get("role") == "user" else "companion"
        text = (turn.get("text") or "").strip().replace("\n", " ")
        lines.append(f"{role}: {text[:160]}")
    return "\n".join(lines)


def _response_policy_section(plan: ResponsePlan | None) -> str:
    lines = [
        "Answer directly and concisely; never emit <think> blocks or chain-of-thought.",
        "Do not invent facts about the user beyond what the context provides.",
        "Speak in the user's language.",
        "If unsure, say so.",
    ]
    if plan is not None:
        lines.append(
            f"Response intent: {plan.intent.value}; tone: {plan.tone.value}; "
            f"warmth {plan.warmth:.2f}; humor {plan.humor:.2f}; "
            f"verbosity {plan.verbosity:.2f}; "
            f"ask follow-up: {'yes' if plan.ask_followup else 'no'}."
        )
    return "\n".join(lines)


class ContextAssembler:
    """Builds bounded context sections with per-section token budgets.

    Section headers use the milestone bracket format [SECTION NAME]; retrieved
    memories are injected inside a single <retrieved_memory> block.
    """

    SECTION_ORDER = (
        "agent_identity",
        "user_profile",
        "memory",
        "episodes",
        "goals",
        "relationship",
        "recent_conversation",
        "response_policy",
        "current_user_message",
    )

    SECTION_LABELS = {
        "agent_identity": "AGENT IDENTITY",
        "user_profile": "USER PROFILE",
        "memory": "MEMORY",
        "episodes": "EPISODES",
        "goals": "GOALS",
        "relationship": "RELATIONSHIP",
        "recent_conversation": "RECENT CONVERSATION",
        "response_policy": "RESPONSE POLICY",
        "current_user_message": "CURRENT USER MESSAGE",
    }

    def __init__(self, budgets: dict | None = None, total_budget: int = 3500,
                 self_model=None) -> None:
        self.self_model = self_model
        self.budgets = budgets or {
            "agent_identity": 220,
            "user_profile": 500,
            "memory": 1200,
            "episodes": 700,
            "goals": 300,
            "relationship": 300,
            "recent_conversation": 600,
            "response_policy": 250,
            "current_user_message": 300,
        }
        self.total_budget = total_budget

    def build(
        self,
        *,
        query: str,
        user_state: UserState | None,
        profile: PersonalityProfile | None,
        retrieved: list[RetrievedMemory],
        goals: list,
        relationships: list,
        agent_state: AgentState | None,
        agent_profile: PersonalityProfile | None = None,
        plan: ResponsePlan | None = None,
        recent_turns: list[dict] | None = None,
        task_context: str = "",
    ) -> str:
        sections: list[str] = []
        budget_left = self.total_budget
        for name in self.SECTION_ORDER:
            budget = min(self.budgets.get(name, 100), budget_left)
            content = self._section(
                name, query, user_state, profile, retrieved, goals, relationships,
                agent_state, agent_profile, plan, recent_turns, task_context,
            )
            if not content:
                continue
            content = self._truncate(content, budget)
            sections.append(f"[{self.SECTION_LABELS.get(name, name.upper())}]\n{content}")
            budget_left -= _est_tokens(content)
            if budget_left <= 0:
                break
        return "\n\n".join(sections)

    def _section(self, name, query, user_state, profile, retrieved, goals, relationships,
                 agent_state, agent_profile, plan, recent_turns, task_context) -> str:
        if name == "agent_identity":
            return _agent_identity(agent_profile, agent_state, self_model=self.self_model)
        if name == "user_profile":
            return _user_profile_section(profile, user_state)
        if name == "memory":
            mems = [r for r in retrieved if r.source_type == "memory"]
            items = "\n".join(f"- {r.content}" for r in mems[:8])
            return f"<retrieved_memory>\n{items}\n</retrieved_memory>" if items else ""
        if name == "episodes":
            eps = [r for r in retrieved if r.source_type == "episode"]
            return "\n".join(f"- {r.content}" for r in eps[:4])
        if name == "goals":
            return "\n".join(f"- {g.get('name', '')}: {g.get('description', '')}" for g in goals[:4])
        if name == "relationship":
            return "\n".join(f"- {r.get('name', '')}: trust {r.get('trust', 0):.2f}" for r in relationships[:3])
        if name == "recent_conversation":
            return _recent_conversation_section(recent_turns)
        if name == "response_policy":
            return _response_policy_section(plan)
        if name == "current_user_message":
            # The current turn is not part of recent_turns until after prompt
            # assembly. Keep it explicit and last so the model always has the
            # instruction it must answer, even with a large memory context.
            return f"<current_user_message>\n{query}\n</current_user_message>"
        if name == "task":
            return task_context[: self.budgets.get("task", 100) * 4]
        return ""

    @staticmethod
    def _truncate(text: str, budget_tokens: int) -> str:
        if _est_tokens(text) <= budget_tokens:
            return text
        # cheap truncation at sentence-ish boundaries
        max_chars = budget_tokens * 4
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        last = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("; "))
        return cut[: last + 1] if last > max_chars // 2 else cut + "…"


# ---------------------------------------------------------------------------
# Response planner
# ---------------------------------------------------------------------------

RESPONSE_PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "intent": {"type": "string",
                   "enum": [i.value for i in Intent]},
        "tone": {"type": "string",
                 "enum": [t.value for t in Tone]},
        "warmth": {"type": "number"},
        "humor": {"type": "number"},
        "verbosity": {"type": "number"},
        "ask_followup": {"type": "boolean"},
        "emotion": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "tone"],
}

PLAN_SYSTEM_PROMPT = (
    "You are planning the agent's response BEFORE writing it. Decide intent, tone, "
    "warmth (0-1), humor (0-1), verbosity (0-1), whether to ask a follow-up, an "
    "emotion label, and overall confidence. Output ONLY the JSON object."
)


class RuleBasedPlanner:
    """Deterministic planning fallback (no LLM needed)."""

    def plan(self, query: str, user_state: UserState | None,
             profile: PersonalityProfile | None) -> ResponsePlan:
        lowered = query.lower().strip()
        if re.search(r"\?$|\? ", lowered) or lowered.startswith(("what", "how", "why", "when", "where", "who", "can", "could", "should")):
            intent = Intent.QUESTION
        elif re.search(r"\b(hi|hello|hey|good (morning|evening|afternoon))\b", lowered):
            intent = Intent.GREETING
        elif re.search(r"\b(bye|goodbye|see you|good night)\b", lowered):
            intent = Intent.FAREWELL
        elif lowered.startswith(("remember ", "forget ", "show memories", "why do you think")):
            intent = Intent.COMMAND
        else:
            intent = Intent.CHAT

        tone = Tone.NEUTRAL
        confusion = user_state.get("confusion").value if user_state and user_state.get("confusion") else 0.0
        frustration = user_state.get("frustration").value if user_state and user_state.get("frustration") else 0.0
        valence = user_state.get("valence").value if user_state and user_state.get("valence") else 0.0
        if confusion > 0.4 or frustration > 0.4:
            tone = Tone.SUPPORTIVE
        elif valence > 0.3:
            tone = Tone.WARM

        verbosity = 0.4
        humor = 0.1
        if profile and "preferred_response_length" in profile.preferences:
            pref = profile.preferences["preferred_response_length"]
            if pref.confidence > 0.2:
                verbosity = pref.value
        if profile and "preferred_humor" in profile.preferences:
            humor = profile.preferences["preferred_humor"].value

        emotion = "neutral"
        if confusion > 0.4:
            emotion = "supportive"
        elif frustration > 0.4:
            emotion = "calming"
        elif valence > 0.3:
            emotion = "warm"
        return ResponsePlan(
            intent=intent,
            tone=tone,
            warmth=0.55 + 0.2 * max(0.0, valence),
            humor=humor,
            verbosity=max(0.1, min(1.0, verbosity)),
            ask_followup=bool(confusion > 0.5),
            emotion=emotion,
            confidence=0.7,
            speech_speed=1.0,
        )


class LLMResponsePlanner:
    """Plans with a model when asked to; otherwise defers to the rules.

    Measured on the target machine (i7-1185G7, CPU-only), an LLM planning call
    costs ~2.1 s per turn while retrieval costs ~7 ms — it was the single
    largest avoidable component of turn latency. It also produced
    ``verbosity: 0.0`` often enough to truncate replies to 64 tokens. The
    deterministic planner is instant and, for intent/tone/verbosity, at least
    as good, so `mode="rules"` is the default and LLM planning is opt-in via
    `conversation.planner: llm`.
    """

    def __init__(self, llm: LanguageModel | None, router: TaskRouter | None = None,
                 fallback: RuleBasedPlanner | None = None, mode: str = "rules") -> None:
        self._llm = llm
        self._router = router
        self._fallback = fallback or RuleBasedPlanner()
        self._mode = (mode or "rules").lower()
        self._plan_llm = None
        if self._mode == "llm" and router is not None:
            try:
                slot = router.llm_for_task("planning")
                provider = router._registry.get_optional(slot)
                if provider is not None:
                    self._plan_llm = provider
            except Exception as exc:
                log.warning("could not resolve fast planning model: %s", exc)

    async def plan(self, query: str, user_state: UserState | None,
                   profile: PersonalityProfile | None) -> ResponsePlan:
        if self._mode != "llm":
            return self._fallback.plan(query, user_state, profile)
        if self._plan_llm is not None:
            try:
                plan = await self._llm_plan(self._plan_llm, query, user_state, profile)
                if plan is not None:
                    return plan
            except Exception as exc:
                log.warning("fast LLM planning failed, using rules: %s", exc)
        if self._llm is not None:
            try:
                plan = await self._llm_plan(self._llm, query, user_state, profile)
                if plan is not None:
                    return plan
            except Exception as exc:
                log.warning("LLM planning failed, using rules: %s", exc)
        return self._fallback.plan(query, user_state, profile)

    async def _llm_plan(self, llm, query, user_state, profile) -> ResponsePlan | None:
        import json

        context = []
        if user_state:
            context.append("user_state: " + json.dumps(user_state.to_dict()))
        if profile:
            context.append("user_profile: " + json.dumps(profile.to_dict())[:800])
        req = GenerationRequest(
            system_prompt=PLAN_SYSTEM_PROMPT,
            prompt=query,
            context=context,
            json_schema=RESPONSE_PLAN_SCHEMA,
            temperature=0.1,
            max_tokens=140,
        )
        resp = await llm.generate(req)
        plan = ResponsePlan.from_dict(json.loads(resp.text))
        # A small model routinely emits verbosity 0, which would cap the reply
        # at 64 tokens and cut it mid-sentence. Keep a floor.
        plan.verbosity = max(0.25, min(1.0, plan.verbosity))
        return plan


# ---------------------------------------------------------------------------
# Conversation service
# ---------------------------------------------------------------------------

class ConversationService:
    def __init__(
        self,
        llm: LanguageModel | None,
        retriever: HybridRetriever,
        assembler: ContextAssembler | None = None,
        planner=None,
        graph: GraphStore | None = None,
        memory=None,
        personality=None,
        relationships=None,
        agent_profile: PersonalityProfile | None = None,
        agent_state: AgentState | None = None,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        router: TaskRouter | None = None,
        system_prompt: str | None = None,
        self_model=None,
        turn_committer=None,
        skills=None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._self_model = self_model
        self._turn_committer = turn_committer
        self._skills = skills
        self._assembler = assembler or ContextAssembler(self_model=self_model)
        self._planner = planner or LLMResponsePlanner(llm, router)
        self._graph = graph
        self._memory = memory
        self._personality = personality
        self._relationships = relationships
        self._agent_profile = agent_profile or PersonalityProfile()
        self._agent_state = agent_state or AgentState()
        self.bus = bus or EventBus(clock=clock)
        self._clock = clock or SystemClock()
        self._router = router
        self._system_prompt = system_prompt
        self._interruption = InterruptionController()

    async def respond(self, text: str, source: str = "text",
                      user_state: UserState | None = None,
                      session_id: str = "", turn_id: str = "") -> "ConversationResult":
        user_state = user_state or self._last_user_state()
        trace = {"session_id": session_id, "turn_id": turn_id}

        # Record the turn and commit any explicit, unambiguous statements to
        # the graph BEFORE retrieval, so the answer is grounded in what was
        # just said and survives an ungraceful exit. Consolidation still runs
        # later for everything subtler than this.
        episode_id = ""
        if self._memory:
            self._memory.append_turn("user", text, source, user_state.to_dict())
            episode = self._memory.current_episode()
            episode_id = episode.id if episode else ""
            if self._personality:
                self._personality.learn_communication_preference(text, episode_id)
        commit = None
        if self._turn_committer is not None:
            try:
                commit = self._turn_committer.commit(text, episode_id=episode_id)
                if commit.changed:
                    self.bus.publish(EVENT_MEMORY_COMMITTED,
                                     {**commit.to_dict(), **trace})
            except Exception:
                log.exception("turn commit failed; continuing without durable facts")

        skill_result = await self._try_skill(text, user_state, trace, episode_id)
        if skill_result is not None:
            return skill_result

        retrieved = await self._retriever.retrieve(text, mode="auto", top_k=8)
        self.bus.publish(EVENT_RETRIEVAL_COMPLETE, {"query": text[:80], **trace})
        goals = [g.to_dict() for g in self._graph.list_goals()] if self._graph else []
        relationships = self._relationships.snapshot() if self._relationships else []
        profile = self._personality.profile() if self._personality else None

        plan = await self._planner.plan(text, user_state, profile)
        plan.retrieved = [r.id for r in retrieved]

        context = self._assembler.build(
            query=text,
            user_state=user_state,
            profile=profile,
            retrieved=retrieved,
            goals=goals,
            relationships=relationships,
            agent_state=self._agent_state,
            agent_profile=self._agent_profile,
            plan=plan,
            recent_turns=self._recent_turns(),
        )

        prompt = self._build_prompt(plan, context)
        self.bus.publish(EVENT_RESPONSE_PLAN_CREATED, {**plan.to_dict(), **trace})
        response = await self._generate(prompt, plan)

        if self._memory:
            self._memory.append_turn("assistant", response, source)
        self.bus.publish(EVENT_RESPONSE_COMPLETE,
                         {"text": response, "plan": plan.to_dict(), **trace})
        self._agent_state.last_response_at = self._clock.now_iso()
        return ConversationResult(text=response, plan=plan, retrieved=retrieved)

    async def _try_skill(self, text: str, user_state: UserState | None,
                         trace: dict, episode_id: str) -> "ConversationResult | None":
        """Give the skill router a chance to handle this turn deterministically.

        A skill answers only when it is confident; otherwise the normal
        LLM path runs. Skill failures never break the turn — they fall
        through to conversation.
        """
        if self._skills is None:
            return None
        try:
            outcome = await self._skills.handle(
                text, user_state=user_state, episode_id=episode_id, **trace
            )
        except Exception:
            log.exception("skill routing failed; falling back to conversation")
            return None
        if outcome is None or not outcome.handled:
            return None
        plan = RuleBasedPlanner().plan(text, user_state, None)
        plan.intent = Intent.ANSWER
        if self._memory:
            self._memory.append_turn("assistant", outcome.text, "skill")
        self.bus.publish(EVENT_RESPONSE_COMPLETE,
                         {"text": outcome.text, "plan": plan.to_dict(),
                          "skill": outcome.skill_id, **trace})
        self._agent_state.last_response_at = self._clock.now_iso()
        return ConversationResult(text=outcome.text, plan=plan, retrieved=[],
                                  skill_id=outcome.skill_id)

    async def _generate(self, prompt: str, plan: ResponsePlan) -> str:
        if self._llm is None:
            return "[no language model available]"
        req = GenerationRequest(
            system_prompt=self._system_prompt or self._default_system_prompt(plan),
            prompt=prompt,
            temperature=0.6 + 0.2 * plan.warmth,
            max_tokens=64 + int(plan.verbosity * 300),
        )
        tokens: list[str] = []
        try:
            async for tok in self._llm.stream(req):
                tokens.append(tok)
                self.bus.publish(EVENT_RESPONSE_TOKEN_GENERATED, {"token": tok})
        except Exception:
            result = await self._llm.generate(req)
            return result.text
        return self._finalize("".join(tokens))

    @staticmethod
    def _finalize(text: str) -> str:
        """Clean the streamed reply: drop any stray chain-of-thought blocks.

        Applied once on the complete text (per-token cleanup would strip the
        inter-word spaces that tokens carry as leading whitespace).
        """
        if not text:
            return text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()

    def _build_prompt(self, plan: ResponsePlan, context: str) -> str:
        return (
            "Given the context below, respond to the user's last message.\n\n"
            f"{context}\n\n"
            "Now write the response text only."
        )

    def _default_system_prompt(self, plan: ResponsePlan) -> str:
        return (
            "You are a local, private AI companion. Speak in the user's language. "
            "Do not invent facts about the user beyond what the context provides. "
            "If unsure, say so. "
            "Answer directly and concisely; never emit <think> blocks or chain-of-thought."
        )

    def _recent_turns(self) -> list[dict]:
        if self._memory is None:
            return []
        ep = self._memory.current_episode()
        if ep is None or not getattr(ep, "transcript", None):
            return []
        return list(ep.transcript)

    def _last_user_state(self) -> UserState:
        if self._memory is None:
            return UserState(timestamp=self._clock.now_iso())
        return UserState(timestamp=self._clock.now_iso())

    def interrupt(self) -> None:
        self._interruption.signal()


class ConversationResult:
    def __init__(self, text: str, plan: ResponsePlan, retrieved: list,
                 skill_id: str = "") -> None:
        self.text = text
        self.plan = plan
        self.retrieved = retrieved
        self.skill_id = skill_id


# ---------------------------------------------------------------------------
# Interruption + backchannel
# ---------------------------------------------------------------------------

class InterruptionController:
    """Detects a user speaking over the agent and coordinates stop/attenuate.

    The architecture supports it from day one; the first release implements the
    signal + handler hook, full audio ducking depends on the audio backend.
    """

    def __init__(self) -> None:
        self._interrupted = False

    def signal(self) -> None:
        self._interrupted = True

    def check_and_clear(self) -> bool:
        hit = self._interrupted
        self._interrupted = False
        return hit

    def start_speaking(self) -> None:
        self._interrupted = False


class BackchannelPolicy:
    """Controlled, optional backchannel behavior (not random LLM text)."""

    def __init__(self, enabled: bool = True, min_utterance_tokens: int = 12) -> None:
        self.enabled = enabled
        self.min_utterance_tokens = min_utterance_tokens

    def maybe_backchannel(self, user_text: str, user_state: UserState | None = None) -> str | None:
        if not self.enabled or len(user_text.split()) < self.min_utterance_tokens:
            return None
        if user_state:
            frustration = user_state.get("frustration").value if user_state.get("frustration") else 0.0
            if frustration > 0.5:
                return "I hear you."
        return "Mm-hm."
