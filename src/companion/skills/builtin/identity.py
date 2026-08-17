"""Identity skill: answers questions about who the agent and user are.

This exists as a skill rather than a special case inside the conversation
service because identity questions have an exact answer that lives in the
graph. Routing them through a 0.6B model and hoping the context survives
truncation is how a companion ends up forgetting its own name.
"""

from __future__ import annotations

from companion.application.identity import (
    asks_agent_name,
    asks_previous_agent_name,
)
from companion.skills.base import (
    BaseSkill,
    SkillContext,
    SkillDecision,
    SkillInput,
    SkillManifest,
    SkillResult,
)

MANIFEST = SkillManifest(
    id="identity",
    name="identity",
    version="1.0.0",
    description="Answers who I am, what you named me, and what I was called before.",
    capabilities=["answer_agent_name", "answer_name_history", "answer_user_name"],
    required_permissions=["memory.read"],
    memory_read_scopes=["self:name", "user:name"],
    keywords=["name", "who are you", "call you"],
    examples=["What's your name?", "What was your previous name?", "Who am I?"],
)

_ASK_USER_NAME = ("what's my name", "whats my name", "what is my name", "who am i",
                  "do you know my name", "do you remember my name")


class IdentitySkill(BaseSkill):
    manifest = MANIFEST

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        text = (input.text or "").strip().lower()
        if not text:
            return SkillDecision.no("empty")
        if asks_previous_agent_name(input.text):
            return SkillDecision.yes(0.95, "asks for previous agent name", query="previous_name")
        if asks_agent_name(input.text):
            return SkillDecision.yes(0.95, "asks for agent name", query="name")
        if any(p in text for p in _ASK_USER_NAME):
            return SkillDecision.yes(0.9, "asks for user name", query="user_name")
        return SkillDecision.no("not an identity question")

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        self_model = context.self_model
        if self_model is None:
            return SkillResult.failure("identity is unavailable")
        context.require("memory.read")
        query = input.args.get("query", "name")

        if query == "name":
            name = self_model.name
            model = self_model.model()
            if model.name_source in ("persisted", "explicit_user_statement"):
                text = f"My name is {name} — you named me that."
            elif model.name_source == "config":
                text = (f"My name is {name}, which came from my configuration. "
                        f"You can rename me any time.")
            else:
                text = f"My name is {name}."
            return SkillResult(text=text, data={"name": name,
                                                "source": model.name_source})

        if query == "previous_name":
            previous = self_model.previous_name()
            if not previous:
                return SkillResult(
                    text=f"You have only ever called me {self_model.name}.",
                    data={"previous": None},
                )
            history = self_model.name_history()
            return SkillResult(
                text=f"Before {self_model.name}, you called me {previous}.",
                data={"previous": previous, "history": history},
            )

        user_name = self_model.user_name()
        if not user_name:
            return SkillResult(
                text="I don't have your name yet — tell me and I'll remember it.",
                data={"user_name": None},
            )
        return SkillResult(text=f"You're {user_name}.", data={"user_name": user_name})


SKILLS = [IdentitySkill]
