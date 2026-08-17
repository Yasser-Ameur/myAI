"""Recall skill: answers direct questions about stored facts, with provenance.

Handles the two question shapes a memory system must never get wrong:

* "what is my favourite colour?"  -> the fact that is true *now*
* "what used to be my favourite colour?" -> the fact that was superseded

Both are answered from the graph. If the graph does not know, the skill says
so rather than letting the model improvise — a confabulated memory is worse
than an admitted gap.
"""

from __future__ import annotations

import re

from companion.skills.base import (
    BaseSkill,
    SkillContext,
    SkillDecision,
    SkillInput,
    SkillManifest,
    SkillResult,
)

MANIFEST = SkillManifest(
    id="recall",
    name="recall",
    version="1.0.0",
    description="Recalls what you've told me — current values and what they used to be.",
    capabilities=["recall_fact", "recall_history", "explain_evidence"],
    required_permissions=["memory.read"],
    memory_read_scopes=["favorite:*", "opinion:*", "works_on", "lives_in"],
    keywords=["favorite", "favourite", "remember", "used to", "what do you know"],
    examples=["What's my favorite color?", "What used to be my favorite color?",
              "What do you know about me?"],
)

_PAST = re.compile(
    r"\b(used\s+to\s+be|was\s+my|before|previously|earlier|old|former)\b", re.IGNORECASE
)
# Allows the filler that separates "what" from "my" in past-tense forms:
# "what's my favourite colour" and "what used to be my favourite colour".
_FAVOURITE_Q = re.compile(
    r"\bwhat\b.{0,28}?\bmy\s+(?:favou?rite|preferred)\s+([a-z][a-z ]{1,24}?)\s*\??\s*$",
    re.IGNORECASE,
)
_ABOUT_ME = re.compile(
    r"\bwhat\s+do\s+you\s+(?:know|remember)\s+about\s+me\b|"
    r"\btell\s+me\s+about\s+(?:me|myself)\b|"
    r"\bwhat\s+have\s+i\s+told\s+you\b",
    re.IGNORECASE,
)
_WORKING_ON = re.compile(
    r"\bwhat\s+(?:am\s+i|was\s+i|are\s+we|were\s+we)\s+working\s+on\b|"
    r"\bwhat'?s?\s+my\s+project\b|\bwhat\s+project\b",
    re.IGNORECASE,
)


class RecallSkill(BaseSkill):
    manifest = MANIFEST

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        text = (input.text or "").strip()
        if not text:
            return SkillDecision.no("empty")
        m = _FAVOURITE_Q.search(text)
        if m:
            category = m.group(1).strip().lower().replace(" ", "_")
            return SkillDecision.yes(
                0.92, "asks for a stored favourite",
                predicate=f"favorite:{category}",
                category=m.group(1).strip(),
                past=bool(_PAST.search(text)),
            )
        if _WORKING_ON.search(text):
            return SkillDecision.yes(0.85, "asks what the user is working on", query="works_on")
        if _ABOUT_ME.search(text):
            return SkillDecision.yes(0.8, "asks for a profile summary", query="about_me")
        return SkillDecision.no("not a recall question")

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        memory = context.memory
        if memory is None:
            return SkillResult.failure("memory is unavailable")
        query = input.args.get("query")
        if query == "about_me":
            return self._about_me(memory)
        if query == "works_on":
            return self._works_on(memory)
        return self._favourite(memory, input.args)

    # -- handlers ---------------------------------------------------------

    def _favourite(self, memory, args: dict) -> SkillResult:
        predicate = args.get("predicate", "")
        category = args.get("category", "that")
        if args.get("past"):
            history = [f for f in memory.fact_history(predicate) if f.valid_to and f.value]
            if not history:
                return SkillResult(
                    text=f"You've only ever told me one favorite {category}"
                         f"{', ' + self._current_value(memory, predicate) if self._current_value(memory, predicate) else ''}.",
                    data={"previous": None},
                )
            previous = max(history, key=lambda f: f.valid_to)
            return SkillResult(
                text=f"Your favorite {category} used to be {previous.value}.",
                data={"previous": previous.value, "until": previous.valid_to},
            )
        value = self._current_value(memory, predicate)
        if not value:
            return SkillResult(
                text=f"You haven't told me your favorite {category} yet.",
                data={"value": None},
            )
        return SkillResult(text=f"Your favorite {category} is {value}.",
                           data={"value": value})

    @staticmethod
    def _current_value(memory, predicate: str) -> str:
        fact = memory.current_fact(predicate)
        return (fact.value or "").strip() if fact is not None else ""

    def _works_on(self, memory) -> SkillResult:
        facts = [f for f in memory.facts(subject_id=memory.user_entity_id(),
                                         predicate="works_on") if not f.valid_to]
        names = []
        for fact in facts:
            if fact.value:
                names.append(fact.value)
            elif fact.object_id:
                names.append(_entity_name(memory, fact.object_id))
        names = [n for n in names if n]
        goals = [g.name for g in memory.goals(status="active")][:3]
        if not names and not goals:
            return SkillResult(text="You haven't told me what you're working on yet.",
                               data={"projects": [], "goals": []})
        parts = []
        if names:
            parts.append("You're working on " + ", ".join(dict.fromkeys(names)) + ".")
        if goals:
            parts.append("Open goals: " + "; ".join(goals) + ".")
        return SkillResult(text=" ".join(parts),
                           data={"projects": names, "goals": goals})

    def _about_me(self, memory) -> SkillResult:
        user_id = memory.user_entity_id()
        facts = [f for f in memory.facts(subject_id=user_id) if not f.valid_to]
        if not facts:
            return SkillResult(
                text="I don't know much about you yet — nothing has been stored so far.",
                data={"facts": []},
            )
        lines = []
        for fact in sorted(facts, key=lambda f: -f.importance)[:8]:
            lines.append("- " + _readable_fact(memory, fact))
        goals = [g.name for g in memory.goals(status="active")][:3]
        if goals:
            lines.append("- open goals: " + "; ".join(goals))
        return SkillResult(
            text="Here's what I actually have on record:\n" + "\n".join(lines),
            data={"facts": [f.to_dict() for f in facts]},
        )


def _entity_name(memory, entity_id: str) -> str:
    return memory.entity_name(entity_id)


def _readable_fact(memory, fact) -> str:
    predicate = fact.predicate
    value = fact.value or _entity_name(memory, fact.object_id or "")
    if predicate.startswith("favorite:"):
        return f"favorite {predicate.split(':', 1)[1].replace('_', ' ')}: {value}"
    if predicate.startswith("opinion:"):
        return f"{value} {predicate.split(':', 1)[1]}"
    if predicate.startswith("prefers:"):
        return f"prefers {predicate.split(':', 1)[1]}"
    if predicate.startswith("experience:"):
        return f"has been {predicate.split(':', 1)[1]} with {value}"
    if predicate == "user:name":
        return f"name: {value}"
    if predicate == "works_on":
        return f"working on {value}"
    return f"{predicate.replace('_', ' ')}: {value}"


SKILLS = [RecallSkill]
