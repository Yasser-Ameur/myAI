"""Provenance skill: "why do you think that?"

Every durable claim carries evidence. This skill turns that evidence back into
a sentence, which is what makes the memory system auditable in conversation
rather than only through the CLI.
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
    id="provenance",
    name="provenance",
    version="1.0.0",
    description="Explains why I believe something, with the evidence and when I learned it.",
    capabilities=["explain_belief"],
    required_permissions=["memory.read"],
    keywords=["why do you think", "how do you know", "evidence"],
    examples=["Why do you think that?", "How do you know that?"],
)

_TRIGGER = re.compile(
    r"\bwhy\s+do\s+you\s+(?:think|believe|say)\b|"
    r"\bhow\s+do\s+you\s+know\b|"
    r"\bwhat\s+makes\s+you\s+(?:think|say)\b|"
    r"\bwhere\s+did\s+you\s+(?:learn|get)\s+that\b",
    re.IGNORECASE,
)


class ProvenanceSkill(BaseSkill):
    manifest = MANIFEST

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        if _TRIGGER.search(input.text or ""):
            return SkillDecision.yes(0.88, "asks for provenance")
        return SkillDecision.no("not a provenance question")

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        memory = context.memory
        if memory is None:
            return SkillResult.failure("memory is unavailable")
        facts = [f for f in memory.facts(subject_id=memory.user_entity_id())
                 if not f.valid_to]
        if not facts:
            return SkillResult(
                text="I haven't formed any beliefs about you yet, so there's nothing "
                     "to justify.",
                data={"facts": []})
        recent = sorted(facts, key=lambda f: f.created_at, reverse=True)[:4]
        lines = ["Because you told me:"]
        for fact in recent:
            when = (fact.valid_from or fact.created_at)[:10]
            source = {
                "explicit_user_statement": "you stated it directly",
                "conversation": "it came up in conversation",
                "model_inference": "I inferred it (not stated outright)",
                "hedged": "you mentioned it tentatively",
            }.get(fact.provenance, fact.provenance)
            # Facts point either at a literal value or at an entity; resolve
            # the entity so provenance never renders as an empty claim.
            value = fact.value or memory.entity_name(fact.object_id or "")
            lines.append(
                f"- {fact.predicate.replace(':', ' ').replace('_', ' ')}"
                f" = {value} — {source} on {when}"
                f" (confidence {fact.confidence:.0%})"
            )
        lines.append("Ask me to forget or correct any of these and I will.")
        return SkillResult(text="\n".join(lines),
                           data={"facts": [f.to_dict() for f in recent]})


SKILLS = [ProvenanceSkill]
