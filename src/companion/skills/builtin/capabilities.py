"""Capabilities skill: answers "what can you do?" from the live registry.

The answer is generated from the registry at the moment of asking, so it can
never drift from reality. Skills that failed validation are reported as
unavailable *with the reason*, which is far more useful than omitting them.
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
    id="capabilities",
    name="capabilities",
    version="1.0.0",
    description="Lists what I can actually do, from my live skill registry.",
    capabilities=["self_description"],
    required_permissions=[],
    keywords=["what can you do", "abilities", "skills", "help"],
    examples=["What can you do?", "What skills do you have?"],
)

_TRIGGER = re.compile(
    r"\bwhat\s+can\s+you\s+do\b|"
    r"\bwhat\s+(?:skills|abilities|capabilities)\s+do\s+you\s+have\b|"
    r"\blist\s+your\s+(?:skills|capabilities)\b|"
    r"\bwhat\s+are\s+you\s+(?:able|capable)\s+(?:to\s+do|of)\b|"
    r"\bwhat\s+skills\s+(?:do\s+you\s+have|are\s+available)\b",
    re.IGNORECASE,
)


class CapabilitiesSkill(BaseSkill):
    manifest = MANIFEST

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        if _TRIGGER.search(input.text or ""):
            return SkillDecision.yes(0.95, "asks what the companion can do")
        return SkillDecision.no("not a capability question")

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        registry = context.registry
        if registry is None:
            return SkillResult.failure("skill registry is unavailable")
        records = registry.describe()
        available = [r for r in records if r["available"]]
        blocked = [r for r in records if not r["available"]]

        lines = ["Here's what I can actually do right now:"]
        for record in sorted(available, key=lambda r: r["name"]):
            lines.append(f"- {record['name']}: {record['description']}")
        if blocked:
            lines.append("")
            lines.append("Not available at the moment:")
            for record in sorted(blocked, key=lambda r: r["name"]):
                lines.append(f"- {record['name']}: {record['reason']}")
        return SkillResult(
            text="\n".join(lines),
            data={"available": [r["id"] for r in available],
                  "unavailable": [r["id"] for r in blocked]},
        )


SKILLS = [CapabilitiesSkill]
