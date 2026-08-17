"""Time/date skill.

Small on purpose: it is the minimal worked example of a skill that owns no
state and delegates entirely to a tool. Use it as the template when adding
a new skill.
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
    id="datetime",
    name="time",
    version="1.0.0",
    description="Tells you the current date, time and day of the week.",
    capabilities=["current_time", "current_date"],
    required_tools=["clock"],
    keywords=["time", "date", "today", "day"],
    examples=["What time is it?", "What's today's date?"],
)

_TRIGGER = re.compile(
    r"\bwhat\s+time\s+is\s+it\b|"
    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?(?:current\s+)?time\b|"
    r"\bwhat(?:'s|\s+is)\s+(?:today'?s?\s+)?(?:the\s+)?date\b|"
    r"\bwhat\s+day\s+is\s+it\b|\btoday'?s?\s+date\b",
    re.IGNORECASE,
)


class DateTimeSkill(BaseSkill):
    manifest = MANIFEST

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        text = input.text or ""
        if not _TRIGGER.search(text):
            return SkillDecision.no("not a time question")
        wants_time = bool(re.search(r"\btime\b", text, re.IGNORECASE))
        return SkillDecision.yes(0.9, "asks for date/time",
                                 want="time" if wants_time else "date")

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        if context.tools is None:
            return SkillResult.failure("clock tool is unavailable")
        result = await context.tools.invoke("clock", caller=self.manifest.id)
        if not result.ok:
            return SkillResult.failure(result.error)
        now = result.value
        if input.args.get("want") == "time":
            text = f"It's {now['time']} on {now['weekday']}, {now['date']}."
        else:
            text = f"Today is {now['weekday']}, {now['date']}."
        return SkillResult(text=text, data=now)


SKILLS = [DateTimeSkill]
