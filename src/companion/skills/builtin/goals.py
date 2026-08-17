"""Goals skill: durable goals and tasks, distinct from generic memories.

A goal is not a fact and not an episode. It has status, priority, progress and
a lifecycle, so it lives in its own table and is answered from there — which is
what lets the companion open a session with "last time you were working on X".
"""

from __future__ import annotations

import re

from companion.core.ids import new_goal_id
from companion.domain.graph import Goal
from companion.skills.base import (
    BaseSkill,
    SkillContext,
    SkillDecision,
    SkillInput,
    SkillManifest,
    SkillResult,
)

MANIFEST = SkillManifest(
    id="goals",
    name="goals",
    version="1.0.0",
    description="Tracks your goals and what we planned to do next.",
    capabilities=["list_goals", "add_goal", "complete_goal"],
    required_permissions=["memory.read", "memory.write"],
    memory_read_scopes=["goal"],
    memory_write_scopes=["goal"],
    keywords=["goal", "task", "next", "todo", "plan"],
    examples=["What were we going to work on next?", "What are my goals?",
              "Mark the retrieval goal as done."],
)

_LIST = re.compile(
    r"\bwhat\s+(?:were\s+we|are\s+we|was\s+i|am\s+i)\s+(?:going\s+to|planning\s+to|supposed\s+to)\b|"
    r"\bwhat(?:'s|\s+is)?\s+next\b|"
    r"\bwhat\s+are\s+my\s+(?:goals|tasks|todos)\b|"
    r"\blist\s+(?:my\s+)?(?:goals|tasks)\b|"
    r"\bwhat\s+should\s+i\s+work\s+on\b",
    re.IGNORECASE,
)
_ADD = re.compile(
    r"\b(?:add|create|set)\s+(?:a\s+)?(?:goal|task)\s*(?:to|:)?\s*(.+)",
    re.IGNORECASE,
)
_COMPLETE = re.compile(
    r"\b(?:mark|set)\s+(.+?)\s+(?:as\s+)?(?:done|complete|completed|finished)\b|"
    r"\bi\s+(?:finished|completed|did)\s+(.+)",
    re.IGNORECASE,
)


class GoalsSkill(BaseSkill):
    manifest = MANIFEST

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        text = (input.text or "").strip()
        if not text:
            return SkillDecision.no("empty")
        m = _ADD.search(text)
        if m:
            return SkillDecision.yes(0.9, "adds a goal", action="add", name=m.group(1).strip(" .!"))
        m = _COMPLETE.search(text)
        if m:
            name = (m.group(1) or m.group(2) or "").strip(" .!")
            if name:
                return SkillDecision.yes(0.85, "completes a goal", action="complete", name=name)
        if _LIST.search(text):
            return SkillDecision.yes(0.85, "lists goals", action="list")
        return SkillDecision.no("not a goal request")

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        memory = context.memory
        if memory is None:
            return SkillResult.failure("memory is unavailable")
        action = input.args.get("action", "list")
        if action == "add":
            return self._add(context, memory, input)
        if action == "complete":
            return self._complete(context, memory, input)
        return self._list(memory)

    def _list(self, memory) -> SkillResult:
        goals = memory.goals(status="active")
        if not goals:
            return SkillResult(text="You don't have any open goals on record.",
                               data={"goals": []})
        lines = ["Open goals:"]
        for goal in sorted(goals, key=lambda g: -g.priority)[:8]:
            progress = f" ({goal.progress:.0%} done)" if goal.progress > 0 else ""
            lines.append(f"- {goal.name}{progress}")
        return SkillResult(text="\n".join(lines),
                           data={"goals": [g.to_dict() for g in goals]})

    def _add(self, context: SkillContext, memory, input: SkillInput) -> SkillResult:
        context.require("memory.write")
        name = (input.args.get("name") or "").strip()
        if not name:
            return SkillResult.failure("no goal name given")
        for existing in memory.goals(status="active"):
            if existing.name.lower() == name.lower():
                return SkillResult(text=f"That goal is already on your list: {existing.name}.",
                                   data={"goal_id": existing.id, "created": False})
        now = context.clock.now_iso() if context.clock else ""
        goal = Goal(id=new_goal_id(), name=name[:120], description=input.text,
                    status="active", priority=0.6, confidence=0.9,
                    source_episode_id=input.episode_id, created_at=now, updated_at=now)
        memory.save_goal(goal)
        return SkillResult(text=f"Added the goal: {goal.name}.",
                           data={"goal_id": goal.id, "created": True})

    def _complete(self, context: SkillContext, memory, input: SkillInput) -> SkillResult:
        context.require("memory.write")
        name = (input.args.get("name") or "").lower()
        for goal in memory.goals(status="active"):
            if name in goal.name.lower() or goal.name.lower() in name:
                goal.status = "completed"
                goal.progress = 1.0
                goal.updated_at = context.clock.now_iso() if context.clock else ""
                memory.save_goal(goal)
                return SkillResult(text=f"Marked '{goal.name}' as done.",
                                   data={"goal_id": goal.id})
        return SkillResult(text=f"I don't have an open goal matching '{name}'.",
                           data={"goal_id": None})


SKILLS = [GoalsSkill]
