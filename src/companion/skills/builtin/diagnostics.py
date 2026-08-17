"""Self-diagnostic skill: lets the companion explain its own failures.

"Why can't you speak right now?" is answered by inspecting the live runtime —
which model slots resolved, which are fallbacks, whether playback exists —
rather than by the model speculating about itself.
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
    id="diagnostics",
    name="diagnostics",
    version="1.0.0",
    description="Explains my own health: what's working, what's degraded and why.",
    capabilities=["self_diagnosis"],
    required_permissions=["runtime.inspect"],
    keywords=["why can't you", "status", "health", "broken", "not working"],
    examples=["Why can't you speak right now?", "Are you working properly?"],
)

_WHY_NOT = re.compile(
    r"\bwhy\s+(?:can'?t|cannot|don'?t|aren'?t)\s+you\b|"
    r"\b(?:what'?s|whats)\s+(?:wrong|broken|your\s+status)\b|"
    r"\bare\s+you\s+(?:ok|okay|working|healthy|broken)\b|"
    r"\b(?:system|runtime|health)\s+(?:status|check)\b|"
    r"\bdiagnos(?:e|tics|tic)\b",
    re.IGNORECASE,
)

_SUBSYSTEM_WORDS = {
    "speak": "tts.default", "speech": "tts.default", "talk": "tts.default",
    "voice": "tts.default", "say": "tts.default",
    "hear": "stt.default", "listen": "stt.default", "understand me": "stt.default",
    "see": "vision.face", "camera": "vision.face", "look": "vision.face",
    "remember": "storage", "memory": "storage",
    "think": "llm.default", "answer": "llm.default", "respond": "llm.default",
}


class DiagnosticsSkill(BaseSkill):
    manifest = MANIFEST

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        text = (input.text or "")
        if not _WHY_NOT.search(text):
            return SkillDecision.no("not a diagnostic question")
        if context.runtime is None:
            return SkillDecision.no("no runtime health provider")
        subsystem = ""
        lowered = text.lower()
        for word, slot in _SUBSYSTEM_WORDS.items():
            if word in lowered:
                subsystem = slot
                break
        return SkillDecision.yes(0.9, "asks about own health", subsystem=subsystem)

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        context.require("runtime.inspect")
        try:
            health = context.runtime()
        except Exception as exc:
            return SkillResult.failure(f"could not read runtime health: {exc}")

        subsystem = input.args.get("subsystem") or ""
        slots = health.get("slots", {})
        if subsystem == "storage":
            store = health.get("storage", {})
            if store.get("ok"):
                return SkillResult(
                    text="My memory store is healthy — facts I commit survive a restart.",
                    data=store)
            return SkillResult(
                text=f"My memory is degraded: {store.get('reason', 'unknown')}. "
                     f"Anything you tell me right now will not be remembered.",
                data=store)

        if subsystem and subsystem in slots:
            return SkillResult(text=_explain_slot(subsystem, slots[subsystem]),
                               data={subsystem: slots[subsystem]})

        lines = ["Here's my current state:"]
        store = health.get("storage", {})
        lines.append(f"- memory store: {'ok' if store.get('ok') else 'DEGRADED — ' + str(store.get('reason'))}")
        for slot, info in sorted(slots.items()):
            lines.append("- " + _explain_slot(slot, info))
        degraded = health.get("degraded")
        if degraded:
            lines.append(f"- overall: degraded ({degraded})")
        return SkillResult(text="\n".join(lines), data=health)


def _explain_slot(slot: str, info: dict) -> str:
    name = slot.split(".")[0]
    if not info.get("present"):
        return f"{slot}: not configured, so I cannot do {name} at all"
    if info.get("fallback"):
        return (f"{slot}: running on a deterministic fallback, not a real model "
                f"({info.get('model', '?')} unavailable) — output will be placeholder")
    if info.get("load_error"):
        return f"{slot}: failed to load — {info['load_error']}"
    state = "loaded" if info.get("loaded") else "installed but not loaded yet"
    return f"{slot}: {info.get('provider', '?')}/{info.get('model', '?')}, {state}"


SKILLS = [DiagnosticsSkill]
