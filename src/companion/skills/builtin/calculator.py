"""Calculator skill: exact arithmetic through the calculator tool.

Demonstrates the skill -> tool boundary. The skill parses intent and formats
the answer; the arithmetic itself happens in a declared, permissioned,
time-bounded tool. A 1.7B model asked to multiply six-digit numbers will
confidently produce a wrong answer, so this never reaches the model.
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
    id="calculator",
    name="calculator",
    version="1.0.0",
    description="Calculates arithmetic exactly (+ - * / ^, sqrt, log, trig).",
    capabilities=["arithmetic"],
    required_permissions=[],
    required_tools=["calculator"],
    keywords=["calculate", "compute", "math", "plus", "times", "divided"],
    examples=["Calculate 1234 * 5678", "What is 15% of 240?", "sqrt(2)"],
)

_TRIGGER = re.compile(
    r"\b(calculate|compute|what(?:'s| is)|how much is|evaluate|work out|solve)\b",
    re.IGNORECASE,
)
_EXPRESSION = re.compile(r"[0-9][0-9\s+\-*/^%().,]*[0-9)]|\b(?:sqrt|log|sin|cos|tan)\s*\([^)]*\)")
_PERCENT_OF = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

# Words that mean the sentence is prose that happens to contain digits.
_NOT_MATH = re.compile(
    r"\b(remember|remind|my|your|name|favou?rite|feel|think|project|goal|schedule)\b",
    re.IGNORECASE,
)

_WORD_OPS = (
    (re.compile(r"\bplus\b|\badded to\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b|\bless\b", re.IGNORECASE), "-"),
    (re.compile(r"\btimes\b|\bmultiplied by\b", re.IGNORECASE), "*"),
    (re.compile(r"\bdivided by\b|\bover\b", re.IGNORECASE), "/"),
    (re.compile(r"\bto the power of\b|\braised to\b", re.IGNORECASE), "**"),
)


def extract_expression(text: str) -> str:
    """Pull an arithmetic expression out of a sentence, or return ''."""
    raw = (text or "").strip()
    if not raw:
        return ""
    percent = _PERCENT_OF.search(raw)
    if percent:
        return f"({percent.group(1)}/100)*{percent.group(2)}"
    normalised = raw
    for pattern, symbol in _WORD_OPS:
        normalised = pattern.sub(symbol, normalised)
    normalised = normalised.replace("^", "**").replace("×", "*").replace("÷", "/")
    candidates = _EXPRESSION.findall(normalised)
    if not candidates:
        return ""
    best = max(candidates, key=len).strip(" .,")
    # Thousands separators would parse as a tuple; digits-only groups of three.
    best = re.sub(r"(?<=\d),(?=\d{3}\b)", "", best)
    if not re.search(r"[+\-*/%(]", best):
        return ""
    return best


class CalculatorSkill(BaseSkill):
    manifest = MANIFEST

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        text = (input.text or "").strip()
        if not text or _NOT_MATH.search(text):
            return SkillDecision.no("prose, not arithmetic")
        expression = extract_expression(text)
        if not expression:
            return SkillDecision.no("no arithmetic expression found")
        explicit = bool(_TRIGGER.search(text))
        return SkillDecision.yes(
            0.95 if explicit else 0.7,
            "arithmetic expression detected",
            expression=expression,
        )

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        expression = input.args.get("expression") or extract_expression(input.text)
        if not expression:
            return SkillResult.failure("no expression to evaluate")
        if context.tools is None:
            return SkillResult.failure("calculator tool is unavailable")
        result = await context.tools.invoke("calculator", caller=self.manifest.id,
                                            expression=expression)
        if not result.ok:
            return SkillResult.failure(result.error)
        value = result.value
        rendered = _render(value)
        return SkillResult(
            text=f"{expression} = {rendered}",
            data={"expression": expression, "value": value},
            produced="number",
        )


def _render(value) -> str:
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.6g}"
    return f"{value:,}"


SKILLS = [CalculatorSkill]
