"""The skill subsystem: bounded, declared, permissioned capabilities.

A skill is not a prompt. It is an executable capability with a manifest, a
permission set, typed input and output, its own tests, and — when it needs one
— its own scoped slice of memory. The cognitive loop routes to a skill when a
skill can answer better than free generation; everything else falls through to
conversation.

Adding a skill requires a manifest, an implementation and tests. It requires no
change to the core cognition, which is the point.
"""

from companion.skills.base import (
    Skill,
    SkillContext,
    SkillDecision,
    SkillInput,
    SkillManifest,
    SkillResult,
)
from companion.skills.permissions import (
    Permission,
    PermissionDenied,
    PermissionManager,
)
from companion.skills.registry import SkillLoader, SkillRegistry, SkillValidator
from companion.skills.router import SkillOutcome, SkillRouter

__all__ = [
    "Permission",
    "PermissionDenied",
    "PermissionManager",
    "Skill",
    "SkillContext",
    "SkillDecision",
    "SkillInput",
    "SkillLoader",
    "SkillManifest",
    "SkillOutcome",
    "SkillRegistry",
    "SkillResult",
    "SkillRouter",
    "SkillValidator",
]
