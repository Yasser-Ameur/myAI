"""Skill contracts: manifest, context, input/output, and the Skill protocol.

Skills are bound to *capabilities*, never to a concrete model. A skill receives
a SkillContext giving it exactly the slices of cognition it declared — memory
access through a permissioned facade, the self-model, the current user state,
the clock, and a tool invoker. It never reaches into the database directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from companion.skills.permissions import Permission, PermissionDenied

SKILL_API_VERSION = "1.0"


@dataclass
class SkillManifest:
    """Everything a skill declares about itself before it is allowed to run."""

    id: str
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    api_version: str = SKILL_API_VERSION
    capabilities: list[str] = field(default_factory=list)
    required_permissions: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    memory_read_scopes: list[str] = field(default_factory=list)
    memory_write_scopes: list[str] = field(default_factory=list)
    configuration: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    # Routing hint only. A skill still decides for itself in can_handle();
    # this just keeps obviously irrelevant skills from being polled.
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name or self.id,
            "version": self.version,
            "description": self.description,
            "api_version": self.api_version,
            "capabilities": list(self.capabilities),
            "required_permissions": list(self.required_permissions),
            "required_tools": list(self.required_tools),
            "memory_read_scopes": list(self.memory_read_scopes),
            "memory_write_scopes": list(self.memory_write_scopes),
            "dependencies": list(self.dependencies),
            "examples": list(self.examples),
            "keywords": list(self.keywords),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillManifest":
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            version=str(d.get("version", "1.0.0")),
            description=str(d.get("description", "")),
            api_version=str(d.get("api_version", SKILL_API_VERSION)),
            capabilities=list(d.get("capabilities", []) or []),
            required_permissions=list(d.get("permissions", d.get("required_permissions", [])) or []),
            required_tools=list(d.get("tools", d.get("required_tools", [])) or []),
            memory_read_scopes=list((d.get("memory") or {}).get("read", []) or []),
            memory_write_scopes=list((d.get("memory") or {}).get("write", []) or []),
            configuration=dict(d.get("configuration", {}) or {}),
            dependencies=list(d.get("dependencies", []) or []),
            examples=list(d.get("examples", []) or []),
            keywords=list(d.get("keywords", []) or []),
        )


@dataclass
class SkillInput:
    text: str = ""
    args: dict = field(default_factory=dict)
    episode_id: str = ""
    session_id: str = ""
    turn_id: str = ""


@dataclass
class SkillDecision:
    """A skill's own judgement about whether it should take this turn."""

    can_handle: bool = False
    confidence: float = 0.0
    reason: str = ""
    args: dict = field(default_factory=dict)

    @classmethod
    def no(cls, reason: str = "") -> "SkillDecision":
        return cls(can_handle=False, confidence=0.0, reason=reason)

    @classmethod
    def yes(cls, confidence: float, reason: str = "", **args) -> "SkillDecision":
        return cls(can_handle=True, confidence=confidence, reason=reason, args=args)


@dataclass
class SkillResult:
    """What a skill produced, and what it wants the companion to say."""

    text: str = ""
    data: dict = field(default_factory=dict)
    success: bool = True
    error: str = ""
    needs_confirmation: bool = False
    confirmation_prompt: str = ""
    # Structured output for composition: the next skill consumes `data`.
    produced: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "data": self.data,
            "success": self.success,
            "error": self.error,
            "needs_confirmation": self.needs_confirmation,
            "produced": self.produced,
        }

    @classmethod
    def failure(cls, error: str) -> "SkillResult":
        return cls(text="", success=False, error=error)


class SkillMemory:
    """Permissioned, scoped memory facade handed to a skill.

    Skills never touch the graph directly. Reads and writes go through here so
    that permissions are enforced and every skill write is attributable.
    """

    def __init__(self, skill_id: str, graph, permissions, fact_writer,
                 self_model=None, clock=None) -> None:
        self._skill_id = skill_id
        self._graph = graph
        self._permissions = permissions
        self._writer = fact_writer
        self._self_model = self_model
        self._clock = clock

    def _check(self, permission: Permission) -> None:
        self._permissions.check(self._skill_id, permission)

    # -- reads ------------------------------------------------------------

    def facts(self, subject_id: str = "", predicate: str = "") -> list:
        self._check(Permission.MEMORY_READ)
        return self._graph.list_facts(subject_id=subject_id, predicate=predicate) or []

    def memories(self, status: str = "", limit: int = 50) -> list:
        self._check(Permission.MEMORY_READ)
        return self._graph.list_memories(status=status, limit=limit) or []

    def goals(self, status: str = "active") -> list:
        self._check(Permission.MEMORY_READ)
        return self._graph.list_goals(status=status) or []

    def user_entity_id(self) -> str:
        self._check(Permission.MEMORY_READ)
        return self._self_model.user_entity_id() if self._self_model else ""

    def entity_name(self, entity_id: str) -> str:
        self._check(Permission.MEMORY_READ)
        if not entity_id:
            return ""
        entity = self._graph.get_entity(entity_id)
        return entity.name if entity is not None else ""

    def current_fact(self, predicate: str, subject_id: str = ""):
        self._check(Permission.MEMORY_READ)
        subject = subject_id or self.user_entity_id()
        return self._writer.current(subject, predicate)

    def fact_history(self, predicate: str, subject_id: str = "") -> list:
        self._check(Permission.MEMORY_READ)
        subject = subject_id or self.user_entity_id()
        return self._writer.history(subject, predicate)

    # -- writes -----------------------------------------------------------

    def remember(self, predicate: str, value: str, *, confidence: float = 0.7,
                 importance: float = 0.5, subject_id: str = "",
                 episode_id: str = "") -> Any:
        """Write a fact into the skill's own namespace.

        The predicate is prefixed with ``skill:<id>:`` unless the skill
        declared a broader write scope, so a misbehaving skill cannot
        overwrite core user facts.
        """
        self._check(Permission.MEMORY_WRITE)
        scoped = predicate if predicate.startswith(f"skill:{self._skill_id}:") \
            else f"skill:{self._skill_id}:{predicate}"
        return self._writer.assert_fact(
            subject_id=subject_id or self.user_entity_id(),
            predicate=scoped,
            value=value,
            confidence=confidence,
            importance=importance,
            provenance="system",
            source_episode_id=episode_id,
            evidence_text=f"written by skill {self._skill_id}",
        )

    def save_goal(self, goal) -> None:
        self._check(Permission.MEMORY_WRITE)
        self._graph.upsert_goal(goal)

    def save_memory(self, memory) -> None:
        self._check(Permission.MEMORY_WRITE)
        self._graph.add_memory(memory)

    def recall(self, predicate: str, subject_id: str = ""):
        self._check(Permission.MEMORY_READ)
        scoped = predicate if predicate.startswith(f"skill:{self._skill_id}:") \
            else f"skill:{self._skill_id}:{predicate}"
        return self._writer.current(subject_id or self.user_entity_id(), scoped)


@dataclass
class SkillContext:
    """The slice of the companion a skill is allowed to see."""

    skill_id: str
    memory: SkillMemory | None = None
    self_model: Any = None
    user_state: Any = None
    clock: Any = None
    tools: Any = None
    registry: Any = None          # SkillRegistry, for self-describing skills
    runtime: Any = None           # health/diagnostics provider, permissioned
    config: dict = field(default_factory=dict)

    def require(self, permission: Permission | str) -> None:
        if self.memory is None:
            raise PermissionDenied(self.skill_id, permission)
        self.memory._check(
            permission if isinstance(permission, Permission)
            else Permission.parse(permission)
        )


@runtime_checkable
class Skill(Protocol):
    manifest: SkillManifest

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        ...

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        ...


class BaseSkill:
    """Convenience base: manifest-driven keyword prefilter and safe defaults."""

    manifest: SkillManifest

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        return SkillDecision.no("not implemented")

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        return SkillResult.failure("not implemented")
