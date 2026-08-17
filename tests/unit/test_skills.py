"""Skill and tool subsystem tests."""

from __future__ import annotations

import asyncio

import pytest

from companion.application.facts import FactWriter
from companion.application.identity import SelfModelService
from companion.skills.base import (
    BaseSkill,
    SkillDecision,
    SkillManifest,
    SkillResult,
)
from companion.skills.permissions import (
    Permission,
    PermissionDenied,
    PermissionManager,
)
from companion.skills.registry import SkillLoader, SkillRegistry
from companion.skills.router import SkillRouter
from companion.tools.base import ToolManifest, ToolRisk, validate_arguments
from companion.tools.builtin import UnsafeExpression, default_tools, evaluate_expression
from companion.tools.registry import ToolInvoker, ToolRegistry

# ---------------------------------------------------------------------------
# permissions
# ---------------------------------------------------------------------------

def test_permissions_default_deny():
    manager = PermissionManager()
    assert manager.allows("anything", Permission.MEMORY_READ)      # baseline grant
    assert not manager.allows("anything", Permission.MEMORY_WRITE)
    assert not manager.allows("anything", Permission.SHELL)
    with pytest.raises(PermissionDenied):
        manager.check("anything", Permission.NETWORK)


def test_permissions_from_config():
    manager = PermissionManager({"grants": {"goals": ["memory.write"]},
                                 "deny": {"recall": ["memory.read"]}})
    assert manager.allows("goals", Permission.MEMORY_WRITE)
    assert not manager.allows("other", Permission.MEMORY_WRITE)
    assert not manager.allows("recall", Permission.MEMORY_READ), "explicit deny must win"


def test_unknown_permission_is_denied_not_crashed():
    manager = PermissionManager({"grants": {"x": ["not.a.permission"]}})
    assert not manager.allows("x", "not.a.permission")


# ---------------------------------------------------------------------------
# registry / validation
# ---------------------------------------------------------------------------

class _NeedsShell(BaseSkill):
    manifest = SkillManifest(id="needs_shell", required_permissions=["shell"])


class _NewerApi(BaseSkill):
    manifest = SkillManifest(id="from_the_future", api_version="9.0")


class _Fine(BaseSkill):
    manifest = SkillManifest(id="fine", name="fine", description="ok")

    async def can_handle(self, context, input):
        return SkillDecision.yes(0.9, "always") if "ping" in input.text else SkillDecision.no("")

    async def execute(self, context, input):
        return SkillResult(text="pong")


def test_skill_missing_permission_is_unavailable_with_a_reason():
    registry = SkillRegistry(PermissionManager())
    record = registry.register(_NeedsShell())
    assert not record.available
    assert "shell" in record.reason
    assert record not in registry.available()


def test_incompatible_api_version_is_rejected():
    registry = SkillRegistry(PermissionManager())
    record = registry.register(_NewerApi())
    assert not record.available
    assert "API" in record.reason


def test_duplicate_skill_id_is_rejected():
    registry = SkillRegistry(PermissionManager())
    registry.register(_Fine())
    with pytest.raises(ValueError):
        registry.register(_Fine())


def test_builtin_skills_all_load_with_project_config():
    from companion.runtime.config import Config

    cfg = Config.load()
    permissions = PermissionManager(cfg.section("skills", default={}) or {})
    tools = ToolRegistry()
    for tool in default_tools():
        tools.register(tool)
    registry = SkillRegistry(permissions, tools=tools)
    SkillLoader(registry).load_package()
    unavailable = [(r.manifest.id, r.reason) for r in registry.all() if not r.available]
    assert not unavailable, f"built-in skills failed to load: {unavailable}"
    assert {"identity", "recall", "calculator", "capabilities",
            "diagnostics", "goals", "datetime", "provenance"} <= set(registry.ids())


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expression,expected", [
    ("1234 * 5678", 7006652),
    ("2+2", 4),
    ("10/4", 2.5),
    ("2**10", 1024),
    ("sqrt(16)", 4.0),
    ("(3+4)*5", 35),
])
def test_calculator_is_exact(expression, expected):
    assert evaluate_expression(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo hi')",
    "open('/etc/passwd')",
    "9**9**9",
    "1/0",
    "eval('2+2')",
    "[].__class__",
    "lambda: 1",
])
def test_calculator_refuses_unsafe_input(expression):
    with pytest.raises(UnsafeExpression):
        evaluate_expression(expression)


def test_tool_argument_validation():
    manifest = ToolManifest(
        id="t",
        parameters={"type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"]},
    )
    assert validate_arguments(manifest, {"n": "5"}) == {"n": 5}
    with pytest.raises(ValueError):
        validate_arguments(manifest, {})                  # missing required
    with pytest.raises(ValueError):
        validate_arguments(manifest, {"n": 1, "extra": 2})  # unknown arg
    with pytest.raises(ValueError):
        validate_arguments(manifest, {"n": "abc"})          # wrong type


class _SideEffecting:
    manifest = ToolManifest(id="danger", side_effects=True, risk=ToolRisk.HIGH,
                            parameters={"type": "object", "properties": {}})
    ran = False

    async def run(self):
        _SideEffecting.ran = True
        return "done"


def test_side_effecting_tool_requires_confirmation():
    registry = ToolRegistry()
    registry.register(_SideEffecting())
    invoker = ToolInvoker(registry, PermissionManager())
    result = asyncio.run(invoker.invoke("danger", caller="x"))
    assert not result.ok
    assert "confirmation" in result.error
    assert _SideEffecting.ran is False, "tool executed without confirmation"

    approved = ToolInvoker(registry, PermissionManager(), confirm=lambda *_: True)
    assert asyncio.run(approved.invoke("danger", caller="x")).ok
    assert _SideEffecting.ran is True


class _Slow:
    manifest = ToolManifest(id="slow", timeout_s=0.05,
                            parameters={"type": "object", "properties": {}})

    async def run(self):
        await asyncio.sleep(5)


def test_tool_timeout_is_enforced():
    registry = ToolRegistry()
    registry.register(_Slow())
    result = asyncio.run(ToolInvoker(registry).invoke("slow"))
    assert not result.ok and "timed out" in result.error


def test_tool_permission_is_checked_against_caller():
    registry = ToolRegistry()
    for tool in default_tools():
        registry.register(tool)
    invoker = ToolInvoker(registry, PermissionManager())
    # system_probe requires runtime.inspect, which is not granted by default.
    result = asyncio.run(invoker.invoke("system_probe", caller="nosy"))
    assert not result.ok and "permission" in result.error


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------

class _Exploding(BaseSkill):
    manifest = SkillManifest(id="boom")

    async def can_handle(self, context, input):
        return SkillDecision.yes(0.99, "claims everything")

    async def execute(self, context, input):
        raise RuntimeError("kaboom")


def test_router_picks_the_most_confident_skill():
    registry = SkillRegistry(PermissionManager())
    registry.register(_Fine())
    router = SkillRouter(registry, PermissionManager())
    outcome = asyncio.run(router.handle("ping"))
    assert outcome.handled and outcome.skill_id == "fine" and outcome.text == "pong"


def test_router_returns_unhandled_when_no_skill_claims_the_turn():
    registry = SkillRegistry(PermissionManager())
    registry.register(_Fine())
    outcome = asyncio.run(SkillRouter(registry).handle("tell me a story"))
    assert not outcome.handled


def test_router_survives_a_failing_skill():
    """A broken skill degrades to conversation; it must not break the turn."""
    registry = SkillRegistry(PermissionManager())
    registry.register(_Exploding())
    outcome = asyncio.run(SkillRouter(registry).handle("anything"))
    assert not outcome.handled
    assert "kaboom" in outcome.error
    assert registry.get("boom").stats["failures"] == 1


def test_skill_memory_writes_are_namespaced(graph, clock):
    """A skill cannot overwrite core user facts through its memory facade."""
    self_model = SelfModelService(graph, clock, configured_name="Companion")
    self_model.load()
    writer = FactWriter(graph, clock)
    permissions = PermissionManager({"grants": {"notes": ["memory.write"]}})
    registry = SkillRegistry(permissions)
    router = SkillRouter(registry, permissions, graph=graph, fact_writer=writer,
                         self_model=self_model, clock=clock)
    context = router.context_for("notes")
    context.memory.remember("colour", "green")

    user = self_model.user_entity_id()
    assert writer.current(user, "favorite:color") is None
    assert writer.current(user, "skill:notes:colour").value == "green"


def test_skill_without_write_permission_cannot_write(graph, clock):
    self_model = SelfModelService(graph, clock, configured_name="Companion")
    self_model.load()
    permissions = PermissionManager()
    router = SkillRouter(SkillRegistry(permissions), permissions, graph=graph,
                         fact_writer=FactWriter(graph, clock), self_model=self_model,
                         clock=clock)
    context = router.context_for("sneaky")
    with pytest.raises(PermissionDenied):
        context.memory.remember("anything", "value")
