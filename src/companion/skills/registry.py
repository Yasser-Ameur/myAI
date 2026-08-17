"""Skill registry, loader and validator.

Skills are discovered at runtime and validated before they are allowed to
serve traffic. A skill whose manifest is malformed, whose API version is
incompatible, or whose required permissions are not granted is registered as
*unavailable* rather than silently dropped — the companion needs to be able to
explain why it cannot do something.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field

from companion.skills.base import SKILL_API_VERSION, Skill, SkillManifest
from companion.skills.permissions import PermissionManager

log = logging.getLogger(__name__)


@dataclass
class SkillRecord:
    skill: Skill
    manifest: SkillManifest
    available: bool = True
    reason: str = ""
    missing_permissions: list[str] = field(default_factory=list)
    missing_tools: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=lambda: {"invocations": 0, "failures": 0})

    def to_dict(self) -> dict:
        return {
            **self.manifest.to_dict(),
            "available": self.available,
            "reason": self.reason,
            "missing_permissions": list(self.missing_permissions),
            "missing_tools": list(self.missing_tools),
            "stats": dict(self.stats),
        }


class SkillValidator:
    """Static checks applied before a skill is allowed to register."""

    def __init__(self, permissions: PermissionManager, tools=None) -> None:
        self._permissions = permissions
        self._tools = tools

    def validate(self, skill: Skill) -> tuple[bool, str, list[str], list[str]]:
        manifest = getattr(skill, "manifest", None)
        if manifest is None or not getattr(manifest, "id", ""):
            return False, "skill has no manifest id", [], []
        if not _compatible(manifest.api_version):
            return (False,
                    f"skill targets API {manifest.api_version}, runtime is {SKILL_API_VERSION}",
                    [], [])
        for hook in ("can_handle", "execute"):
            if not callable(getattr(skill, hook, None)):
                return False, f"skill does not implement {hook}()", [], []

        missing_perms = [p.value for p in
                         self._permissions.missing_for(manifest.id, manifest.required_permissions)]
        missing_tools: list[str] = []
        if self._tools is not None:
            available = set(self._tools.ids())
            missing_tools = [t for t in manifest.required_tools if t not in available]

        if missing_perms or missing_tools:
            parts = []
            if missing_perms:
                parts.append("permissions not granted: " + ", ".join(missing_perms))
            if missing_tools:
                parts.append("tools unavailable: " + ", ".join(missing_tools))
            return False, "; ".join(parts), missing_perms, missing_tools
        return True, "", [], []


def _compatible(api_version: str) -> bool:
    """Same major version is compatible; a newer minor is not."""
    try:
        want_major, want_minor = (int(x) for x in str(api_version).split(".")[:2])
        have_major, have_minor = (int(x) for x in SKILL_API_VERSION.split(".")[:2])
    except (TypeError, ValueError):
        return False
    return want_major == have_major and want_minor <= have_minor


class SkillRegistry:
    def __init__(self, permissions: PermissionManager | None = None, tools=None) -> None:
        self._permissions = permissions or PermissionManager()
        self._validator = SkillValidator(self._permissions, tools)
        self._records: dict[str, SkillRecord] = {}

    def register(self, skill: Skill) -> SkillRecord:
        manifest = getattr(skill, "manifest", None)
        ok, reason, missing_perms, missing_tools = self._validator.validate(skill)
        record = SkillRecord(
            skill=skill,
            manifest=manifest if manifest is not None else SkillManifest(id="<invalid>"),
            available=ok,
            reason=reason,
            missing_permissions=missing_perms,
            missing_tools=missing_tools,
        )
        key = record.manifest.id
        if key in self._records:
            raise ValueError(f"duplicate skill id {key!r}")
        self._records[key] = record
        if ok:
            log.info("registered skill %s v%s", key, record.manifest.version)
        else:
            log.warning("skill %s registered as unavailable: %s", key, reason)
        return record

    # -- lookup -----------------------------------------------------------

    def get(self, skill_id: str) -> SkillRecord | None:
        return self._records.get(skill_id)

    def available(self) -> list[SkillRecord]:
        return [r for r in self._records.values() if r.available]

    def all(self) -> list[SkillRecord]:
        return list(self._records.values())

    def ids(self) -> list[str]:
        return sorted(self._records)

    def capability_names(self) -> list[str]:
        """What the companion can honestly claim to do right now."""
        return sorted(r.manifest.name or r.manifest.id for r in self.available())

    def describe(self) -> list[dict]:
        return [r.to_dict() for r in self._records.values()]

    def record_invocation(self, skill_id: str, success: bool) -> None:
        record = self._records.get(skill_id)
        if record is None:
            return
        record.stats["invocations"] += 1
        if not success:
            record.stats["failures"] += 1

    def snapshot(self) -> dict:
        return {
            "total": len(self._records),
            "available": len(self.available()),
            "skills": self.describe(),
            "permissions": self._permissions.snapshot(),
        }


class SkillLoader:
    """Discovers skills from a package and registers them."""

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def load_package(self, package: str = "companion.skills.builtin") -> list[SkillRecord]:
        records: list[SkillRecord] = []
        try:
            module = importlib.import_module(package)
        except ImportError as exc:
            log.warning("skill package %s not importable: %s", package, exc)
            return records
        for info in pkgutil.iter_modules(module.__path__):
            if info.name.startswith("_"):
                continue
            full = f"{package}.{info.name}"
            try:
                submodule = importlib.import_module(full)
            except Exception as exc:
                log.warning("could not import skill module %s: %s", full, exc)
                continue
            for skill in getattr(submodule, "SKILLS", []):
                try:
                    records.append(self._registry.register(
                        skill() if isinstance(skill, type) else skill))
                except Exception as exc:
                    log.warning("could not register skill from %s: %s", full, exc)
        return records
