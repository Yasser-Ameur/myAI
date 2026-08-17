"""Skill permissions: default deny.

A skill gets nothing it did not declare in its manifest, and nothing the user
has not allowed. Permissions are checked at two points: statically at load
time (is this skill allowed to ask for `shell` at all?) and dynamically at the
moment of use (does *this* call have `memory.write`?).

Grants live in configuration, so a user can inspect and revoke them without
editing code.
"""

from __future__ import annotations

import logging
from enum import Enum

log = logging.getLogger(__name__)


class Permission(str, Enum):
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    NETWORK = "network"
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_WRITE = "filesystem.write"
    MICROPHONE = "microphone"
    CAMERA = "camera"
    NOTIFICATIONS = "notifications"
    CALENDAR = "calendar"
    PROCESS = "process"
    SHELL = "shell"
    RUNTIME_INSPECT = "runtime.inspect"

    @classmethod
    def parse(cls, raw: str) -> "Permission | None":
        try:
            return cls(str(raw).strip().lower())
        except ValueError:
            return None


# Permissions that can never be granted implicitly: they need the user to say
# yes in configuration, and side-effecting tools additionally confirm per call.
DANGEROUS = frozenset({
    Permission.SHELL,
    Permission.PROCESS,
    Permission.FILESYSTEM_WRITE,
    Permission.NETWORK,
})

# Granted to every skill unless the user revokes them: reading the companion's
# own memory is the baseline expectation of a memory-centric system.
DEFAULT_GRANTS = frozenset({Permission.MEMORY_READ})


class PermissionDenied(Exception):
    def __init__(self, skill_id: str, permission: Permission | str) -> None:
        self.skill_id = skill_id
        self.permission = permission
        super().__init__(f"skill {skill_id!r} is not permitted to use {permission}")


class PermissionManager:
    """Resolves what each skill may do, from config + manifest declarations."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._default_grants = set(DEFAULT_GRANTS)
        for raw in cfg.get("default_grants", []) or []:
            perm = Permission.parse(raw)
            if perm is not None:
                self._default_grants.add(perm)
        self._per_skill: dict[str, set[Permission]] = {}
        for skill_id, perms in (cfg.get("grants") or {}).items():
            resolved = {p for p in (Permission.parse(x) for x in perms or []) if p}
            self._per_skill[str(skill_id)] = resolved
        self._denied: dict[str, set[Permission]] = {}
        for skill_id, perms in (cfg.get("deny") or {}).items():
            resolved = {p for p in (Permission.parse(x) for x in perms or []) if p}
            self._denied[str(skill_id)] = resolved
        self.violations: list[dict] = []

    def granted(self, skill_id: str) -> set[Permission]:
        allowed = set(self._default_grants) | self._per_skill.get(skill_id, set())
        return allowed - self._denied.get(skill_id, set())

    def allows(self, skill_id: str, permission: Permission | str) -> bool:
        perm = permission if isinstance(permission, Permission) else Permission.parse(permission)
        if perm is None:
            return False
        return perm in self.granted(skill_id)

    def check(self, skill_id: str, permission: Permission | str) -> None:
        if not self.allows(skill_id, permission):
            self.violations.append({"skill": skill_id, "permission": str(permission)})
            raise PermissionDenied(skill_id, permission)

    def missing_for(self, skill_id: str, required: list) -> list[Permission]:
        """Which of a manifest's required permissions are not granted."""
        granted = self.granted(skill_id)
        out = []
        for raw in required or []:
            perm = raw if isinstance(raw, Permission) else Permission.parse(raw)
            if perm is not None and perm not in granted:
                out.append(perm)
        return out

    def snapshot(self) -> dict:
        return {
            "default_grants": sorted(p.value for p in self._default_grants),
            "per_skill": {k: sorted(p.value for p in v) for k, v in self._per_skill.items()},
            "denied": {k: sorted(p.value for p in v) for k, v in self._denied.items()},
            "violations": list(self.violations),
        }
