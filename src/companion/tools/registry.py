"""Tool registry and invoker."""

from __future__ import annotations

import asyncio
import logging
import time

from companion.skills.permissions import PermissionManager
from companion.tools.base import Tool, ToolManifest, ToolResult, validate_arguments

log = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        manifest: ToolManifest = tool.manifest
        if not manifest.id:
            raise ValueError("tool manifest must declare an id")
        if manifest.id in self._tools:
            raise ValueError(f"duplicate tool id {manifest.id!r}")
        self._tools[manifest.id] = tool
        log.debug("registered tool %s (risk=%s)", manifest.id, manifest.risk.value)

    def get(self, tool_id: str) -> Tool | None:
        return self._tools.get(tool_id)

    def ids(self) -> list[str]:
        return sorted(self._tools)

    def manifests(self) -> list[dict]:
        return [t.manifest.to_dict() for t in self._tools.values()]


class ToolInvoker:
    """Validates, permissions, times and records every tool call."""

    def __init__(self, registry: ToolRegistry, permissions: PermissionManager | None = None,
                 confirm=None) -> None:
        self._registry = registry
        self._permissions = permissions or PermissionManager()
        # confirm(tool_id, args) -> bool. Absent means "never auto-approve a
        # high-risk tool", which is the safe default for an autonomous loop.
        self._confirm = confirm
        self.calls: list[dict] = []

    async def invoke(self, tool_id: str, caller: str = "", **args) -> ToolResult:
        tool = self._registry.get(tool_id)
        if tool is None:
            return ToolResult.failure(tool_id, f"unknown tool {tool_id!r}")
        manifest = tool.manifest

        for permission in manifest.permissions:
            if caller and not self._permissions.allows(caller, permission):
                return ToolResult.failure(
                    tool_id, f"{caller} lacks permission {permission} required by {tool_id}"
                )

        try:
            validated = validate_arguments(manifest, args)
        except ValueError as exc:
            return ToolResult.failure(tool_id, str(exc))

        if manifest.risk.needs_confirmation or manifest.side_effects:
            approved = False
            if self._confirm is not None:
                try:
                    approved = bool(self._confirm(tool_id, validated))
                except Exception as exc:
                    log.warning("confirmation handler failed for %s: %s", tool_id, exc)
            if not approved:
                return ToolResult.failure(
                    tool_id, f"{tool_id} needs explicit user confirmation and it was not given"
                )

        started = time.monotonic()
        try:
            value = await asyncio.wait_for(tool.run(**validated), timeout=manifest.timeout_s)
            ok, error = True, ""
        except asyncio.TimeoutError:
            value, ok, error = None, False, f"{tool_id} timed out after {manifest.timeout_s}s"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            value, ok, error = None, False, f"{type(exc).__name__}: {exc}"
            log.warning("tool %s failed: %s", tool_id, exc)
        elapsed = (time.monotonic() - started) * 1000.0

        truncated = False
        if isinstance(value, str) and len(value) > manifest.max_output_chars:
            value = value[: manifest.max_output_chars]
            truncated = True

        result = ToolResult(ok=ok, value=value, error=error, elapsed_ms=elapsed,
                            tool_id=tool_id, truncated=truncated)
        self.calls.append({"tool": tool_id, "caller": caller, "ok": ok,
                           "elapsed_ms": round(elapsed, 2), "error": error})
        return result

    def stats(self) -> dict:
        total = len(self.calls)
        failed = sum(1 for c in self.calls if not c["ok"])
        return {"calls": total, "failed": failed,
                "tools": self._registry.ids()}
