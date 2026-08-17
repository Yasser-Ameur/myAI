"""Tool contracts.

Every tool declares its side effects, the permissions it needs, its risk
class, a timeout and resource limits. Anything that changes state outside the
companion is `side_effects=True` and, above `ToolRisk.LOW`, requires explicit
user confirmation before it runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ToolRisk(str, Enum):
    LOW = "low"          # pure computation, no side effects
    MEDIUM = "medium"    # reads local state the user might consider private
    HIGH = "high"        # writes, deletes, or leaves the machine

    @property
    def needs_confirmation(self) -> bool:
        return self is ToolRisk.HIGH


@dataclass
class ToolManifest:
    id: str
    description: str = ""
    version: str = "1.0.0"
    parameters: dict = field(default_factory=dict)   # JSON-schema-ish
    returns: str = ""
    side_effects: bool = False
    permissions: list[str] = field(default_factory=list)
    risk: ToolRisk = ToolRisk.LOW
    timeout_s: float = 5.0
    max_output_chars: int = 8000

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "version": self.version,
            "parameters": dict(self.parameters),
            "returns": self.returns,
            "side_effects": self.side_effects,
            "permissions": list(self.permissions),
            "risk": self.risk.value,
            "timeout_s": self.timeout_s,
        }


@dataclass
class ToolResult:
    ok: bool = True
    value: Any = None
    error: str = ""
    elapsed_ms: float = 0.0
    tool_id: str = ""
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "value": self.value,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "tool_id": self.tool_id,
            "truncated": self.truncated,
        }

    @classmethod
    def failure(cls, tool_id: str, error: str) -> "ToolResult":
        return cls(ok=False, error=error, tool_id=tool_id)


@runtime_checkable
class Tool(Protocol):
    manifest: ToolManifest

    async def run(self, **kwargs) -> Any:
        ...


def validate_arguments(manifest: ToolManifest, args: dict) -> dict:
    """Check arguments against the manifest, coercing simple scalar types.

    Deliberately small: enough to reject a malformed call before it executes,
    without pulling in a schema library for a local-first project.
    """
    schema = manifest.parameters or {}
    props = schema.get("properties", schema)
    required = schema.get("required", [])
    out: dict = {}
    for key in required:
        if key not in args:
            raise ValueError(f"missing required argument {key!r} for tool {manifest.id}")
    for key, value in (args or {}).items():
        spec = props.get(key)
        if spec is None:
            raise ValueError(f"unknown argument {key!r} for tool {manifest.id}")
        expected = spec.get("type") if isinstance(spec, dict) else None
        out[key] = _coerce(value, expected, key, manifest.id)
    return out


def _coerce(value: Any, expected: str | None, key: str, tool_id: str) -> Any:
    if expected in (None, "any"):
        return value
    try:
        if expected == "string":
            return str(value)
        if expected == "number":
            return float(value)
        if expected == "integer":
            return int(value)
        if expected == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("1", "true", "yes", "on")
        if expected == "array":
            if isinstance(value, (list, tuple)):
                return list(value)
            raise ValueError
        if expected == "object":
            if isinstance(value, dict):
                return value
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"argument {key!r} for tool {tool_id} is not a valid {expected}"
        ) from None
    return value
