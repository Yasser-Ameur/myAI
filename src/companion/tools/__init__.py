"""Tool subsystem: typed, permissioned, time-bounded side effects.

Skills act on the world through tools. The LLM never executes anything
directly — it can at most cause a skill to request a tool, and every tool call
is validated against a manifest, checked against permissions, run with a
timeout, and recorded.
"""

from companion.tools.base import (
    Tool,
    ToolManifest,
    ToolResult,
    ToolRisk,
)
from companion.tools.registry import ToolInvoker, ToolRegistry

__all__ = [
    "Tool",
    "ToolInvoker",
    "ToolManifest",
    "ToolRegistry",
    "ToolResult",
    "ToolRisk",
]
