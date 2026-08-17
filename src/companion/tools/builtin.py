"""Built-in tools.

Deliberately few, and deliberately dull. Each one is pure or read-only; there
is no shell tool and no arbitrary-code tool, because nothing in the current
skill set needs one and the LLM must not be one prompt away from executing
commands.
"""

from __future__ import annotations

import ast
import math
import operator
import os
import platform
from datetime import datetime, timezone

from companion.tools.base import ToolManifest, ToolRisk

# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "min": min, "max": max,
    "pow": math.pow,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# Guards against a one-character denial of service such as 9**9**9.
MAX_EXPONENT = 128
MAX_RESULT_DIGITS = 400


class UnsafeExpression(ValueError):
    pass


def evaluate_expression(expression: str) -> float | int:
    """Evaluate arithmetic from an AST whitelist.

    Never uses eval(): the input reaches this function straight from user
    speech, so the only safe design is one where unsupported syntax cannot be
    expressed at all.
    """
    expression = (expression or "").strip()
    if not expression:
        raise UnsafeExpression("empty expression")
    if len(expression) > 500:
        raise UnsafeExpression("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"cannot parse: {exc.msg}") from exc
    value = _eval_node(tree.body)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise UnsafeExpression("result is not a finite number")
    if isinstance(value, int) and len(str(abs(value))) > MAX_RESULT_DIGITS:
        raise UnsafeExpression("result is too large")
    return value


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise UnsafeExpression("only numeric literals are allowed")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"operator {type(node.op).__name__} is not allowed")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(_as_number(right)) > MAX_EXPONENT:
            raise UnsafeExpression("exponent too large")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and _as_number(right) == 0:
            raise UnsafeExpression("division by zero")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression("unary operator not allowed")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise UnsafeExpression("only whitelisted math functions are allowed")
        if node.keywords:
            raise UnsafeExpression("keyword arguments are not allowed")
        return _FUNCTIONS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise UnsafeExpression(f"unknown name {node.id!r}")
    raise UnsafeExpression(f"{type(node).__name__} is not allowed")


def _as_number(value):
    if isinstance(value, (int, float)):
        return value
    raise UnsafeExpression("expected a number")


class CalculatorTool:
    manifest = ToolManifest(
        id="calculator",
        description="Evaluate an arithmetic expression exactly.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        returns="number",
        side_effects=False,
        risk=ToolRisk.LOW,
        timeout_s=2.0,
    )

    async def run(self, expression: str):
        return evaluate_expression(expression)


# ---------------------------------------------------------------------------
# clock
# ---------------------------------------------------------------------------

class ClockTool:
    manifest = ToolManifest(
        id="clock",
        description="Current local date and time.",
        parameters={"type": "object", "properties": {}},
        returns="object",
        side_effects=False,
        risk=ToolRisk.LOW,
        timeout_s=1.0,
    )

    async def run(self):
        now = datetime.now()
        return {
            "iso": now.isoformat(timespec="seconds"),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "weekday": now.strftime("%A"),
        }


# ---------------------------------------------------------------------------
# system probe (read-only)
# ---------------------------------------------------------------------------

class SystemProbeTool:
    manifest = ToolManifest(
        id="system_probe",
        description="Read-only snapshot of local CPU/RAM/platform.",
        parameters={"type": "object", "properties": {}},
        returns="object",
        side_effects=False,
        permissions=["runtime.inspect"],
        risk=ToolRisk.MEDIUM,
        timeout_s=3.0,
    )

    async def run(self):
        info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }
        try:
            import psutil  # optional

            vm = psutil.virtual_memory()
            info["ram_total_mb"] = vm.total // (1024 * 1024)
            info["ram_available_mb"] = vm.available // (1024 * 1024)
            info["process_rss_mb"] = psutil.Process().memory_info().rss // (1024 * 1024)
        except ImportError:
            info["ram_total_mb"] = None
            info["note"] = "psutil not installed; live RAM figures unavailable"
        return info


def default_tools() -> list:
    return [CalculatorTool(), ClockTool(), SystemProbeTool()]
