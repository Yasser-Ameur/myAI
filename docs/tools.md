# Tools

Skills act on the world through tools. The LLM never executes anything: it can
at most produce text that causes a *skill* to request a tool, and every request
is validated against a manifest, checked against the caller's permissions, run
under a timeout, size-capped and recorded.

There is deliberately no shell tool and no eval tool.

## Contract

```python
from companion.tools.base import ToolManifest, ToolRisk

class WeatherTool:
    manifest = ToolManifest(
        id="weather_api",
        description="Current conditions for a place.",
        parameters={"type": "object",
                    "properties": {"place": {"type": "string"}},
                    "required": ["place"]},
        returns="object",
        side_effects=False,
        permissions=["network"],
        risk=ToolRisk.MEDIUM,
        timeout_s=5.0,
        max_output_chars=8000,
    )

    async def run(self, place: str):
        ...
```

Register it in `default_tools()` (or your own registry) and declare it in the
skill's `required_tools`. A skill whose tools are missing is registered as
unavailable with that reason.

## Risk and confirmation

| risk | meaning | confirmation |
|------|---------|--------------|
| `LOW` | pure computation, no side effects | no |
| `MEDIUM` | reads local state the user may consider private | no, but needs the permission |
| `HIGH` | writes, deletes, or leaves the machine | **yes, every call** |

Any tool with `side_effects=True` or `risk=HIGH` will not run unless a
confirmation handler approves that specific call. With no handler configured
the answer is no — the safe default for an autonomous loop:

```
ToolResult(ok=False, error="danger needs explicit user confirmation and it was not given")
```

## Validation

Arguments are checked against the manifest before execution: missing required
arguments, unknown arguments and wrong types are all refused with a message
naming the tool and argument. Coercion is limited to unambiguous scalar cases
(`"5"` → `5` for an integer parameter).

## Built-in tools

| id | risk | side effects | permissions | notes |
|----|------|--------------|-------------|-------|
| `calculator` | LOW | no | — | AST whitelist, never `eval` |
| `clock` | LOW | no | — | local date/time |
| `system_probe` | MEDIUM | no | `runtime.inspect` | CPU/RAM/platform, read-only |

### The calculator is an AST evaluator, not `eval`

Input reaches it straight from user speech, so the only safe design is one
where unsupported syntax cannot be expressed. It walks a whitelist of nodes
(numeric literals, arithmetic operators, a fixed set of math functions, `pi`/
`e`/`tau`) and rejects everything else, plus explicit guards for
division by zero, oversized exponents (`9**9**9`) and oversized results.

```python
evaluate_expression("1234 * 5678")   # 7006652
evaluate_expression("__import__('os')")  # UnsafeExpression
```

## Observability

`ToolInvoker` records every call — tool, caller, ok, elapsed, error — available
through `invoker.stats()`. Timeouts surface as failures with the bound named,
never as a hang.
