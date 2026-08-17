# Skills

A skill is a bounded, declared, permissioned capability. It is not a prompt
template. It has a manifest, typed input and output, its own tests, and — when
it needs one — a scoped slice of memory.

Skills exist because some questions have exact answers. Routing "what is your
name?" or "1234 × 5678" through a 1.7B model and hoping the context survives
truncation is how a companion ends up forgetting its own name and doing
arithmetic wrong. Everything that *is* open-ended still goes to the model.

## The loop

```
user turn
  → turn commit (durable facts)
  → skill router polls every available skill's can_handle()
      → a skill claims it (confidence ≥ route_threshold) → execute() → reply
      → nobody claims it → retrieval → context → LLM → reply
```

Routing is a poll, not a classification prompt: deterministic, testable, and
free of model latency. A skill can never be invoked for a turn it did not
claim, and a skill that fails degrades to normal conversation rather than
breaking the turn.

## Writing a skill

Three things: a manifest, an implementation, tests. No core changes.

```python
from companion.skills.base import (
    BaseSkill, SkillContext, SkillDecision, SkillInput, SkillManifest, SkillResult,
)

MANIFEST = SkillManifest(
    id="weather",
    name="weather",
    version="1.0.0",
    description="Current conditions for a place.",
    required_permissions=["memory.read"],
    required_tools=["weather_api"],
    memory_read_scopes=["lives_in"],
    examples=["What's the weather?"],
)


class WeatherSkill(BaseSkill):
    manifest = MANIFEST

    async def can_handle(self, context: SkillContext, input: SkillInput) -> SkillDecision:
        if "weather" not in input.text.lower():
            return SkillDecision.no("not about weather")
        return SkillDecision.yes(0.9, "asks about weather", place="here")

    async def execute(self, context: SkillContext, input: SkillInput) -> SkillResult:
        result = await context.tools.invoke("weather_api", caller=self.manifest.id,
                                            place=input.args["place"])
        if not result.ok:
            return SkillResult.failure(result.error)
        return SkillResult(text=f"It's {result.value['summary']}.", data=result.value)


SKILLS = [WeatherSkill]      # the loader discovers this
```

Drop the module in `src/companion/skills/builtin/`, grant its permissions in
`config/companion.yaml`, and it is live. `SkillLoader.load_package()` also
accepts any other package name, so out-of-tree skill packages work the same way.

### Rules that matter

- **`can_handle` must be cheap and deterministic.** It runs on every turn for
  every skill, under a 2 s timeout. No model calls.
- **Be honest about confidence.** `route_threshold` (default 0.6) is what keeps
  an over-eager skill from hijacking conversation.
- **Return `SkillResult.failure(...)` rather than raising** for expected
  problems. Exceptions are caught, logged and counted, but a failure with a
  reason is more useful.
- **Never touch the graph directly.** Use `context.memory`; that is where
  permissions are enforced.

## Permissions

Default deny. A skill gets only what its manifest declares *and* the user has
granted in `config/companion.yaml`:

```yaml
skills:
  route_threshold: 0.6
  default_grants: [memory.read]
  grants:
    goals: [memory.read, memory.write]
    diagnostics: [runtime.inspect]
  deny:
    recall: [memory.read]     # switch a skill off without uninstalling it
```

Available: `memory.read`, `memory.write`, `network`, `filesystem.read`,
`filesystem.write`, `microphone`, `camera`, `notifications`, `calendar`,
`process`, `shell`, `runtime.inspect`.

A skill whose permissions are not granted is registered as **unavailable with
the reason** rather than dropped, so the companion can explain what it cannot
do and why:

```
$ companion skills list
[OFF] goals          v1.0.0  Tracks your goals and what we planned to do next.
        reason: permissions not granted: memory.write
```

Skill writes are namespaced: `context.memory.remember("colour", "green")` from
skill `notes` writes the predicate `skill:notes:colour`. A skill cannot
overwrite core user facts through the facade.

## API versioning

Manifests declare `api_version` (currently `1.0`). Same major and an equal or
older minor is compatible; anything else is refused at registration with a
clear reason. Bump the minor when adding optional capability, the major when
breaking the protocol.

## Built-in skills

| id | what it does | permissions |
|----|--------------|-------------|
| `identity` | who I am, what you named me, what I was called before | `memory.read` |
| `recall` | current and historical values of stored facts | `memory.read` |
| `provenance` | why I believe something, with evidence and dates | `memory.read` |
| `goals` | list, add and complete goals | `memory.read`, `memory.write` |
| `capabilities` | what I can do, from the live registry | — |
| `diagnostics` | my own health, and why a subsystem is unavailable | `runtime.inspect` |
| `calculator` | exact arithmetic via the calculator tool | — |
| `datetime` | current date and time | — |

`identity`, `recall` and `provenance` are the reference examples for
graph-backed skills; `calculator` and `datetime` for tool-backed ones;
`diagnostics` for runtime introspection.

## Inspecting

```powershell
companion skills list
companion skills describe recall
companion skills permissions
companion skills tools
```
