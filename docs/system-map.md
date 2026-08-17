# System map

What each subsystem is, what state it owns, how it fails, and — separately —
what was actually verified to work by running it. Status labels are used
strictly:

- **REAL** — verified end-to-end with real weights on the target machine
- **REAL (untested here)** — real implementation, not exercised in this pass
- **FALLBACK** — deterministic stand-in used when a provider is missing
- **DEGRADED** — runs, but does not do what its name implies

## Dependency graph

```
CLI / HTTP API
     ↓
CompanionApp  (runtime/orchestration.py)  — the only place that builds infrastructure
     ├── EventBus            in-process pub/sub, async task ownership
     ├── Scheduler           interactive vs background arbitration
     ├── MemoryGuard         RAM pressure policy
     ├── ModelLifecycle      lazy load / idle unload, RAM budget
     ├── Reflection          idle consolidation (owns MemoryPipeline)
     └── Services
           ├── SelfModel ──────┐ identity, reconstructed from graph at boot
           ├── FactWriter ─────┤ the ONLY durable-write path
           ├── TurnCommitter ──┤ per-turn salient facts
           ├── Conversation ───┤ context assembly, planning, generation
           │     └── SkillRouter → SkillRegistry → Skills → ToolInvoker → Tools
           ├── Memory          episodes, turns, lifecycle commands
           ├── Retrieval       hybrid lexical/semantic/graph/temporal
           ├── Personality     probabilistic traits with evidence
           ├── Relationship    per-person trust/familiarity
           ├── Perception      VAD/STT/face → UserState
           ├── Speech          TTS + playback + barge-in
           └── Avatar          canonical expression state → driver
                          ↓
                    CognitiveGraph (SQLite) — source of truth
```

Everything below `Services` depends on `application/ports.py` protocols only.
No application module imports a concrete provider.

## Subsystems

| Subsystem | Owns | Persistent state | Failure mode | Status |
|---|---|---|---|---|
| `SqliteStorage` | connection, WAL, migrations | the whole DB | raises `MemoryUnavailableError`; app degrades to `NullGraphStore` and *says so* | **REAL** |
| `CognitiveGraph` | entities, facts, episodes, memories, goals, beliefs, observations | all | same | **REAL** |
| `FactWriter` | supersession, authority, confirmation, evidence | facts, observations | refuses low-authority writes, returns outcome | **REAL** |
| `SelfModelService` | agent identity, name history | `self:name`, `self:persona` | falls back to config name on storage error | **REAL** |
| `TurnCommitter` | per-turn durable extraction | facts, goals, memories | logged and skipped; turn still answers | **REAL** |
| `MemoryPipeline` | consolidation, dedup, grounding filter | memories, facts, evidence | LLM failure → rule extractor | **REAL** |
| `Reflection` | idle consolidation, decay, contradiction resolution | episodes, memory status | bounded, interruptible, yields to interaction | **REAL** |
| `HybridRetriever` | ranking | none | embedding failure → lexical + graph only | **REAL** |
| `PersonalityEngine` | traits/values/preferences | personality tables | in-memory profile if graph is null | **REAL** |
| `SkillRegistry` / `Router` | skill lifecycle, routing | none | failing skill → falls through to LLM | **REAL** |
| `ToolInvoker` | validation, permissions, timeouts | none | returns `ToolResult(ok=False)`, never hangs | **REAL** |
| `ModelRegistry` / `Lifecycle` | provider construction, RAM budget | none | per-slot `load_error`, surfaced by `doctor` | **REAL** |
| LLM (`llama_cpp`, Qwen3) | generation | none | fallback provider marked `fallback=True` | **REAL** |
| Embeddings (ONNX, bge-small) | vectors | `embeddings` table | retrieval degrades to lexical | **REAL** |
| STT (faster-whisper) | transcripts | none | perception emits nothing | **REAL (untested here)** |
| TTS (Kokoro) | audio | none | speech silently skipped, reported by `diagnostics` | **REAL (untested here)** |
| VAD (Silero) | speech segmentation | none | no barge-in | **REAL (untested here)** |
| Vision (MediaPipe) | face landmarks | none | user state loses visual channel | **REAL (untested here)** |
| `EventBus` | subscriptions, queues | none | handler exception isolated | **REAL** |
| Avatar | canonical expression | none | console driver no-ops | **REAL (untested here)** |

The four "untested here" rows need a microphone, speaker and camera. Their
weights are installed and their providers construct without error, but this
pass verified them only at that level — they are not claimed as end-to-end
verified.

## Traces verified in this pass

**Text turn** (real Qwen3-1.7B, real SQLite):

```
input → append_turn → turn commit (facts durable)
      → skill router → [claimed? → skill reply]
      → retrieval (7 ms) → context assembly → plan (rules, ~0 ms)
      → LLM generate (~4.2 s) → append_turn → episode
```

**Restart** (two separate OS processes):

```
process 1: "You are Jarvis." "My favorite color is purple."  → exit
process 2: boot → SelfModelService.load() → identity=Jarvis (source=persisted)
           "What's your name?"        → "My name is Jarvis — you named me that."
           "What's my favorite color?" → "Your favorite color is purple."
```

**Correction with history**:

```
"I don't like purple anymore. It's blue now."   (a later process)
  → referent resolved from the graph by value → favorite:color
  → purple closed (valid_to set), blue asserted
"What's my favorite color?"        → blue
"What used to be my favorite color?" → purple
```

**Degradation**: with the DB pointed at an unwritable path, the app starts,
reports `DEGRADED`, answers "Why can't you remember things right now?" with the
actual reason, and keeps skills working. `doctor` exits 1.

## Known hidden coupling

- `ConversationService` reaches into `MemoryService` for the current episode
  and into `PersonalityEngine` for communication learning. Both are injected
  interfaces, but the ordering (append turn → commit → route → retrieve) is
  implicit in `respond()` rather than expressed as a pipeline.
- `LLMResponsePlanner` reads `router._registry`, a private attribute.
- `InteractSession` builds a *second* `AvatarService` alongside the one
  `CompanionApp` owns.
- `_idle_unload_once` matches slots by string prefix (`"llm."`, `"stt."`).

These are recorded rather than fixed: each is contained, and none of them
caused an observed defect.
