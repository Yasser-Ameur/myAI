# Architecture

## Layers

| Layer             | Responsibility                                                        | Depends on            |
|-------------------|-----------------------------------------------------------------------|-----------------------|
| `core/`           | Protocols/contracts, events, ids, clock, errors, types                | nothing               |
| `domain/`         | Pure data + behavior: `Fact`, `Memory`, `Episode`, `ResponsePlan`, `PersonalityProfile`, `Relationship`, `UserState`, `AgentState` | `core`                |
| `application/`    | Orchestration of domain objects: memory pipeline, retrieval, extraction, personality engine, conversation, reflection, avatar, ports | `core`, `domain`      |
| `infrastructure/` | Concrete adapters: SQLite storage + graph, vector index, model providers, registry | `core`, `domain`      |
| `interfaces/`     | CLI and HTTP API                                                      | everything above      |
| `runtime/`        | Wiring and lifecycle: config, hardware profiling, memory guard, scheduler, metrics, orchestration, benchmarks, model installer | all layers            |

The only dependency edges between application logic and model weights are the
protocols in `core/contracts.py`. `application/ports.py` defines what the
storage and graph layers must provide; `infrastructure/sqlite_graph.py`
implements them. Nothing in application code imports a concrete provider.

## Startup / lifecycle

`runtime/orchestration.py` builds the whole system from `config/companion.yaml`:

```
Config.load()
  -> build_hardware_profile()          # RAM / CPU / GPU detection, budget
  -> ModelRegistry.build_from_config() # one provider per slot, fallbacks on failure
  -> CompanionComponents               # graph, memory, retrieval, personality, ...
  -> CompanionApp.build()              # wires EventBus + background loops
      |  reflection loop   (interactive -> consolidation)
      |  memory guard loop (RAM pressure -> unload slots)
CompanionApp.respond(text)             # conversation round-trip
CompanionApp.shutdown()                # flush, stop loops, close storage
```

## Turn flow

1. `conversation.respond(text)` — assemble context (state + profile + retrieved
   memories + episodes + goals + relationships, inside a token budget).
2. `HybridRetriever.retrieve(query, mode)` — lexical + semantic + graph +
   recency + importance + confidence, deduped and reranked.
3. `LLMResponsePlanner` — plans `intent`, `tone`, `expressive_parameters`
   (structured JSON; falls back to rules when no LLM). `ResponsePlan.from_dict`
   is tolerant of malformed enum values.
4. LLM generates the reply text.
5. Speech/avatar services publish `EVENT_SPEECH_CHUNK_READY` /
   `EVENT_EXPRESSION_CHANGED` when real TTS / vision adapters exist.
6. `MemoryService.append_turn()` records the exchange; `close_episode()` runs
   the extraction pipeline (see `docs/memory.md`).

## Event bus

`core/events.py` provides a synchronous in-process `EventBus`:
`subscribe(kind, handler, policy, queue_size)`, `publish`, `publish_async`,
`unsubscribe`. Handlers run on the caller's thread or the loop via
`publish_async`. Canonical event constants:

- `EVENT_MEMORY_COMMITTED`
- `EVENT_SPEECH_CHUNK_READY`
- `EVENT_RESPONSE_PLAN_CREATED`
- `EVENT_USER_STATE_UPDATED`
- `EVENT_EXPRESSION_CHANGED`

`DropPolicy` controls backpressure when a subscriber is slow.

## Runtime components

| Component            | File                          | Role                                                        |
|----------------------|-------------------------------|-------------------------------------------------------------|
| `CompanionApp`       | `orchestration.py`            | composition root, lifecycle, background loops               |
| `HardwareProfile`    | `hardware.py`                 | auto/balanced/performance/ultra_low profiles + RAM budget   |
| `Scheduler`          | `scheduler.py`                | `WorkloadClass` (REALTIME/INTERACTIVE/BACKGROUND), background gating |
| `MemoryGuard`        | `memory_guard.py`             | normal/elevated/critical pressure; unloads stt/tts/embeddings/llm |
| `Metrics`            | `metrics.py`                  | counters, gauges, latency series                            |
| `benchmark_*`        | `benchmarks.py`               | slot + end-to-end benchmarks; honest SIMULATED labeling     |
| `ModelInstaller`     | `model_installer.py`          | manifest-driven download + sha256 verify + cache            |

## Concurrency

`MemoryGuard`, `Scheduler` and background loops run on the asyncio event loop;
SQLite access is serialized through `SqliteStorage`'s lock and supports nested
transactions via savepoints, so an inner `save_profile()` inside an outer
episode transaction cannot deadlock or clobber the outer write.
