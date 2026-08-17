# Human Companion

A local-first, offline, modular cognitive AI companion in Python.

It remembers what you tell it (temporal knowledge graph + hybrid retrieval),
builds a probabilistic model of your personality and its own, adapts its
communication style, and can speak and express itself through a face avatar —
all on CPU, all configurable, with no cloud dependency.

## Highlights

- **Clean architecture** — `core` / `domain` / `application` / `infrastructure` /
  `interfaces` / `runtime`. Application code never imports concrete providers.
- **Persistent identity** — name it once and it stays named, across restarts.
  Renaming keeps the old name queryable; a hedged guess cannot overwrite a
  stated identity.
- **Memory graph** — SQLite-backed episodic memory, semantic facts, goals,
  relationships and knowledge chunks with temporal validity. History is never
  silently deleted; updates invalidate rather than overwrite. Salient facts
  commit per turn, so an ungraceful exit loses nothing.
- **Skills and tools** — bounded, declared capabilities with default-deny
  permissions, typed tool manifests, timeouts and confirmation gating for
  side effects. "What can you do?" is answered from the live registry.
- **Hybrid retrieval** — lexical (informative-token matching) + semantic
  (embeddings) + graph + recency + importance + confidence, with reranking.
- **Probabilistic personality** — evidence-based `PersonalityProfile` with
  traits / values / preferences, contradiction tracking, and behavioral
  communication-preference learning.
- **Model adapters** — LLM, STT, TTS, VAD, vision, embeddings selected by
  config only. Every optional dependency degrades to a deterministic fallback
  with a clear warning; the system runs with zero model weights installed.
- **Runtime services** — hardware profiling, memory guard, scheduler, metrics,
  benchmark, model installer, HTTP API, interactive CLI.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -e ".[dev]"
```

Runs with no models at all (fallback mocks, fully deterministic):

```powershell
companion run
```

Add real models per modality (see `docs/models.md` for offline install):

```powershell
pip install "human-companion[llm]"      # llama-cpp-python
pip install "human-companion[stt]"      # faster-whisper
pip install "human-companion[tts]"      # kokoro-onnx + soundfile + piper-tts
pip install "human-companion[embeddings]"
pip install "human-companion[vad]"      # onnxruntime
pip install "human-companion[vision]"   # mediapipe

companion models list
companion models install qwen3-1.7b whisper-base kokoro-82m bge-small-en silero-vad face-landmarker
```

## Quick start

```powershell
companion doctor                 # environment + hardware + slots
companion run                    # interactive REPL (stdin, no models needed)
companion api                    # HTTP API on http://127.0.0.1:8377
companion benchmark              # honest slot benchmarks (SIMULATED label when mocked)
companion runtime                # runtime health + profile + metrics
companion memory list            # inspect stored memories
companion personality inspect    # inspect the probabilistic personality
companion graph facts            # inspect the knowledge graph
companion identity show          # who the companion currently is, and why
companion identity history       # every name it has had
companion why <fact-id>          # evidence behind a stored belief
companion skills list            # skills, with reasons for any unavailable
companion data export ./backup   # portable cognitive state (never weights)
companion models install qwen3-1.7b   # download and verify a model
```

## Does it actually remember?

```powershell
companion run
you> You are Jarvis.
you> My favorite color is purple.
you> quit

companion run
you> What's your name?
companion> My name is Jarvis — you named me that.
you> What's my favorite color?
companion> Your favorite color is purple.
```

Both answers come from the graph, not from chat history — the identity is
reconstructed before any model call. See `docs/final-report.md` for measured
numbers and `docs/system-map.md` for what is verified versus assumed.

## HTTP API

| Method | Path        | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/health`   | liveness + degraded-mode status      |
| GET    | `/metrics`  | runtime metrics (events, latencies)  |
| GET    | `/runtime`  | hardware profile, slots, personality |
| POST   | `/chat`     | `{"text": "...", "source": "text"}`  |

## Layout

```
src/companion/
  core/            contracts, events, ids, types, clock, errors
  domain/          graph, memory, conversation, personality, relationship, state, agent
  application/     perception, extraction, memory, retrieval, personality, conversation,
                   reflection, avatar, speech_output, ports,
                   facts (durable writes), identity (self-model), salience (turn commit)
  skills/          base, permissions, registry, router, builtin/
  tools/           base, registry, builtin
  infrastructure/  storage, sqlite_graph, vector, models/ (adapters + registry)
  interfaces/      cli, api
  runtime/         config, hardware, scheduler, memory_guard, metrics, orchestration,
                   benchmarks, model_installer, portability
config/companion.yaml    all provider/model/tuning selection
models/manifest.json     offline model catalog
tests/                   unit / integration / simulation suites
```

## Tests

```powershell
python -m pytest tests -q
python -m ruff check src tests
```

## Documentation

- `docs/architecture.md` — layers, data flow, event bus, runtime components
- `docs/system-map.md` — per-subsystem state, failure modes, and what is
  actually verified versus assumed
- `docs/memory.md` — durability model, episode pipeline, supersession,
  hallucinated-memory defence
- `docs/identity.md` — identity authority, renaming, name history
- `docs/skills.md` — writing a skill, permissions, the built-in set
- `docs/tools.md` — tool manifests, risk classes, confirmation
- `docs/personality.md` — evidence model, dimensions, update math, contradictions
- `docs/models.md` — slots, providers, fallbacks, offline install, tuning
- `docs/hardware.md` — profiles, RAM budget, Vulkan gating
- `docs/replacing-models.md` — swap in your own models per slot
- `docs/final-report.md` — status report with honest measured numbers
