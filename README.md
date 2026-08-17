# myAI

Local-first multimodal cognitive AI companion with persistent temporal memory,
probabilistic personality, agent identity, modular skills and tools, local model
adapters, voice, vision and expressive embodiment.

Everything runs on your machine. No cloud, no telemetry, no accounts.

## What it does

- **Remembers** what you tell it across sessions using a temporal knowledge
  graph (SQLite) with hybrid retrieval (lexical + semantic + graph + recency
  + importance + confidence).
- **Learns your personality** through a probabilistic evidence-based model
  with traits, values, preferences, contradiction tracking, and communication
  style adaptation.
- **Keeps an identity** you name once and it stays named, across restarts.
  Renaming keeps the old name queryable; a hedged guess cannot overwrite a
  stated identity.
- **Routes exact questions to skills** before hitting the model: arithmetic,
  datetime, identity recall, goal management, provenance, diagnostics.
- **Runs locally** with replaceable model adapters for LLM, STT, TTS, VAD,
  vision and embeddings — every slot degrades to a deterministic fallback
  when a provider is missing.

## Why it is different

This is not a chatbot wrapper. It is a cognitive architecture:

- **Durable memory** — salient facts commit per turn (~1 ms, no model), so
  an ungraceful exit loses nothing. Consolidation runs at idle for subtler
  extractions.
- **Supersession, not overwrite** — changing a fact closes the old row and
  inserts a new one. History is never silently deleted.
- **Hallucinated-memory defence** — every LLM extraction must be traceable
  to the user's own words. Ungrounded extractions are dropped.
- **Default-deny skills** — a skill gets only what its manifest declares
  and the config grants. No shell, no eval, no side effects without
  confirmation.
- **Honest benchmarks** — fallback performance is always labelled
  `SIMULATED`; real model performance is labelled `REAL`.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  interfaces/    CLI (14 subcommands) + HTTP API  │
├─────────────────────────────────────────────────┤
│  runtime/       Orchestration, config, hardware, │
│                 memory guard, scheduler, metrics  │
├─────────────────────────────────────────────────┤
│  application/   Conversation, memory pipeline,   │
│                 extraction, personality, retrieval │
│                 identity, salience, facts         │
├─────────────────────────────────────────────────┤
│  domain/        Pure data: Entity, Fact, Memory,  │
│                 PersonalityProfile, Relationship, │
│                 UserState, AgentState             │
├─────────────────────────────────────────────────┤
│  core/          Protocols, events, ids, clock,    │
│                 errors, types                     │
└─────────────────────────────────────────────────┘
         ↕ contracts (Protocols)
┌─────────────────────────────────────────────────┐
│  infrastructure/  SQLite graph, vector store,     │
│                   model adapters (LLM, STT, TTS,  │
│                   VAD, vision, embeddings)         │
└─────────────────────────────────────────────────┘
```

Application code never imports concrete providers. Every optional dependency
degrades gracefully to a deterministic mock fallback with a clear warning.

## Skills

Bounded, declared, permissioned capabilities with default-deny access:

| Skill | What it does | Permissions |
|-------|-------------|-------------|
| `identity` | Name and history | `memory.read` |
| `recall` | Current and historical facts | `memory.read` |
| `provenance` | Why I believe something | `memory.read` |
| `goals` | Goal management | `memory.read`, `memory.write` |
| `capabilities` | What I can do | — |
| `diagnostics` | Health and status | `runtime.inspect` |
| `calculator` | Exact arithmetic (AST whitelist) | — |
| `datetime` | Current date and time | — |

See `docs/skills.md` for writing your own.

## Hardware

Targets a modest CPU laptop with shared-memory iGPU — no discrete GPU
required. Profiles: `ultra_low` (8 GB), `balanced` (16 GB), `performance`,
`gpu`, `auto`. The memory guard enforces a RAM budget and unloads models
under pressure.

Measured on Intel Core i7-1185G7 @ 3.00 GHz, 16 GB RAM, Intel Iris Xe,
CPU-only execution: Qwen3-1.7B q4_k_m at ~10 tok/s, LLM turn ~4.2 s,
retrieval 7 ms, process RSS 3.8 GB.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\Activate on Windows
pip install -e .
pip install -e ".[dev]"
```

Runs with no models installed (fallback mocks, fully deterministic):

```bash
companion run
```

### Model installation

Models are downloaded on demand. No weights are shipped with the repository.

```bash
# Install optional provider dependencies
pip install -e ".[llm]"       # llama-cpp-python
pip install -e ".[stt]"       # faster-whisper
pip install -e ".[tts]"       # kokoro-onnx, soundfile, piper-tts
pip install -e ".[embeddings]"
pip install -e ".[vad]"       # onnxruntime
pip install -e ".[vision]"    # mediapipe

# Download and verify models
companion models list
companion models install qwen3-1.7b whisper-base kokoro-82m bge-small-en silero-vad face-landmarker
```

See `docs/models.md` and `docs/replacing-models.md` for offline install and
swapping in your own models per slot.

## Quick start

```bash
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

```bash
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

## Configuration

All provider and model selection lives in `config/companion.yaml`. Application
code never hard-codes a model name. Environment variables can override any
value.

## Privacy

- Cloud is disabled by default (`privacy.cloud_enabled: false`).
- Telemetry is off (`privacy.telemetry: false`).
- The API server binds to `127.0.0.1` only.
- All data stays in `data/cognitive.db` on your machine.
- You can export and audit your entire cognitive state:
  `companion data export ./backup`.

## Testing

```bash
python -m pytest tests -q
python -m ruff check src tests
```

146 tests covering unit, integration and simulation suites. The E2E durability
tests (`COMPANION_E2E=1`) run real subprocess restarts to verify memory
survives process termination.

## Documentation

- `docs/architecture.md` — layers, data flow, event bus, runtime components
- `docs/system-map.md` — per-subsystem state, failure modes, verification status
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

## Known limitations

- Voice, camera and avatar need real hardware; not verified end-to-end on
  CI.
- Generation is ~10 tok/s on CPU. Only a smaller model, shorter budget, or
  GPU offload changes this.
- Salient extraction is English-only.
- The grounding filter is lexical — deliberately biased toward false
  negatives.
- Skill routing is regex-based; paraphrases fall through to the model.

## Roadmap

1. Verify the voice pipeline end-to-end on real hardware.
2. French/mixed-language salient extraction.
3. Longitudinal simulation for personality convergence.
4. Move consolidation fully to idle-only budget.
5. Document ingestion and notes skill.

## License

MIT — see [LICENSE](LICENSE).
