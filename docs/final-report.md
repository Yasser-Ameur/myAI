# Status report

Measured on the target machine: Intel Core i7-1185G7 @ 3.00 GHz, 16 GB RAM,
Intel Iris Xe (shared memory), Windows 11, CPU-only execution, profile
`balanced`, AI budget 8192 MB.

Every number below was produced by running the system with real weights. Where
something was not verified, it says so.

## The defect that mattered

**No cognitive write had ever been persisted.**

`SqliteStorage` connected with sqlite3's default `isolation_level=''`, which
opens an implicit transaction before every INSERT/UPDATE/DELETE. Nothing in
`sqlite_graph.py` committed (1 of 685 lines used a transaction), and `close()`
discards an open transaction. The user's real `data/cognitive.db` contained
**zero rows in every cognitive table** — turns, episodes, memories, facts,
entities, goals, relationships — after every session ever run.

It was invisible to the 98-test suite because an open connection sees its own
uncommitted data, so every in-process assertion passed. The system was, in
practice, exactly what it was not supposed to be: an LLM with chat history in
RAM.

Everything else about memory, identity and personality was downstream of this.

## Bugs found and fixed

| # | Bug | Evidence | Fix |
|---|-----|----------|-----|
| 1 | SQLite writes never committed; all state lost at exit | real DB had 0 rows in all tables | autocommit + explicit `transaction()` for atomic blocks |
| 2 | `doctor` reported **OK** while the graph was dead | log said "database is locked", doctor said `DOCTOR OK` | per-check status, live write/read-back probe, exit 1 on problems |
| 3 | Graph failure left `storage` non-None, so degradation was invisible | `CognitiveGraph()` raising still produced "store: ok" | storage closed and nulled on failure; degradation surfaced everywhere |
| 4 | Episode consolidation never ran in the real runtime | `aclose()` never called `close_episode()` | bounded (20 s) consolidation on shutdown; unfinished episodes retried at idle |
| 5 | Agent name never reached the prompt | `_agent_identity()` returned a fixed string with no name | identity section built from the persistent self-model |
| 6 | Learned identity never reconstructed | identity built from config only | `SelfModelService.load()` restores from the graph before any model call |
| 7 | Agent identity could not be *stated* | extractor had `my name is`, nothing for "your name is X" | full detection set incl. bare copula, hedge and question guards |
| 8 | LLM extractor never wired | `MemoryPipeline` built without an extractor → rules only, always | `StructuredExtractor(llm.fast, router)` injected |
| 9 | Assistant's own prose stored as user memory | consolidation stored "Memory is like a recording device…" | grounding filter; ungrounded extractions dropped and counted |
| 10 | Idle reflection marked episodes consolidated after only a summary | permanently prevented real extraction | reflection runs the full pipeline; leaves episode open if it cannot |
| 11 | `Observation` rows written but unreadable | no `list_observations` | added; `companion why <id>` follows the evidence chain |
| 12 | LLM planning cost 2.1 s/turn and emitted `verbosity: 0.0` (64-token truncation) | direct measurement | rules planner by default; verbosity floor when LLM planning is on |
| 13 | Correction referent lost across restart | "it's blue now" resolved only in-process | referent recovered from the graph by value |
| 14 | Superseded values still retrievable as active memories | mirror rows never archived | supersession archives the mirror |

## Performance

Turn latency, three consecutive LLM turns, identical prompts:

| | before | after |
|---|---|---|
| LLM turn | 12330 / 10451 / 12088 ms | **5274 / 4194 / 4458 ms** |
| mean | 11.6 s | **4.6 s (2.5× faster)** |

The cause was measured, not guessed: an LLM planning call before every
generation cost **2093 ms**, against **7 ms** for retrieval. Removing it from
the default path is the entire gain; generation itself is unchanged at roughly
10 tok/s for Qwen3-1.7B q4_k_m on 8 CPU threads.

| Stage | Measured |
|---|---|
| `build()` (config, storage, registry, skills) | 2416 ms |
| LLM load (Qwen3-1.7B q4_k_m) | 1143 ms |
| Embeddings load (bge-small-en) | 220 ms |
| Retrieval (hybrid, top_k=8) | **7 ms** |
| Planning (rules) | <1 ms |
| Planning (LLM, opt-in) | 2093 ms |
| Skill-routed turn (calculator, no LLM) | 599 ms |
| LLM turn | 4.2–5.3 s |
| Consolidation at shutdown (LLM extraction) | 2840 ms |
| Process RSS, all models loaded | 3825 MB |

RSS stays well inside the 8192 MB budget with LLM + embeddings resident.

## What was verified end-to-end

Run with real weights, in **separate OS processes**:

- **Agent identity across restart** — name persists, `source=persisted` at boot
- **Rename with history** — Jarvis → Friday; "what was your previous name?" → Jarvis
- **User memory across restart** — favourite colour recalled
- **Correction with history** — purple → blue; "what used to be" → purple
- **Hedged guess refused** — "I think your name might be Bob" does not rename
- **Skill routing** — identity, recall, provenance, capabilities, calculator
- **Exact arithmetic** — 1234 × 5678 = 7,006,652 (via tool, not the model)
- **Live capability listing** — "what can you do?" from the registry
- **Graceful degradation** — unwritable DB: app runs, states the real reason,
  skills keep working, `doctor` exits 1
- **Hallucinated memory rejected** — invented facts dropped by the grounding filter

## Tests

**148 passing, lint clean** (`ruff check src tests`).

- 98 pre-existing (unchanged, still green)
- +29 skills/tools: permissions default-deny, manifest validation, API version
  compatibility, calculator safety (7 hostile inputs), tool timeouts,
  confirmation gating, router failure isolation, memory namespacing
- +14 salience/grounding: explicit-statement extraction, questions asserting
  nothing, supersession archiving, hallucinated-memory rejection
- +7 durability: including 2 **real subprocess restart** tests
  (`COMPANION_E2E=1`), which are the only tests that can catch the class of bug
  described above

## Architecture added

- `application/facts.py` — the single durable-write path (supersession,
  authority, confirmation, evidence)
- `application/identity.py` — self-model, identity authority, name history
- `application/salience.py` — per-turn durable commit
- `skills/` — manifest, permissions (default deny), registry, validator,
  loader, router, 8 built-in skills
- `tools/` — manifest, risk classes, validation, timeouts, confirmation
  gating, 3 built-in tools
- `runtime/portability.py` — JSON export/import of cognitive state, no weights

## Known limitations

Stated plainly.

- **Voice, camera and avatar were not verified end-to-end in this pass.** The
  weights are installed and the providers construct, but STT/TTS/VAD/vision
  need a microphone, speaker and camera. They are marked "REAL (untested here)"
  in `docs/system-map.md`, not claimed as working.
- **Generation is ~10 tok/s.** 4.6 s per turn is dominated by decode. Only a
  smaller model, a shorter reply budget, or GPU offload changes this; the
  cognitive architecture cannot.
- **Salient extraction is English-only.** French utterances fall through to
  LLM consolidation, so they persist with lower confidence rather than as
  high-authority facts. Multilingual support is not done.
- **The grounding filter is lexical.** It will reject a correctly-inferred fact
  that shares no vocabulary with the user's phrasing. Deliberately biased
  toward false negatives — a missing memory is recoverable, a fabricated one
  is not.
- **Consolidation can exceed its 20 s shutdown bound** on long sessions. It
  degrades correctly (per-turn facts are already durable, episode retried at
  idle) and says so, but the session's subtler extractions are delayed.
- **Skill routing is regex-based.** Precise and fast for the phrasings it
  covers, silent on paraphrases it does not; those fall through to the LLM.
- **No longitudinal simulation.** Personality convergence over hundreds of
  synthetic conversations is not measured.
- **`psutil` is an optional import.** Without it the memory guard cannot read
  live RSS. It is now installed in this environment but is not a hard
  dependency.

## Meaningful next steps

1. Verify the voice pipeline on real hardware with a trace ID per turn through
   VAD → STT → retrieval → LLM → TTS → playback.
2. French/mixed-language salient extraction.
3. Longitudinal simulation to measure personality convergence and false-memory
   rate over time.
4. Move consolidation fully off the shutdown path onto an idle-only budget.
5. A `notes` skill and document ingestion, exercising skill-scoped memory.
