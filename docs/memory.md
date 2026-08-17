# Memory

Memory is the companion's durable world model. It lives in a single SQLite
database (`data/cognitive.db`).

## Durability

Two things make a claim durable, and both matter.

**1. Writes are committed.** `SqliteStorage` opens its connection with
`isolation_level=None` (autocommit). This is load-bearing, not incidental:
with the driver default (`''`), sqlite3 opens an implicit transaction before
every INSERT/UPDATE/DELETE and `close()` discards it. Multi-statement
atomicity is expressed explicitly through `storage.transaction()`
(`BEGIN IMMEDIATE … COMMIT`).

> This was a real defect. Every cognitive write in the project was being
> silently discarded at process exit — the database contained zero rows in
> every table despite many sessions. It was invisible to the test suite
> because an open connection sees its own uncommitted data, so in-process
> assertions passed. `tests/integration/test_persistence_across_restart.py`
> guards it now by closing the connection, and by running real subprocesses.

**2. Salient facts are committed per turn, not per episode.** Consolidation
alone made durability hostage to a graceful shutdown; a Ctrl+C lost the
session. So the pipeline is split:

| | when | cost | what it captures |
|---|---|---|---|
| **Turn commit** (`application/salience.py`) | every user turn, inline | ~1 ms, no model | explicit statements: identities, favourites, corrections, goals, "remember that …" |
| **Consolidation** (`application/memory.py`) | episode close + idle reflection | seconds, uses `llm.fast` | everything subtler; dedup, contradictions, summaries |

The turn committer is deliberately narrow: anything it writes is committed with
high authority, so it only fires on statements whose meaning is unambiguous in
the surface form. Everything else waits for consolidation, where it lands with
lower confidence.

If consolidation is cut short (shutdown has a 20 s bound), the episode stays
marked unconsolidated and the idle reflection pass runs the *full* pipeline
over it later.

## Supersession, not overwrite

All durable writes go through `FactWriter`, which owns four policies:

- **Supersede** — a new value for an occupied slot closes the old fact
  (`valid_to = now`) and inserts a new row. "What used to be my favourite
  colour?" is a real query, not a guess.
- **Authority** — a lower-authority claim cannot displace a higher-authority
  incumbent (see `docs/identity.md`).
- **Confirm** — re-stating the same value stamps `last_confirmed_at` and moves
  confidence a fraction toward certainty rather than duplicating the fact.
- **Evidence** — every assertion records an `Observation` with the utterance,
  so `companion why <fact-id>` can reconstruct the chain.

A slot is `(subject, predicate)`. Predicates that name a category encode it —
`favorite:color`, not `has_favorite` — so setting a favourite colour does not
invalidate a favourite food.

When a fact is superseded, its readable mirror in the `memories` table is
archived, so the old value cannot resurface through lexical or semantic
retrieval as though it were still true.

## Hallucinated-memory defence

A model handed a transcript will extract its *own* answers as facts about the
user. This was observed, not hypothesised: consolidation stored "Memory is like
a recording device…" — the assistant's own sentence — as a user memory.

Anything the LLM extractor proposes must now be traceable to the user's own
words: the claim's object, or a majority of its informative tokens, must appear
in the user turns. Ungrounded extractions are dropped and counted
(`pipeline.stats["ungrounded_rejected"]`). The check is deliberately lexical —
cheap, deterministic, and incapable of hallucinating on its own.

## Episode pipeline

```
begin_episode()
  append_turn(user, "...")   / append_turn(assistant, "...")
  turn commit -> durable facts (immediate)
close_episode()
  -> StructuredExtractor.extract(transcript)
       |  LLM extraction (strict JSON, validated) if an LLM is available
       `- RuleBasedExtractor (deterministic fallback, no weights)
  -> grounding filter: drop extractions not supported by the user's own words
  -> for each memory:  dedup check -> add_memory (candidate) -> embedding
  -> for each extracted fact/goal/relationship/evidence: commit to graph
  -> personality evidence applied (conservative updates)
```

Turns are recorded with a stable `turn_id = "turn:{episode_id}:{index}"` so
repeated turns can be identified without collision.

## Storage model

| Object       | Notes                                                                 |
|--------------|-----------------------------------------------------------------------|
| `Episode`    | transcript of `{role, text, ...}` dicts, `started_at`, `ended_at`, `summary` |
| `Memory`     | `content`, `type` (semantic/episodic/preference/..., `MemoryType`), `status` (`candidate` -> `validated`), `importance`, `confidence`, `locked`, `meta` |
| `Fact`       | `subject_id`, `predicate`, `object_id` **or** `value`, `confidence`, `importance`, `valid_from`/`valid_to`, `source_episode_id`, `provenance` |
| `Entity`     | typed nodes (person, organization, concept, ...)                        |
| `Goal`       | `name`, `description`, `status`, `priority`, `confidence`              |
| `Relationship` | `subject_id`, `target_id`, `name`, `trust`, `familiarity`, `emotional_valence`, `interaction_count`, `important_events` |
| `Source`     | provenance record; `Source.type` is the `SourceType` enum               |
| `KnowledgeChunk` | injected knowledge, retrieved on lexical overlap                      |

## Dedup

New memories are compared against existing ones with a token-Jaccard
similarity. Above a threshold they are treated as the same fact:

- the existing memory is **reinforced** (`importance` rises, confidence
  updates) instead of creating a duplicate, and
- the stored fact's `last_confirmed_at` is refreshed.

## Temporal semantics

Facts are never edited in place. Changing a fact:

1. `invalidate_fact(id, now)` sets `valid_to` (and hides it from default
   listings).
2. A new fact row is inserted with the new value and `valid_from = now`.

`list_facts(subject_id, predicate, include_deleted=False)` returns only active
facts by default; `include_deleted=True` reveals history. This is what lets
"in 2025 Alex studied physics; in 2026 he studies computer science" coexist:
both facts exist, only the current one is active. `forget_memory`/`lock_memory`
and `correct_memory` (which archives the old row and writes a new one) follow
the same "never destroy history" rule.

## Retrieval

`HybridRetriever` combines:

- **lexical** — informative-token match (stopwords removed; query-focused, not
  diluted by content length)
- **semantic** — vector search over memory/fact embeddings when an embedding
  model is present
- **graph** — entity-anchored facts
- **recency** — exponential decay with configurable half-life (default 30d)
- **importance** and **confidence**

Modes: `recent | semantic | entity | relationship | temporal | goal |
personality | episodic`, or `auto` (recency + semantic + entity). Candidates
are deduplicated and sorted by the weighted score, then `top_k` returned.
