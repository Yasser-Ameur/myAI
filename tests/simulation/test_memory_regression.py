"""Memory regression tests from the spec.

- Distinct memories must stay distinct (mountain hikes vs work tasks).
- Temporal facts must answer the right query for the right period.
"""


import pytest

from companion.application.memory import MemoryPipeline, MemoryService
from companion.application.personality import PersonalityEngine
from companion.application.relationship import RelationshipEngine
from companion.application.retrieval import HybridRetriever
from companion.core.clock import FakeClock
from companion.infrastructure.sqlite_graph import CognitiveGraph
from tests.helpers.fixtures import FakeEmbeddings


def _services(graph: CognitiveGraph, clock: FakeClock):
    personality = PersonalityEngine(graph, clock)
    relationships = RelationshipEngine(graph, clock)
    pipeline = MemoryPipeline(
        graph=graph, vector_store=None, embeddings=FakeEmbeddings(), clock=clock,
        personality=personality, relationships=relationships, embedding_model_id="fake")
    memory = MemoryService(graph, pipeline, clock)
    retriever = HybridRetriever(graph=graph, vector_store=None,
                                embeddings=FakeEmbeddings(), clock=clock,
                                embedding_model_id="fake")
    return memory, retriever, pipeline


@pytest.mark.asyncio
async def test_distinct_memories_stay_distinct(graph):
    clock = FakeClock()
    memory, retriever, pipeline = _services(graph, clock)

    async def chat(text):
        memory.begin_episode()
        memory.append_turn("user", text)
        await memory.close_episode()

    await chat("I love hiking in the mountains every weekend.")
    await chat("I am working hard on my thesis about memory systems.")
    await chat("My project involves building a personal assistant.")

    memories = graph.list_memories()
    assert len(memories) >= 3
    # distinct topics must not collapse into a single memory
    contents = " ".join(m.content for m in memories).lower()
    assert "hiking" in contents
    assert "thesis" in contents or "memory systems" in contents

    hits = await retriever.retrieve("tell me about hiking", mode="auto", top_k=3)
    top = hits[0].content.lower() if hits else ""
    assert "hiking" in top


@pytest.mark.asyncio
async def test_temporal_facts_answer_period_questions(graph):
    """2025 the user studied physics; in 2026 they study computer science."""
    clock = FakeClock(start=1_750_000_000.0)  # ~mid 2025
    memory, retriever, _ = _services(graph, clock)

    # note: entity id is shared; facts are temporal on the same predicate
    from companion.core.ids import new_entity_id, new_fact_id
    from companion.domain.graph import Entity, Fact

    subject = Entity(id=new_entity_id(), type="person", name="alex",
                     created_at=clock.now_iso(), updated_at=clock.now_iso())
    graph.upsert_entity(subject)

    f1 = Fact(id=new_fact_id(), subject_id=subject.id, predicate="studies",
              value="physics", confidence=0.9, importance=0.5,
              created_at=clock.now_iso(), valid_from=clock.now_iso(),
              source_episode_id="ep1", provenance="conversation")
    graph.add_fact(f1)

    clock.advance(366 * 86400)  # ~1 year later, now 2026
    graph.invalidate_fact(f1.id, clock.now_iso())
    f2 = Fact(id=new_fact_id(), subject_id=subject.id, predicate="studies",
              value="computer science", confidence=0.9, importance=0.5,
              created_at=clock.now_iso(), valid_from=clock.now_iso(),
              source_episode_id="ep2", provenance="conversation")
    graph.add_fact(f2)

    current = graph.list_facts(subject.id)
    assert len(current) == 1
    assert current[0].value == "computer science"

    # history preserved: old fact visible with include_deleted / all
    all_facts = graph.list_facts(subject.id, include_deleted=True)
    objects = {f.value for f in all_facts if f.value}
    assert {"physics", "computer science"}.issubset(objects)


@pytest.mark.asyncio
async def test_reinforce_changes_retrieval_ranking(graph):
    clock = FakeClock()
    memory, retriever, _ = _services(graph, clock)

    async def chat(text):
        memory.begin_episode()
        memory.append_turn("user", text)
        await memory.close_episode()

    await chat("I love hiking in the mountains.")
    # reinforce the hiking memory twice
    await chat("I love hiking in the mountains every weekend.")
    await chat("Mountain hiking is my favorite hobby.")

    hits = await retriever.retrieve("hiking", mode="auto", top_k=2)
    assert hits and "hiking" in hits[0].content.lower()
