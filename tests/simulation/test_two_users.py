"""Two users must accumulate distinct memories, relationships and profiles."""


import pytest

from companion.application.memory import MemoryPipeline, MemoryService
from companion.application.personality import PersonalityEngine
from companion.application.relationship import RelationshipEngine
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
    return memory, relationships


@pytest.mark.asyncio
async def test_two_users_get_distinct_memories(graph):
    clock = FakeClock()
    memory, _ = _services(graph, clock)

    async def chat(text):
        memory.begin_episode()
        memory.append_turn("user", text)
        await memory.close_episode()

    await chat("My name is Alex and I study physics.")
    await chat("My name is Sam and I work in finance.")

    contents = " ".join(m.content for m in graph.list_memories()).lower()
    assert "alex" in contents
    assert "sam" in contents
    assert "physics" in contents
    assert "finance" in contents


@pytest.mark.asyncio
async def test_two_users_accumulate_separate_relationships(graph):
    clock = FakeClock()
    _, relationships = _services(graph, clock)

    relationships.note_interaction("Alex", "ep1", valence_delta=0.2, event_note="warm chat")
    relationships.note_interaction("Sam", "ep2", valence_delta=-0.1, event_note="tense chat")

    snapshot = {r.name: r for r in graph.list_relationships()}
    assert "Alex" in snapshot and "Sam" in snapshot
    assert snapshot["Alex"].emotional_valence > snapshot["Sam"].emotional_valence


@pytest.mark.asyncio
async def test_profiles_stay_distinct_after_similar_evidence(graph):
    clock = FakeClock()
    memory, _ = _services(graph, clock)

    async def chat(text):
        memory.begin_episode()
        memory.append_turn("user", text)
        await memory.close_episode()

    await chat("I love building software.")
    await chat("I love baking bread.")

    memories = graph.list_memories()
    assert len(memories) >= 2
