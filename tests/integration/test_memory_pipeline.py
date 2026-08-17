
import pytest


def _turn(service, role, text):
    service.append_turn(role, text)


@pytest.mark.asyncio
async def test_episode_becomes_memory_fact_entity(memory_service, graph):
    memory_service.begin_episode()
    _turn(memory_service, "user", "My name is Alex, I am a physics student.")
    _turn(memory_service, "assistant", "Nice to meet you, Alex!")
    await memory_service.close_episode()

    memories = graph.list_memories()
    assert len(memories) == 1
    assert "alex" in memories[0].content.lower()
    assert memories[0].status.value == "candidate"  # no real embeddings -> stays candidate

    entities = {e.name.lower(): e for e in graph.list_entities()}
    assert "alex" in entities or any("alex" in e.name.lower() for e in entities)

    facts = graph.list_facts()
    assert len(facts) >= 1
    alex_id = next(e.id for e in entities.values() if "alex" in e.name.lower())
    assert any(f.predicate == "has_name" and f.object_id == alex_id
               or "alex" in (f.value or "").lower() for f in facts)


@pytest.mark.asyncio
async def test_dedup_reinforces_not_duplicates(memory_service, graph):
    async def chat(text):
        memory_service.begin_episode()
        _turn(memory_service, "user", text)
        await memory_service.close_episode()

    await chat("My name is Alex, I am a physics student.")
    first = graph.list_memories()[0]
    await chat("My name is Alex, I am a physics student.")
    memories = graph.list_memories()
    assert len(memories) == 1
    assert memories[0].importance > first.importance


@pytest.mark.asyncio
async def test_correct_preserves_history(memory_service, graph):
    memory_service.begin_episode()
    _turn(memory_service, "user", "My name is Alex, I am a physics student.")
    await memory_service.close_episode()
    original = graph.list_memories()[0]

    memory_service.correct_memory(original.id, "My name is Alex, I am a chemistry student.")
    all_memories = graph.list_memories(limit=100)
    assert len(all_memories) == 2
    statuses = [m.status.value for m in all_memories]
    assert "archived" in statuses
    assert any("chemistry" in m.content for m in all_memories)


@pytest.mark.asyncio
async def test_forget_and_lock(memory_service, graph):
    memory_service.begin_episode()
    _turn(memory_service, "user", "My name is Alex, I am a physics student.")
    await memory_service.close_episode()
    mem = graph.list_memories()[0]

    memory_service.lock_memory(mem.id, True)
    locked = graph.get_memory(mem.id)
    assert locked.locked

    memory_service.forget_memory(mem.id)
    forgotten = graph.get_memory(mem.id)
    assert forgotten.status.value == "forgotten"
    assert graph.list_memories(status="") == []  # forgotten excluded from default listing


def test_episode_and_turn_records(memory_service, graph):
    memory_service.begin_episode()
    _turn(memory_service, "user", "hello")
    _turn(memory_service, "assistant", "hi")
    episodes = graph.list_episodes()
    assert len(episodes) == 1
    assert len(episodes[0].transcript) == 2
