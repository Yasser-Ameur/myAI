
import asyncio

import pytest

from companion.application.conversation import (
    ContextAssembler,
    ConversationService,
    LLMResponsePlanner,
)
from companion.application.memory import MemoryPipeline, MemoryService
from companion.application.personality import PersonalityEngine
from companion.application.relationship import RelationshipEngine
from companion.application.retrieval import HybridRetriever
from companion.core.clock import FakeClock
from companion.core.events import EventBus
from companion.domain.agent import AgentState
from companion.domain.personality import PersonalityProfile
from companion.infrastructure.models.llm.mock import MockLLMProvider
from companion.infrastructure.models.router import TaskRouter
from companion.infrastructure.sqlite_graph import CognitiveGraph
from tests.helpers.fixtures import FakeEmbeddings  # noqa: F401


def _service(graph: CognitiveGraph, bus: EventBus):
    clock = FakeClock()
    personality = PersonalityEngine(graph, clock)
    relationships = RelationshipEngine(graph, clock)
    pipeline = MemoryPipeline(
        graph=graph, vector_store=None, embeddings=FakeEmbeddings(),
        clock=clock, personality=personality, relationships=relationships,
        embedding_model_id="fake")
    memory = MemoryService(graph, pipeline, clock)
    retriever = HybridRetriever(
        graph=graph, vector_store=None, embeddings=FakeEmbeddings(),
        clock=clock, embedding_model_id="fake")
    llm = MockLLMProvider(config={
        "model_id": "mock-llm",
        "script": [{"match": "name", "response": "I remember your name from earlier."},
                   {"match": "", "response": "That is interesting, tell me more."}],
    })
    router = TaskRouter({})
    conv = ConversationService(
        llm=llm,
        retriever=retriever,
        assembler=ContextAssembler(total_budget=2000),
        planner=LLMResponsePlanner(llm, router),
        graph=graph,
        memory=memory,
        personality=personality,
        relationships=relationships,
        agent_profile=PersonalityProfile(),
        agent_state=AgentState(),
        bus=bus,
        clock=clock,
        router=router,
    )
    return conv, memory


@pytest.mark.asyncio
async def test_conversation_round_trip(graph):
    bus = EventBus()
    conv, memory = _service(graph, bus)

    memory.begin_episode()
    result = await conv.respond("What is my name?", source="text")
    await memory.close_episode()
    assert result.text
    assert result.plan.intent.value  # a real intent was planned
    assert len(graph.list_episodes()) >= 1


@pytest.mark.asyncio
async def test_conversation_records_memory(graph):
    conv, memory = _service(graph, EventBus())
    memory.begin_episode()
    await conv.respond("My name is Alex.", source="text")
    await memory.close_episode()
    memories = graph.list_memories()
    assert len(memories) >= 1


@pytest.mark.asyncio
async def test_respond_uses_retrieved_memories(graph):
    conv, memory = _service(graph, EventBus())
    memory.begin_episode()
    await conv.respond("My name is Alex and I love hiking.", source="text")
    await memory.close_episode()

    memory.begin_episode()
    result = await conv.respond("Do you remember anything about hiking?", source="text")
    await memory.close_episode()
    assert len(result.retrieved) >= 1


def test_assembler_uses_bracket_sections_and_memory_tags():
    from companion.domain.personality import Trait
    from companion.domain.state import UserState

    profile = PersonalityProfile()
    profile.traits["openness"] = Trait(name="openness", value=0.8,
                                       confidence=0.9, evidence_count=3)
    assembler = ContextAssembler(total_budget=3000)
    text = assembler.build(
        query="hi",
        user_state=UserState(),
        profile=profile,
        retrieved=[],
        goals=[],
        relationships=[],
        agent_state=AgentState(),
        agent_profile=PersonalityProfile(),
        plan=None,
        recent_turns=[{"role": "user", "text": "hello"},
                      {"role": "assistant", "text": "hi there"}],
    )
    assert "[AGENT IDENTITY]" in text
    assert "[USER PROFILE]" in text
    assert "[RESPONSE POLICY]" in text
    assert "[RECENT CONVERSATION]" in text
    assert "[CURRENT USER MESSAGE]" in text
    assert "<current_user_message>\nhi\n</current_user_message>" in text
    assert "user: hello" in text
    assert "companion: hi there" in text


def test_assembler_wraps_retrieved_memory_block():
    from companion.application.retrieval import RetrievedMemory

    assembler = ContextAssembler(total_budget=3000)
    text = assembler.build(
        query="what do I like?",
        user_state=None,
        profile=None,
        retrieved=[RetrievedMemory(id="m1", content="user loves hiking",
                                   source_type="memory", score=0.9)],
        goals=[],
        relationships=[],
        agent_state=None,
    )
    assert text.count("<retrieved_memory>") == 1
    assert text.count("</retrieved_memory>") == 1
    assert "user loves hiking" in text


class _StreamingLLM(MockLLMProvider):
    """Emits tokens with leading whitespace like real llama.cpp deltas."""

    def __init__(self, response: str) -> None:
        super().__init__(config={"model_id": "mock-llm-stream",
                                 "script": [{"match": "", "response": response}]})
        self.response = response

    async def stream(self, request):
        for tok in _spacey_chunks(self.response):
            yield tok

    async def generate(self, request):
        return self._text_response(request)


def _spacey_chunks(text: str):
    """Model llama.cpp deltas: each word token carries its leading space."""
    for i, word in enumerate(text.split()):
        yield word if i == 0 else " " + word


@pytest.mark.asyncio
async def test_streamed_reply_preserves_word_spaces(graph):
    bus = EventBus()
    llm = _StreamingLLM("I hope this works well.")
    clock = FakeClock()
    personality = PersonalityEngine(graph, clock)
    relationships = RelationshipEngine(graph, clock)
    pipeline = MemoryPipeline(
        graph=graph, vector_store=None, embeddings=FakeEmbeddings(),
        clock=clock, personality=personality, relationships=relationships,
        embedding_model_id="fake")
    memory = MemoryService(graph, pipeline, clock)
    retriever = HybridRetriever(
        graph=graph, vector_store=None, embeddings=FakeEmbeddings(),
        clock=clock, embedding_model_id="fake")
    router = TaskRouter({})
    conv = ConversationService(
        llm=llm,
        retriever=retriever,
        assembler=ContextAssembler(total_budget=2000),
        planner=LLMResponsePlanner(llm, router),
        graph=graph,
        memory=memory,
        personality=personality,
        relationships=relationships,
        agent_profile=PersonalityProfile(),
        agent_state=AgentState(),
        bus=bus,
        clock=clock,
        router=router,
    )
    memory.begin_episode()
    result = await conv.respond("hi", source="text")
    await memory.close_episode()
    assert result.text == "I hope this works well."


@pytest.mark.asyncio
async def test_respond_events_carry_session_and_turn_ids(graph):
    from companion.core.events import (
        EVENT_RESPONSE_COMPLETE,
        EVENT_RESPONSE_PLAN_CREATED,
        EVENT_RETRIEVAL_COMPLETE,
    )

    bus = EventBus()
    conv, memory = _service(graph, bus)
    captured = {}

    async def on_event(event) -> None:
        captured[event.kind] = event.payload

    for kind in (EVENT_RETRIEVAL_COMPLETE, EVENT_RESPONSE_PLAN_CREATED, EVENT_RESPONSE_COMPLETE):
        bus.subscribe(kind, on_event)

    memory.begin_episode()
    await conv.respond("hello there", source="text", session_id="sess_x", turn_id="turn_y")
    await memory.close_episode()
    await asyncio.sleep(0.05)  # let the bus deliver queued events to subscribers

    assert captured[EVENT_RETRIEVAL_COMPLETE]["session_id"] == "sess_x"
    assert captured[EVENT_RETRIEVAL_COMPLETE]["turn_id"] == "turn_y"
    assert captured[EVENT_RESPONSE_PLAN_CREATED]["session_id"] == "sess_x"
    assert captured[EVENT_RESPONSE_COMPLETE]["turn_id"] == "turn_y"
