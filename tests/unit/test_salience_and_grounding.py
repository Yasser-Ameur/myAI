"""Turn-level extraction and hallucinated-memory defence."""

from __future__ import annotations

import asyncio

import pytest

from companion.application.extraction import ExtractedMemory, ExtractionResult
from companion.application.identity import SelfModelService
from companion.application.memory import MemoryPipeline, _grounded_in
from companion.application.salience import SalientExtractor, TurnCommitter
from companion.domain.memory import Episode

# ---------------------------------------------------------------------------
# salient extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,predicate,value", [
    ("My favorite color is purple.", "favorite:color", "purple"),
    ("my favourite food is ramen", "favorite:food", "ramen"),
    ("I live in Paris.", "lives_in", "Paris"),
    ("I prefer concise technical explanations.",
     "prefers:concise technical explanations", "yes"),
])
def test_extracts_explicit_statements(text, predicate, value):
    facts, _ = SalientExtractor().extract(text)
    found = {f.predicate: f.value for f in facts}
    assert predicate in found, f"{text!r} produced {found}"
    assert found[predicate].lower() == value.lower()


def test_questions_assert_nothing():
    facts, _ = SalientExtractor().extract("What is my favorite color?")
    assert facts == []


def test_frustration_is_recorded_as_experience_not_trait():
    facts, _ = SalientExtractor().extract("I've been frustrated with the architecture.")
    assert any(f.predicate == "experience:frustrated"
               and "architecture" in f.value for f in facts)
    # It must not be written as a stable personality claim.
    assert not any(f.predicate.startswith("trait:") for f in facts)


def test_explicit_remember_is_high_importance():
    facts, _ = SalientExtractor().extract(
        "Remember that I want this project to eventually have skills.")
    memory_facts = [f for f in facts if f.predicate == "explicit_memory"]
    assert memory_facts and memory_facts[0].importance >= 0.8


def test_negation_marks_dislike(graph, clock):
    facts, _ = SalientExtractor().extract("I don't like purple anymore.")
    assert any(f.predicate == "opinion:purple" and f.value == "dislikes" for f in facts)


def test_commit_creates_goal_for_intent(graph, clock):
    self_model = SelfModelService(graph, clock, configured_name="C")
    self_model.load()
    committer = TurnCommitter(graph, self_model, clock=clock)
    result = committer.commit("I want to add a planner to the project.", episode_id="ep")
    assert result.goals
    assert any("planner" in g.name.lower() for g in graph.list_goals(status="active"))


def test_superseded_memory_mirror_is_archived(graph, clock):
    """The old value must not remain retrievable as if still true."""
    self_model = SelfModelService(graph, clock, configured_name="C")
    self_model.load()
    committer = TurnCommitter(graph, self_model, clock=clock)
    committer.commit("My favorite color is purple.", episode_id="ep1")
    committer.commit("My favorite color is blue.", episode_id="ep2")

    active = [m.content for m in graph.list_memories(status="validated", limit=100)]
    assert any("blue" in c for c in active)
    assert not any("purple" in c for c in active), (
        "superseded value still retrievable as an active memory"
    )


# ---------------------------------------------------------------------------
# hallucinated-memory defence
# ---------------------------------------------------------------------------

def test_grounding_accepts_what_the_user_said():
    mem = ExtractedMemory(content="user likes ramen", object="ramen")
    assert _grounded_in(mem, "i really like ramen and sushi")


def test_grounding_rejects_model_invention():
    mem = ExtractedMemory(content="user is a professional chef in Lyon",
                          object="professional chef in Lyon")
    assert not _grounded_in(mem, "i like ramen")


def test_grounding_rejects_assistant_prose():
    """The observed failure: the model's own answer stored as a user fact."""
    mem = ExtractedMemory(
        content="Memory is like a recording device, capturing moments and experiences.")
    assert not _grounded_in(mem, "tell me something short about memory")


class _InventingExtractor:
    """Returns one grounded and one invented memory."""

    async def extract(self, transcript: str, episode_id: str = "") -> ExtractionResult:
        return ExtractionResult(
            memories=[
                ExtractedMemory(content="user likes ramen", object="ramen",
                                subject="user", predicate="likes"),
                ExtractedMemory(content="user owns three yachts in Monaco",
                                object="three yachts in Monaco",
                                subject="user", predicate="owns"),
            ],
            method="llm",
        )


def test_pipeline_persists_grounded_and_drops_invented(graph, clock):
    pipeline = MemoryPipeline(graph=graph, vector_store=None, embeddings=None,
                              clock=clock, extractor=_InventingExtractor())
    episode = Episode(id="ep-1", transcript=[
        {"role": "user", "text": "I like ramen."},
        {"role": "assistant", "text": "Noted."},
    ])
    asyncio.run(pipeline.process_episode(episode))

    contents = [m.content for m in graph.list_memories(limit=100)]
    assert any("ramen" in c for c in contents)
    assert not any("yacht" in c.lower() for c in contents), (
        "an invented memory was persisted"
    )
    assert pipeline.stats.get("ungrounded_rejected") == 1
