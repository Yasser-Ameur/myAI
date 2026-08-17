from companion.core.clock import FakeClock
from companion.core.ids import new_entity_id, new_fact_id
from companion.domain.graph import Entity, Fact
from companion.infrastructure.sqlite_graph import CognitiveGraph


def _entity(graph: CognitiveGraph, name: str) -> Entity:
    e = Entity(id=new_entity_id(), type="person", name=name,
               created_at="2024-01-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z")
    graph.upsert_entity(e)
    return e


def _fact(graph: CognitiveGraph, subject: Entity, predicate: str, value: str,
          valid_from: str, confidence: float = 0.9) -> Fact:
    f = Fact(
        id=new_fact_id(), subject_id=subject.id, predicate=predicate,
        value=value, confidence=confidence, importance=0.5,
        created_at=valid_from, valid_from=valid_from,
        source_episode_id="ep-test", provenance="test",
    )
    graph.add_fact(f)
    return f


def test_temporal_facts_hidden_after_invalidation(graph: CognitiveGraph):
    clock = FakeClock(start=1_700_000_000.0)
    subject = _entity(graph, "alex")
    f = _fact(graph, subject, "studies", "physics", clock.now_iso())

    assert graph.list_facts(subject.id) != []

    clock.advance(3600)
    graph.invalidate_fact(f.id, clock.now_iso())
    assert graph.list_facts(subject.id) == []

    # historical information is never destroyed silently
    assert graph.list_facts(subject.id, include_deleted=True) != []


def test_active_and_past_facts_coexist(graph: CognitiveGraph):
    clock = FakeClock(start=1_700_000_000.0)
    subject = _entity(graph, "alex")

    past = _fact(graph, subject, "studied", "physics", clock.now_iso(), confidence=0.8)
    clock.advance(10 * 86400)
    clock.advance(3600)
    graph.invalidate_fact(past.id, clock.now_iso())
    _fact(graph, subject, "studies", "computer science", clock.now_iso())

    facts = graph.list_facts(subject.id)
    assert any(f.predicate == "studies" and f.value == "computer science" for f in facts)
    assert not any(f.predicate == "studied" for f in facts)


def test_fact_updates_preserve_history(graph: CognitiveGraph):
    clock = FakeClock(start=1_700_000_000.0)
    subject = _entity(graph, "alex")
    f1 = _fact(graph, subject, "lives_in", "paris", clock.now_iso())
    clock.advance(86400)
    graph.invalidate_fact(f1.id, clock.now_iso())
    _fact(graph, subject, "lives_in", "lyon", clock.now_iso())

    facts = graph.list_facts(subject.id)
    assert len(facts) == 1
    assert facts[0].value == "lyon"


def test_relationships_snapshot(graph: CognitiveGraph):
    graph.upsert_relationship(graph_relationship(graph))
    rels = graph.list_relationships()
    assert len(rels) == 1
    assert rels[0].name == "Alex"


def graph_relationship(graph):
    from companion.domain.relationship import Relationship

    return Relationship(subject_id="usr_0", target_id="usr_1", name="Alex",
                        emotional_valence=0.5, familiarity=0.3, interaction_count=1)
