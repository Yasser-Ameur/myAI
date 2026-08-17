from companion.application.extraction import RuleBasedExtractor


def _extract(text: str):
    return RuleBasedExtractor().extract(text, "ep_test")


def test_identity_memory():
    result = _extract("My name is Alex, I am a physics student.")
    assert any("alex" in m.content.lower() for m in result.memories)
    assert any(m.type == "semantic" for m in result.memories)


def test_preference_memory_and_personality_evidence():
    result = _extract("I love building software tools that help people learn.")
    assert any(m.type == "preference" for m in result.memories)
    assert any(e.target.startswith("likes:") for e in result.personality_evidence)


def test_goal_extraction():
    result = _extract("I want to become a doctor and help children.")
    names = [g.get("name", "") for g in result.goals]
    assert any("doctor" in n.lower() for n in names)


def test_relationship_extraction():
    result = _extract("My son wants to study medicine.")
    assert any(r.get("person", "") == "son" for r in result.relationships)


def test_empty_handling():
    result = _extract("")
    assert result.memories == []
    assert result.personality_evidence == []
