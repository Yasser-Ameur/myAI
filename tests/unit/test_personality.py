from companion.application.personality import UPDATE_RATES, PersonalityEngine
from companion.application.ports import NullGraphStore
from companion.core.clock import SystemClock
from companion.domain.personality import (
    AGENT_PERSONALITY_DIMENSIONS,
    COMMUNICATION_PREFERENCES,
    CORE_VALUES,
    PersonalityEvidence,
)


def _engine():
    return PersonalityEngine(NullGraphStore(), SystemClock(), update_mode="balanced")


def test_profile_has_all_core_dimensions():
    profile = _engine().profile()
    assert set(profile.traits) == set(AGENT_PERSONALITY_DIMENSIONS)
    assert set(profile.values) == set(CORE_VALUES)
    assert set(profile.preferences) == set(COMMUNICATION_PREFERENCES)


def test_stable_traits_resist_single_evidence():
    engine = _engine()
    t = engine.profile().traits["warmth"]
    t.stability = 0.9
    t.value = 0.5
    before = t.value
    engine.apply_evidence(PersonalityEvidence(
        target="warmth", direction="positive", strength=0.35, confidence=0.9))
    assert abs(engine.profile().traits["warmth"].value - before) < 0.05


def test_repeated_evidence_moves_unstable_trait():
    engine = _engine()
    t = engine.profile().traits["curiosity"]
    t.stability = 0.4
    t.value = 0.5
    for _ in range(8):
        engine.apply_evidence(PersonalityEvidence(
            target="curiosity", direction="positive", strength=0.3, confidence=0.8))
    assert engine.profile().traits["curiosity"].value > 0.52


def test_contradicting_evidence_does_not_erase_prior():
    engine = _engine()
    engine.apply_evidence(PersonalityEvidence(
        target="energy", direction="positive", strength=0.4, confidence=0.9,
        context="loves being busy", timestamp=SystemClock().now_iso()))
    after_positive = engine.profile().traits["energy"].value
    engine.apply_evidence(PersonalityEvidence(
        target="energy", direction="negative", strength=0.4, confidence=0.9,
        context="always exhausted", timestamp=SystemClock().now_iso()))
    moved = engine.profile().traits["energy"].value
    # a single contradicting statement must not erase the accumulated evidence
    assert abs(moved - after_positive) < 0.15


def test_update_modes_exist_and_bounded():
    assert UPDATE_RATES["conservative"] < UPDATE_RATES["balanced"] < UPDATE_RATES["responsive"]
    assert all(0 < rate < 0.2 for rate in UPDATE_RATES.values())


def test_communication_preference_learning():
    engine = _engine()
    engine.learn_communication_preference(
        "Keep it short, just the answer please", episode_id="ep1")
    prefs = engine.profile().preferences
    assert prefs["preferred_response_length"].value < 0.5
    assert prefs["preferred_response_length"].confidence > 0.0


def test_directional_evidence_signed_strength():
    assert PersonalityEvidence(target="x", direction="positive", strength=0.3).signed_strength() > 0
    assert PersonalityEvidence(target="x", direction="negative", strength=0.3).signed_strength() < 0
    assert PersonalityEvidence(target="x", direction="neutral").signed_strength() == 0.0
