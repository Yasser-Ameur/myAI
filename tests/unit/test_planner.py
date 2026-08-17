from companion.application.conversation import RuleBasedPlanner
from companion.core.types import ValueEstimate
from companion.domain.conversation import Intent, ResponsePlan, Tone
from companion.domain.state import UserState


def _plan(query: str) -> ResponsePlan:
    return RuleBasedPlanner().plan(query, None, None)


def test_question_intent():
    assert _plan("What is the capital of France?").intent == Intent.QUESTION
    assert _plan("How does memory work?").intent == Intent.QUESTION


def test_greeting_intent():
    assert _plan("hello there").intent == Intent.GREETING
    assert _plan("good morning!").intent == Intent.GREETING


def test_farewell_intent():
    assert _plan("bye now").intent == Intent.FAREWELL


def test_command_intent():
    assert _plan("remember that I love coffee").intent == Intent.COMMAND


def test_chat_default():
    assert _plan("I bought a new keyboard").intent == Intent.CHAT


def test_supportive_tone_when_confused():
    state = UserState()
    state.set("confusion", ValueEstimate(0.7, 0.8, "now", ("test",)))
    plan = RuleBasedPlanner().plan("I don't understand anything about this", state, None)
    assert plan.tone == Tone.SUPPORTIVE


def test_plan_from_dict_tolerates_bad_enum():
    plan = ResponsePlan.from_dict({"intent": "gibberish", "tone": "nonsense"})
    assert plan.intent == Intent.CHAT
    assert plan.tone == Tone.NEUTRAL
    assert plan.warmth == 0.5


def test_plan_round_trip():
    plan = _plan("hello")
    restored = ResponsePlan.from_dict(plan.to_dict())
    assert restored.intent == plan.intent
    assert restored.tone == plan.tone
