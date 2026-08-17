import types

from companion.application.reflection import IdleDetector
from companion.core.clock import FakeClock
from companion.runtime.orchestration import CompanionApp


class _FakeLifecycle:
    def __init__(self, loaded):
        self._loaded = list(loaded)

    def stats(self):
        return {"loaded": list(self._loaded)}

    def unload(self, slot):
        if slot in self._loaded:
            self._loaded.remove(slot)


def _comp(clock: FakeClock, loaded: list[str], last_activity: float):
    lifecycle = _FakeLifecycle(loaded)
    idle = types.SimpleNamespace(last_activity=last_activity)
    return types.SimpleNamespace(
        app=types.SimpleNamespace(clock=clock),
        reflection=types.SimpleNamespace(idle=idle),
        lifecycle=lifecycle,
    ), lifecycle


def test_idle_unload_recent_activity_keeps_models():
    clock = FakeClock()
    comp, lifecycle = _comp(clock, ["llm.default", "llm.fast", "vad.default"],
                            last_activity=clock.monotonic())
    clock.advance(60)  # only 60s since activity, threshold is 300s
    assert CompanionApp._idle_unload_once(comp, 300) == []
    assert lifecycle.stats()["loaded"] == ["llm.default", "llm.fast", "vad.default"]


def test_idle_unload_frees_heavy_models_only():
    clock = FakeClock()
    comp, lifecycle = _comp(
        clock,
        ["llm.default", "llm.fast", "stt.default", "tts.default", "vad.default"],
        last_activity=clock.monotonic(),
    )
    clock.advance(600)  # way past the 300s idle threshold
    victims = CompanionApp._idle_unload_once(comp, 300)
    assert sorted(victims) == ["llm.default", "llm.fast", "stt.default"]
    assert lifecycle.stats()["loaded"] == ["tts.default", "vad.default"]


def test_idle_unload_disabled_when_threshold_zero():
    clock = FakeClock()
    comp, lifecycle = _comp(clock, ["llm.default"], last_activity=0.0)
    clock.advance(9999)
    assert CompanionApp._idle_unload_once(comp, 0) == []
    assert lifecycle.stats()["loaded"] == ["llm.default"]


def test_idle_detector_exposes_last_activity():
    clock = FakeClock(start=500.0)
    det = IdleDetector(clock=clock)
    clock.advance(42)
    det.register_activity()
    assert det.last_activity == 542.0
    clock.advance(10)
    assert det.last_activity == 542.0  # stable until next register_activity
