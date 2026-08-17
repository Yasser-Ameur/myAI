import asyncio
import types
import warnings

import pytest

from companion.core.events import EVENT_MEMORY_COMMITTED, EventBus
from companion.interfaces import cli
from companion.runtime.config import Config
from companion.runtime.orchestration import CompanionApp


class _Lifecycle:
    def load(self, _slot):
        return None


class _Registry:
    def has(self, _slot):
        return False


class _RunApp:
    def __init__(self):
        self.bus = EventBus()
        self.hardware = types.SimpleNamespace(profile_name="test")
        self.components = types.SimpleNamespace(
            degraded=None, lifecycle=_Lifecycle(), registry=_Registry()
        )
        self.started_on = None
        self.closed = False

    async def start(self):
        self.started_on = asyncio.get_running_loop()

        async def handler(_event):
            return None

        self.bus.subscribe(EVENT_MEMORY_COMMITTED, handler)

    async def aclose(self):
        await self.bus.aclose()
        self.closed = True


def test_run_cli_owns_loop_and_closes_subscriptions(monkeypatch):
    app = _RunApp()
    monkeypatch.setattr(cli, "_app", lambda _args: app)
    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert cli.cmd_run(types.SimpleNamespace()) == 0

    assert app.started_on is not None
    assert app.closed
    assert not [w for w in caught if "was never awaited" in str(w.message)]


@pytest.mark.asyncio
async def test_subscription_shutdown_awaits_cancelled_handler():
    bus = EventBus()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(_event):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    bus.subscribe(EVENT_MEMORY_COMMITTED, handler)
    bus.publish(EVENT_MEMORY_COMMITTED)
    await entered.wait()
    await bus.aclose()

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_runtime_start_is_supported_inside_running_loop(tmp_path):
    config = Config.from_dict({"memory": {"database": str(tmp_path / "memory.db")}})
    app = CompanionApp(config)
    app.build()
    await app.start()

    assert app.bus.subscriber_count("ResponsePlanCreated") == 1

    await app.aclose()
    assert app.bus.subscriber_count("ResponsePlanCreated") == 0


def test_runtime_build_does_not_create_subscriptions_before_a_loop(tmp_path):
    config = Config.from_dict({"memory": {"database": str(tmp_path / "memory.db")}})
    app = CompanionApp(config)
    app.build()

    assert app.bus.subscriber_count("ResponsePlanCreated") == 0

    app.shutdown()
