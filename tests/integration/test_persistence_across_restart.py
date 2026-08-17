"""Durability tests that a same-process test cannot fake.

The persistence bug this suite exists to prevent was invisible to every
existing test: SQLite writes were never committed, but an *open connection*
sees its own uncommitted data, so in-process assertions all passed while the
database was in fact empty at exit.

The only honest check is to close the connection — or leave the process
entirely — and look again. `test_writes_survive_a_new_connection` is the
minimal regression guard; the subprocess tests exercise the real runtime.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile

import pytest

from companion.application.facts import FactWriter
from companion.application.identity import SelfModelService
from companion.application.salience import TurnCommitter
from companion.core.clock import SystemClock
from companion.infrastructure.sqlite_graph import CognitiveGraph
from companion.infrastructure.storage import SqliteStorage


@pytest.fixture
def db_path() -> str:
    return os.path.join(tempfile.mkdtemp(), "durability.db")


def _reopen(path: str) -> CognitiveGraph:
    return CognitiveGraph(SqliteStorage(path))


def test_writes_survive_a_new_connection(db_path):
    """A write must be visible to a *different* connection after close()."""
    storage = SqliteStorage(db_path)
    graph = CognitiveGraph(storage)
    clock = SystemClock()
    writer = FactWriter(graph, clock)
    user = writer.entity_id("primary_user_entity", name="user", type="person")
    writer.assert_fact(subject_id=user, predicate="favorite:color", value="purple",
                       confidence=0.95, provenance="explicit_user_statement")
    storage.close()

    # Fresh connection: this is what a restart actually sees.
    with sqlite3.connect(db_path) as raw:
        rows = raw.execute(
            "SELECT predicate, value FROM facts WHERE predicate='favorite:color'"
        ).fetchall()
    assert rows == [("favorite:color", "purple")], (
        "facts did not survive connection close — writes are not being committed"
    )


def test_turn_commit_is_durable_without_graceful_shutdown(db_path):
    """Facts land at turn time, not only at episode close."""
    storage = SqliteStorage(db_path)
    graph = CognitiveGraph(storage)
    clock = SystemClock()
    self_model = SelfModelService(graph, clock, configured_name="Companion")
    self_model.load()
    committer = TurnCommitter(graph, self_model, clock=clock)

    committer.commit("You are Jarvis.", episode_id="ep-1")
    committer.commit("My favorite color is purple.", episode_id="ep-1")
    # Simulate a kill: no close_episode(), no consolidation, just gone.
    storage.close()

    graph2 = _reopen(db_path)
    self_model2 = SelfModelService(graph2, clock, configured_name="Companion")
    model = self_model2.load()
    assert model.identity.name == "Jarvis"
    assert model.name_source == "persisted"

    writer2 = FactWriter(graph2, clock)
    user = writer2.entity_id("primary_user_entity", name="user", type="person")
    current = writer2.current(user, "favorite:color")
    assert current is not None and current.value == "purple"


def test_correction_supersedes_and_keeps_history(db_path):
    storage = SqliteStorage(db_path)
    graph = CognitiveGraph(storage)
    clock = SystemClock()
    self_model = SelfModelService(graph, clock, configured_name="Companion")
    self_model.load()
    committer = TurnCommitter(graph, self_model, clock=clock)
    committer.commit("My favorite color is purple.", episode_id="ep-1")
    storage.close()

    # A *new process worth* of state: the referent must be recovered from the
    # graph, not from in-memory conversation context.
    graph2 = _reopen(db_path)
    self_model2 = SelfModelService(graph2, clock, configured_name="Companion")
    self_model2.load()
    committer2 = TurnCommitter(graph2, self_model2, clock=clock)
    committer2.commit("I don't like purple anymore. It's blue now.", episode_id="ep-2")

    writer = FactWriter(graph2, clock)
    user = writer.entity_id("primary_user_entity", name="user", type="person")
    assert writer.current(user, "favorite:color").value == "blue"
    previous = writer.previous(user, "favorite:color")
    assert previous is not None and previous.value == "purple"
    assert previous.valid_to, "superseded fact must be closed, not deleted"


def test_rename_keeps_previous_name_queryable(db_path):
    storage = SqliteStorage(db_path)
    graph = CognitiveGraph(storage)
    clock = SystemClock()
    self_model = SelfModelService(graph, clock, configured_name="Companion")
    self_model.load()
    self_model.set_name("Jarvis")
    self_model.set_name("Friday")
    storage.close()

    graph2 = _reopen(db_path)
    reloaded = SelfModelService(graph2, clock, configured_name="Companion")
    model = reloaded.load()
    assert model.identity.name == "Friday"
    assert reloaded.previous_name() == "Jarvis"
    history = reloaded.name_history()
    assert [h["name"] for h in history if h["current"]] == ["Friday"]
    assert "Jarvis" in [h["name"] for h in history]


def test_hedged_guess_does_not_overwrite_stated_identity(db_path):
    """'I think your name might be Bob' must not rename a stated identity."""
    graph = CognitiveGraph(SqliteStorage(db_path))
    clock = SystemClock()
    self_model = SelfModelService(graph, clock, configured_name="Companion")
    self_model.load()
    committer = TurnCommitter(graph, self_model, clock=clock)

    committer.commit("Your name is Jarvis.", episode_id="ep-1")
    assert self_model.name == "Jarvis"

    result = committer.commit("I think your name might be Bob.", episode_id="ep-1")
    assert self_model.name == "Jarvis", "a hedged guess overwrote a stated identity"
    assert result.identity is not None and result.identity["changed"] is False


# ---------------------------------------------------------------------------
# Real subprocess restarts: separate interpreters, separate connections.
# ---------------------------------------------------------------------------

_RUNNER = """
import asyncio, json, os, sys
from companion.runtime.config import Config
from companion.runtime.orchestration import CompanionApp

async def main():
    app = CompanionApp(Config.load())
    comp = app.build()
    out = {"degraded": comp.degraded, "name_at_boot": comp.self_model.name,
           "name_source": comp.self_model.model().name_source, "turns": []}
    await app.start()
    for line in json.loads(sys.argv[1]):
        r = await app.respond(line)
        out["turns"].append({"in": line, "out": r["text"], "skill": r.get("skill", "")})
    await app.aclose()
    print("@@RESULT@@" + json.dumps(out))

asyncio.run(main())
"""


def _run_session(db: str, lines: list[str], timeout: int = 600) -> dict:
    env = dict(os.environ)
    env["COMPANION_MEMORY_DATABASE"] = db
    env["COMPANION_LOGGING_LEVEL"] = "error"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, json.dumps(lines)],
        capture_output=True, text=True, env=env, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    if "@@RESULT@@" not in proc.stdout:
        pytest.fail(f"session failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout.split("@@RESULT@@", 1)[1].strip())


requires_models = pytest.mark.skipif(
    os.environ.get("COMPANION_E2E") != "1",
    reason="set COMPANION_E2E=1 to run real-model subprocess restart tests",
)


@requires_models
def test_identity_survives_a_real_process_restart(tmp_path):
    db = str(tmp_path / "e2e.db")
    first = _run_session(db, ["You are Jarvis.", "My favorite color is purple."])
    assert not first["degraded"], first["degraded"]

    second = _run_session(db, ["What's your name?", "What's my favorite color?"])
    assert second["name_at_boot"] == "Jarvis"
    assert second["name_source"] == "persisted"
    assert "jarvis" in second["turns"][0]["out"].lower()
    assert "purple" in second["turns"][1]["out"].lower()


@requires_models
def test_correction_and_history_survive_a_real_process_restart(tmp_path):
    db = str(tmp_path / "e2e2.db")
    _run_session(db, ["My favorite color is purple."])
    _run_session(db, ["I don't like purple anymore. It's blue now."])
    third = _run_session(db, ["What's my favorite color?",
                              "What used to be my favorite color?"])
    assert "blue" in third["turns"][0]["out"].lower()
    assert "purple" in third["turns"][1]["out"].lower()
