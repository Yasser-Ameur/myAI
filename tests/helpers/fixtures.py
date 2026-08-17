"""Shared test fixtures."""

from __future__ import annotations

import os
import tempfile

import pytest

from companion.application.memory import MemoryPipeline, MemoryService
from companion.application.personality import PersonalityEngine
from companion.core.clock import Clock, SystemClock
from companion.infrastructure.sqlite_graph import CognitiveGraph
from companion.infrastructure.storage import SqliteStorage


class FakeEmbeddings:
    """Deterministic dependency-free embeddings."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] * 8 for t in texts]


@pytest.fixture
def clock() -> Clock:
    return SystemClock()


@pytest.fixture
def storage() -> SqliteStorage:
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    return SqliteStorage(db)


@pytest.fixture
def graph(storage) -> CognitiveGraph:
    return CognitiveGraph(storage)


@pytest.fixture
def personality(graph, clock) -> PersonalityEngine:
    return PersonalityEngine(graph, clock)


@pytest.fixture
def memory_service(graph, clock) -> MemoryService:
    pipe = MemoryPipeline(
        graph=graph,
        vector_store=None,
        embeddings=FakeEmbeddings(),
        clock=clock,
        embedding_model_id="fake",
    )
    return MemoryService(graph, pipe, clock)
