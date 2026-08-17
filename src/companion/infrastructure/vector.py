"""Vector store.

The vector backend is swappable (sqlite_bruteforce | sqlite_vec | external)
through configuration. Memory records store `embedding_model_id`, `dimension`
and the vector blob, so a model change never corrupts existing vectors: each
model_id is its own namespace.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from companion.core.errors import ProviderError
from companion.core.types import Vector
from companion.infrastructure.storage import SqliteStorage

log = logging.getLogger(__name__)


def pack_vector(v: Vector) -> bytes:
    return struct.pack(f"<{len(v)}f", *v)


def unpack_vector(blob: bytes) -> Vector:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


@dataclass
class VectorHit:
    id: str
    score: float
    owner_type: str
    owner_id: str


class VectorStore(Protocol):
    def upsert(self, vector_id: str, model_id: str, vector: Vector, owner_type: str, owner_id: str) -> None: ...
    def remove(self, vector_id: str) -> None: ...
    def search(self, model_id: str, query: Vector, top_k: int) -> list[VectorHit]: ...
    def clear_namespace(self, model_id: str) -> None: ...
    def count(self, model_id: str) -> int: ...


class SqliteVectorStore:
    """Brute-force cosine over an in-memory numpy cache backed by SQLite.

    Personal scale (thousands of vectors) makes this fast and dependency-free.
    sqlite-vec or any other backend can replace it behind the same Protocol.
    """

    def __init__(self, storage: SqliteStorage) -> None:
        self._storage = storage
        self._cache: dict[str, tuple[np.ndarray, list[tuple[str, str, str]]]] = {}
        self._dirty = True

    def _load(self, model_id: str) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
        if model_id in self._cache and not self._dirty:
            return self._cache[model_id]
        rows = self._storage.query(
            "SELECT id, vector, owner_type, owner_id FROM embeddings WHERE model_id=?",
            (model_id,),
        )
        if not rows:
            empty = (np.zeros((0, 1), dtype=np.float32), [])
            self._cache[model_id] = empty
            return empty
        ids: list[str] = []
        owners: list[tuple[str, str]] = []
        arrays: list[np.ndarray] = []
        dim = 0
        for r in rows:
            blob = bytes(r["vector"])
            arr = np.frombuffer(blob, dtype=np.float32).astype(np.float32)
            if dim == 0:
                dim = arr.shape[0]
            arrays.append(arr)
            ids.append(str(r["id"]))
            owners.append((str(r["owner_type"]), str(r["owner_id"])))
        mat = np.stack(arrays) if arrays else np.zeros((0, dim), dtype=np.float32)
        self._cache[model_id] = (mat, [(i, o, ow) for i, o, ow in zip(ids, [x[0] for x in owners], [x[1] for x in owners])])
        self._dirty = False
        return self._cache[model_id]

    def _invalidate(self) -> None:
        self._cache.clear()
        self._dirty = True

    def upsert(self, vector_id: str, model_id: str, vector: Vector, owner_type: str, owner_id: str) -> None:
        if not vector:
            raise ProviderError("refusing to store an empty vector")
        self._storage.execute(
            "INSERT INTO embeddings(id, model_id, dimension, vector, owner_type, owner_id) "
            "VALUES(?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "vector=excluded.vector, owner_type=excluded.owner_type, owner_id=excluded.owner_id",
            (vector_id, model_id, len(vector), pack_vector(vector), owner_type, owner_id),
        )
        self._invalidate()

    def remove(self, vector_id: str) -> None:
        self._storage.execute("DELETE FROM embeddings WHERE id=?", (vector_id,))
        self._invalidate()

    def search(self, model_id: str, query: Vector, top_k: int) -> list[VectorHit]:
        mat, metas = self._load(model_id)
        if mat.shape[0] == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.shape[0] != mat.shape[1]:
            raise ProviderError(
                f"embedding dimension mismatch: query={q.shape[0]} store={mat.shape[1]} "
                f"(model_id={model_id})"
            )
        norm = np.linalg.norm(mat, axis=1)
        norm[norm == 0] = 1.0
        qn = q / (np.linalg.norm(q) or 1.0)
        scores = (mat @ qn) / norm
        order = np.argsort(-scores)[:top_k]
        hits: list[VectorHit] = []
        for idx in order:
            vec_id, owner_type, owner_id = metas[int(idx)]
            hits.append(
                VectorHit(
                    id=vec_id,
                    score=float(scores[int(idx)]),
                    owner_type=owner_type,
                    owner_id=owner_id,
                )
            )
        return hits

    def clear_namespace(self, model_id: str) -> None:
        self._storage.execute("DELETE FROM embeddings WHERE model_id=?", (model_id,))
        self._invalidate()

    def count(self, model_id: str) -> int:
        return int(self._storage.scalar("SELECT COUNT(*) FROM embeddings WHERE model_id=?", (model_id,), 0))

    def owner_vector_ids(self, owner_type: str, owner_id: str) -> list[str]:
        rows = self._storage.query(
            "SELECT id FROM embeddings WHERE owner_type=? AND owner_id=?",
            (owner_type, owner_id),
        )
        return [str(r["id"]) for r in rows]


def build_vector_store(backend: str, storage: SqliteStorage) -> VectorStore:
    if backend in ("sqlite_bruteforce", "default"):
        return SqliteVectorStore(storage)
    if backend == "sqlite_vec":
        try:
            from companion.infrastructure.vector_sqlitevec import SqliteVecVectorStore

            return SqliteVecVectorStore(storage)
        except ImportError as exc:  # pragma: no cover
            log.warning("sqlite_vec not installed, falling back to brute-force (%s)", exc)
            return SqliteVectorStore(storage)
    raise ProviderError(f"unknown vector backend: {backend}")
