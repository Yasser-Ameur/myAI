"""SQLite storage primitives: connection management, WAL, schema migrations.

The companion keeps ONE local database. All cognitive state (graph, memories,
vectors, personalities, relationships, agent memory) lives there. Vector search
and graph traversal share the same DB file.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from companion.core.errors import MemoryUnavailableError

log = logging.getLogger(__name__)


def _default_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _parse_json(text: str | bytes | None, default: Any) -> Any:
    if text is None:
        return default
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default


class SqliteStorage:
    """Thread-safe SQLite storage with WAL and versioned schema migrations."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._open()

    def _open(self) -> None:
        try:
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, exist_ok=True)
            # isolation_level=None puts the driver in autocommit mode. This is
            # deliberate and load-bearing: with the default ('') the driver
            # opens an implicit transaction before every INSERT/UPDATE/DELETE
            # and close() discards it, so every cognitive write would be lost
            # at process exit. Multi-statement atomicity is expressed
            # explicitly through transaction() (BEGIN IMMEDIATE ... COMMIT).
            self._conn = sqlite3.connect(self.path, check_same_thread=False,
                                         timeout=30, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error as exc:  # pragma: no cover - depends on platform
            raise MemoryUnavailableError(f"cannot open cognitive db at {self.path}: {exc}") from exc

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise MemoryUnavailableError("database is not open")
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            try:
                return self._conn.execute(sql, params)
            except sqlite3.Error as exc:
                raise MemoryUnavailableError(str(exc)) from exc

    def executemany(self, sql: str, seq: list[tuple]) -> sqlite3.Cursor:
        with self._lock:
            try:
                return self._conn.executemany(sql, seq)
            except sqlite3.Error as exc:
                raise MemoryUnavailableError(str(exc)) from exc

    def executescript(self, script: str) -> None:
        with self._lock:
            try:
                self._conn.executescript(script)
            except sqlite3.Error as exc:
                raise MemoryUnavailableError(str(exc)) from exc

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        cur = self.execute(sql, params)
        return cur.fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: tuple = (), default: Any = None) -> Any:
        row = self.query_one(sql, params)
        return row[0] if row is not None else default

    def last_rowid(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            if self._conn.in_transaction:
                # nested transaction -> savepoint so an inner rollback does not
                # clobber the outer transaction
                savepoint = f"sp_{id(self)}"
                self._conn.execute(f"SAVEPOINT {savepoint}")
                try:
                    yield
                except Exception:
                    self._conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    raise
                else:
                    self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                return
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            if self._conn is None:
                return
            try:
                if self._conn.in_transaction:
                    # An explicit transaction that outlived its owner: the
                    # writer neither committed nor rolled back. Roll back
                    # rather than commit a half-applied unit of work, and say
                    # so — silence here is what hid the persistence bug.
                    log.warning("closing storage with an open transaction; rolling back")
                    self._conn.rollback()
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as exc:
                log.warning("error while closing storage: %s", exc)
            finally:
                self._conn.close()
                self._conn = None

    def ensure_schema(self, schema_sql: str, version: int, current_version_key: str = "schema_version") -> None:
        """Create tables if missing; apply schema version bookkeeping."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
            )
            has_meta = cur.fetchone() is not None
            if not has_meta:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
                )
            existing = self._conn.execute(
                "SELECT value FROM meta WHERE key=?", (current_version_key,)
            ).fetchone()
            installed = int(existing[0]) if existing else 0
            if installed < version:
                self._conn.executescript(schema_sql)
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    (current_version_key, str(version)),
                )
                self._conn.commit()
                log.info("cognitive schema at version %s", version)


def encode_json(obj: Any) -> str:
    return _default_json(obj)


def decode_json(text: Any, default: Any = None) -> Any:
    return _parse_json(text, default)
