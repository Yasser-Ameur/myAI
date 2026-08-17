"""Export and import the cognitive state.

The user owns their memory, so it has to be extractable in a form that is
readable without this program and restorable into a fresh install. Model
weights are never included: they are large, replaceable, and not part of who
the companion is.

The export is plain JSON, one object per table, with a schema version so a
future import can migrate it.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

EXPORT_VERSION = 1

# Ordered so that imports satisfy references: entities and sources first.
_SECTIONS = (
    "entities",
    "sources",
    "facts",
    "relations",
    "episodes",
    "memories",
    "goals",
    "relationships",
    "beliefs",
    "observations",
    "knowledge",
    "system_state",
)


def export_state(graph, path: str) -> dict:
    """Write the full cognitive state to a directory as JSON."""
    storage = _storage_of(graph)
    if storage is None:
        raise RuntimeError("cognitive store is unavailable; nothing to export")
    os.makedirs(path, exist_ok=True)
    counts: dict[str, int] = {}
    for table in _SECTIONS:
        rows = _dump_table(storage, table)
        counts[table] = len(rows)
        with open(os.path.join(path, f"{table}.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2, default=str)
    manifest = {
        "export_version": EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
        "contains_model_weights": False,
        "note": "Cognitive state only. Model weights are not included by design.",
    }
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    log.info("exported cognitive state to %s (%d facts, %d memories)",
             path, counts.get("facts", 0), counts.get("memories", 0))
    return manifest


def import_state(graph, path: str, replace: bool = False) -> dict:
    """Restore an exported state.

    Rows are inserted with INSERT OR REPLACE keyed on the primary key, so
    importing the same export twice is idempotent rather than duplicating the
    user's history.
    """
    storage = _storage_of(graph)
    if storage is None:
        raise RuntimeError("cognitive store is unavailable; cannot import")
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"no manifest.json in {path}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    version = int(manifest.get("export_version", 0))
    if version > EXPORT_VERSION:
        raise ValueError(
            f"export was written by a newer version ({version} > {EXPORT_VERSION})"
        )

    counts: dict[str, int] = {}
    with storage.transaction():
        if replace:
            for table in reversed(_SECTIONS):
                try:
                    storage.execute(f"DELETE FROM {table}")
                except Exception as exc:
                    log.warning("could not clear %s: %s", table, exc)
        for table in _SECTIONS:
            file_path = os.path.join(path, f"{table}.json")
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as fh:
                rows = json.load(fh)
            counts[table] = _load_table(storage, table, rows)
    log.info("imported cognitive state from %s: %s", path, counts)
    return {"imported": counts, "source_manifest": manifest, "replace": replace}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _storage_of(graph):
    return getattr(graph, "_storage", None)


def _table_exists(storage, table: str) -> bool:
    row = storage.query_one(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return row is not None


def _dump_table(storage, table: str) -> list[dict]:
    if not _table_exists(storage, table):
        return []
    return [dict(r) for r in storage.query(f"SELECT * FROM {table}")]


def _load_table(storage, table: str, rows: list[dict]) -> int:
    if not rows or not _table_exists(storage, table):
        return 0
    columns = [r[1] for r in storage.query(f"PRAGMA table_info({table})")]
    usable = [c for c in columns if c in rows[0]]
    if not usable:
        return 0
    placeholders = ",".join("?" for _ in usable)
    sql = (f"INSERT OR REPLACE INTO {table}({','.join(usable)}) "
           f"VALUES({placeholders})")
    payload = [tuple(row.get(c) for c in usable) for row in rows]
    storage.executemany(sql, payload)
    return len(payload)
