"""SQLite 的基础布局；记录追加写入，索引与向量是可重建的派生数据。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ourmem-v5-3"
SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS commits (seq INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY, message_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
    source_order INTEGER NOT NULL UNIQUE, payload TEXT NOT NULL, seq INTEGER NOT NULL,
    UNIQUE(conversation_id, message_id)
);
CREATE TABLE IF NOT EXISTS versions (
    id TEXT PRIMARY KEY, memory_key TEXT NOT NULL, payload TEXT NOT NULL,
    source_order INTEGER NOT NULL, seq INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS versions_by_key ON versions(memory_key, seq);
CREATE TABLE IF NOT EXISTS dependencies (
    id TEXT PRIMARY KEY, target_id TEXT NOT NULL, effect TEXT NOT NULL,
    signature TEXT NOT NULL UNIQUE, payload TEXT NOT NULL,
    source_order INTEGER NOT NULL, seq INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS dependencies_by_target ON dependencies(target_id, seq);
CREATE TABLE IF NOT EXISTS refs (
    owner_type TEXT NOT NULL, owner_id TEXT NOT NULL, ref_type TEXT NOT NULL,
    ref_id TEXT NOT NULL, PRIMARY KEY(owner_type, owner_id, ref_type, ref_id)
);
CREATE INDEX IF NOT EXISTS refs_reverse ON refs(ref_id);
CREATE TABLE IF NOT EXISTS operations (
    id TEXT PRIMARY KEY, target_id TEXT, payload TEXT NOT NULL,
    source_order INTEGER NOT NULL, seq INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS operations_by_target ON operations(target_id, seq);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, commit_seq INTEGER NOT NULL,
    source_cutoff INTEGER NOT NULL, maintenance_incomplete INTEGER NOT NULL,
    created_at TEXT NOT NULL, UNIQUE(commit_seq, source_cutoff)
);
CREATE TABLE IF NOT EXISTS embeddings (
    record_id TEXT NOT NULL, model TEXT NOT NULL, text_hash TEXT NOT NULL,
    vector BLOB NOT NULL, dimensions INTEGER NOT NULL,
    PRIMARY KEY(record_id, model, text_hash)
);
CREATE TABLE IF NOT EXISTS state_cache (
    version_id TEXT PRIMARY KEY, commit_seq INTEGER NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
"""


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def open_database(path: str | Path, namespace: str) -> sqlite3.Connection:
    path = str(path)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA)
    expected = {"schema": SCHEMA_VERSION, "namespace": namespace}
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    for key, value in expected.items():
        if key in existing and existing[key] != value:
            connection.close()
            raise ValueError(f"Database {key} mismatch: {existing[key]!r} != {value!r}")
    with connection:
        connection.executemany(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", expected.items()
        )
    return connection
