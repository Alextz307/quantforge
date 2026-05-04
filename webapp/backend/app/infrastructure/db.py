"""SQLite connection factory + idempotent schema bootstrap for the webapp store."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from webapp.backend.app.core.settings import get_settings

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    deleted_at TEXT
);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def bootstrap_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(USERS_SCHEMA)
    conn.commit()


@contextmanager
def open_db() -> Iterator[sqlite3.Connection]:
    conn = get_connection(get_settings().db_path)
    try:
        yield conn
    finally:
        conn.close()
