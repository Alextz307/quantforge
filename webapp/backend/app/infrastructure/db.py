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

JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    kind TEXT NOT NULL,
    command TEXT NOT NULL,
    config_path TEXT NOT NULL,
    log_path TEXT NOT NULL,
    pid INTEGER,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    exit_code INTEGER,
    experiment_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_experiment_id ON jobs(experiment_id)
    WHERE experiment_id IS NOT NULL;
"""

STUDY_SPEC_UPLOADS_SCHEMA = """
CREATE TABLE IF NOT EXISTS study_spec_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    slug TEXT NOT NULL,
    yaml_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_study_spec_uploads_active_slug
    ON study_spec_uploads(user_id, slug) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_study_spec_uploads_user_id
    ON study_spec_uploads(user_id);
"""

UNIVERSE_SPEC_UPLOADS_SCHEMA = """
CREATE TABLE IF NOT EXISTS universe_spec_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    slug TEXT NOT NULL,
    yaml_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_universe_spec_uploads_active_slug
    ON universe_spec_uploads(user_id, slug) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_universe_spec_uploads_user_id
    ON universe_spec_uploads(user_id);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI's threadpool can resolve a `with`-yielding
    # dependency on one worker and invoke the consuming endpoint on another.
    # Each request still owns its connection (open_db is a per-call context
    # manager) so two threads never touch the same connection concurrently.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def bootstrap_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(USERS_SCHEMA)
    conn.executescript(JOBS_SCHEMA)
    conn.executescript(STUDY_SPEC_UPLOADS_SCHEMA)
    conn.executescript(UNIVERSE_SPEC_UPLOADS_SCHEMA)
    conn.commit()


@contextmanager
def open_db() -> Iterator[sqlite3.Connection]:
    conn = get_connection(get_settings().db_path)
    try:
        yield conn
    finally:
        conn.close()
