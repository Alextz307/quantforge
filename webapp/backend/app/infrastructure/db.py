"""
SQLite connection factory + idempotent schema bootstrap for the webapp store.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from webapp.backend.app.core.settings import get_settings

# SQLite does not support parameter binding for table/column identifiers,
# so migration helpers below must interpolate them. These regexes pin the
# accepted shape so a future caller cannot pass user-supplied input into
# the schema layer: identifiers are restricted to ``^[A-Za-z_][A-Za-z0-9_]*$``
# (no quotes, no semicolons, no whitespace) and type definitions to a small
# alphanumeric + whitespace allowlist that covers ``TEXT``, ``INTEGER NOT
# NULL``, ``INTEGER DEFAULT 0`` and similar — string defaults requiring
# quotes are deliberately not allowed; extending the regex forces review.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TYPE_DEFINITION_RE = re.compile(r"^[A-Za-z0-9_ ]+$")

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    deleted_at TEXT,
    auto_created_at TEXT
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


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """
    Idempotently ``ALTER TABLE ... ADD COLUMN``.

    SQLite supports ``ADD COLUMN`` but not ``ADD COLUMN IF NOT EXISTS``;
    introspect ``PRAGMA table_info`` first so re-running on an already-
    migrated DB is a no-op.

    Identifiers and the type clause are validated against the module-level
    regexes so a future caller that pipes user input into a migration call
    fails loudly here rather than executing an arbitrary statement.
    """

    if not _IDENTIFIER_RE.match(table):
        raise ValueError(f"invalid table name: {table!r}")
    if not _IDENTIFIER_RE.match(column):
        raise ValueError(f"invalid column name: {column!r}")
    if not _TYPE_DEFINITION_RE.match(definition):
        raise ValueError(f"invalid column definition: {definition!r}")
    cursor = conn.execute(f"PRAGMA table_info({table})")  # noqa: S608 - identifier validated above
    existing = {str(row["name"]) for row in cursor.fetchall()}
    if column in existing:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")  # noqa: S608 - identifiers validated above


def bootstrap_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(USERS_SCHEMA)
    conn.executescript(JOBS_SCHEMA)
    conn.executescript(STUDY_SPEC_UPLOADS_SCHEMA)
    conn.executescript(UNIVERSE_SPEC_UPLOADS_SCHEMA)
    # Migration: ``users.auto_created_at`` distinguishes CLI ``--user`` auto-
    # creates from deliberate ``scripts.create_user`` runs so an admin can find
    # typo-stub accounts. Pre-existing rows stay NULL — only new auto-creates
    # stamp it.
    _ensure_column(conn, "users", "auto_created_at", "TEXT")
    conn.commit()


@contextmanager
def open_db() -> Iterator[sqlite3.Connection]:
    conn = get_connection(get_settings().db_path)
    try:
        yield conn
    finally:
        conn.close()
