"""Tests for the SQLite infrastructure helpers, especially identifier guards."""

from __future__ import annotations

import sqlite3

import pytest

from webapp.backend.app.infrastructure.db import (
    _ensure_column,
    bootstrap_schema,
)


class TestEnsureColumnIdentifierGuard:
    """``_ensure_column`` must refuse anything that doesn't look like a SQLite
    identifier in the ``table`` / ``column`` positions, and anything that
    doesn't look like a bounded type clause in the ``definition`` position.

    The helper is currently called with hardcoded constants, but its
    signature accepts strings, so a future caller could pipe untrusted
    input through it. The guards are the tripwire that turns that into a
    loud ``ValueError`` instead of an arbitrary ``ALTER TABLE`` execution.
    """

    @pytest.mark.parametrize(
        "bad_table",
        [
            "users; DROP TABLE users",
            "users--",
            "'users'",
            '"users"',
            "users WHERE 1=1",
            "1users",
            "",
            "users users",
        ],
    )
    def test_rejects_malicious_table(self, db_conn: sqlite3.Connection, bad_table: str) -> None:
        with pytest.raises(ValueError, match="invalid table name"):
            _ensure_column(db_conn, bad_table, "col", "TEXT")

    @pytest.mark.parametrize(
        "bad_column",
        [
            "col, password_hash",
            "col); --",
            "col WHERE 1=1",
            "1col",
            "",
        ],
    )
    def test_rejects_malicious_column(self, db_conn: sqlite3.Connection, bad_column: str) -> None:
        with pytest.raises(ValueError, match="invalid column name"):
            _ensure_column(db_conn, "users", bad_column, "TEXT")

    @pytest.mark.parametrize(
        "bad_definition",
        [
            "TEXT; DROP TABLE users",
            "TEXT --",
            "TEXT DEFAULT 'x'",
            "TEXT, password_hash TEXT",
            "",
            "TEXT)",
        ],
    )
    def test_rejects_malicious_definition(
        self, db_conn: sqlite3.Connection, bad_definition: str
    ) -> None:
        with pytest.raises(ValueError, match="invalid column definition"):
            _ensure_column(db_conn, "users", "col", bad_definition)

    def test_accepts_simple_type(self, db_conn: sqlite3.Connection) -> None:
        # Idempotent: re-applying the same migration is a no-op (column
        # already exists), so two calls in a row must both return cleanly.
        _ensure_column(db_conn, "users", "auto_created_at", "TEXT")
        _ensure_column(db_conn, "users", "auto_created_at", "TEXT")

    def test_accepts_definition_with_modifiers(self, db_conn: sqlite3.Connection) -> None:
        _ensure_column(db_conn, "users", "new_int_col", "INTEGER NOT NULL DEFAULT 0")
        rows = db_conn.execute("PRAGMA table_info(users)").fetchall()
        assert any(row["name"] == "new_int_col" for row in rows)


def test_bootstrap_schema_is_idempotent(db_conn: sqlite3.Connection) -> None:
    bootstrap_schema(db_conn)
