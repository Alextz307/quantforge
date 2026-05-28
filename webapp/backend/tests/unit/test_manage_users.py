"""
Unit tests for ``scripts.manage_users``.

Covers the cleanup-view filter (auto-created only vs all), the
soft-delete confirmation gating, and the marker round-trip via
``create_user(..., auto_created=True)``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from scripts._attribution import resolve_user_id
from scripts.manage_users import _query_users, delete_cmd
from webapp.backend.app.core.types import Role
from webapp.backend.app.services.user_service import (
    create_user,
    get_user,
    soft_delete_user,
)

_PASSWORD = "alice-password"


def _user(db_conn: sqlite3.Connection, name: str, *, auto: bool, role: Role = Role.USER) -> int:
    user = create_user(
        db_conn, username=name, password=_PASSWORD, role=role, auto_created=auto
    )
    return int(user.id)


def test_create_user_with_auto_created_stamps_marker(
    db_conn: sqlite3.Connection,
) -> None:
    _user(db_conn, "alxe", auto=True)
    row = db_conn.execute(
        "SELECT auto_created_at FROM users WHERE username = ?", ("alxe",)
    ).fetchone()
    assert row is not None
    assert row["auto_created_at"] is not None


def test_create_user_without_flag_leaves_marker_null(
    db_conn: sqlite3.Connection,
) -> None:
    _user(db_conn, "alex", auto=False)
    row = db_conn.execute(
        "SELECT auto_created_at FROM users WHERE username = ?", ("alex",)
    ).fetchone()
    assert row is not None
    assert row["auto_created_at"] is None


def test_query_users_unfiltered_returns_all(db_conn: sqlite3.Connection) -> None:
    _user(db_conn, "alex", auto=False)
    _user(db_conn, "alxe", auto=True)
    rows = _query_users(db_conn, auto_created_only=False)
    assert {r.username for r in rows} == {"alex", "alxe"}


def test_query_users_auto_created_only(db_conn: sqlite3.Connection) -> None:
    _user(db_conn, "alex", auto=False)
    _user(db_conn, "alxe", auto=True)
    rows = _query_users(db_conn, auto_created_only=True)
    assert {r.username for r in rows} == {"alxe"}
    assert rows[0].auto_created_at is not None


def test_query_users_excludes_soft_deleted(db_conn: sqlite3.Connection) -> None:
    alxe_id = _user(db_conn, "alxe", auto=True)
    soft_delete_user(db_conn, alxe_id)
    assert _query_users(db_conn, auto_created_only=True) == []


def test_resolve_user_id_finds_active_user(db_conn: sqlite3.Connection) -> None:
    expected = _user(db_conn, "alex", auto=False)
    assert resolve_user_id(db_conn, "alex") == expected


def test_resolve_user_id_returns_none_for_soft_deleted(
    db_conn: sqlite3.Connection,
) -> None:
    alex_id = _user(db_conn, "alex", auto=False)
    soft_delete_user(db_conn, alex_id)
    assert resolve_user_id(db_conn, "alex") is None


def test_delete_cmd_refuses_non_tty_without_yes(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Scripts/CI calling ``users delete`` MUST pass ``--yes`` — never silently drop.
    """

    _user(db_conn, "alxe", auto=True)
    runner = CliRunner()
    with patch("scripts.manage_users.stdin_is_tty", return_value=False):
        result = runner.invoke(delete_cmd, ["alxe"])
    assert result.exit_code != 0
    assert "non-interactive" in result.output.lower()
    assert get_user(db_conn, resolve_user_id(db_conn, "alxe") or 0) is not None


def test_delete_cmd_soft_deletes_with_yes_flag(
    db_conn: sqlite3.Connection,
) -> None:
    alxe_id = _user(db_conn, "alxe", auto=True)
    runner = CliRunner()
    result = runner.invoke(delete_cmd, ["alxe", "--yes"])
    assert result.exit_code == 0, result.output
    assert "soft-deleted" in result.output
    assert get_user(db_conn, alxe_id) is None


def test_delete_cmd_errors_on_unknown_username(db_conn: sqlite3.Connection) -> None:
    runner = CliRunner()
    result = runner.invoke(delete_cmd, ["ghost", "--yes"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
