"""
Verify scripts/create_user.py inserts/updates rows via Click's CliRunner.
"""

from __future__ import annotations

import sqlite3

from click.testing import CliRunner

from scripts.create_user import main
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.services.auth_service import verify_password

ALEX = "alex"
PASSWORD = "password123"


def _users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM users ORDER BY id"))


def _password_input(password: str) -> str:
    return f"{password}\n{password}\n"


def test_creates_admin_user_with_bcrypt_hash() -> None:
    runner = CliRunner()

    result = runner.invoke(main, [ALEX, "--role", "admin"], input=_password_input(PASSWORD))

    assert result.exit_code == 0, result.output
    db_path = get_settings().db_path
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = _users(conn)
    assert len(rows) == 1
    assert rows[0]["username"] == ALEX
    assert rows[0]["role"] == "admin"
    assert verify_password(PASSWORD, str(rows[0]["password_hash"]))


def test_overwrites_existing_user_password_and_role() -> None:
    runner = CliRunner()
    new_password = "different456"

    runner.invoke(main, [ALEX, "--role", "user"], input=_password_input(PASSWORD))
    result = runner.invoke(main, [ALEX, "--role", "admin"], input=_password_input(new_password))

    assert result.exit_code == 0, result.output
    db_path = get_settings().db_path
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = _users(conn)
    assert len(rows) == 1
    assert rows[0]["role"] == "admin"
    assert verify_password(new_password, str(rows[0]["password_hash"]))
    assert not verify_password(PASSWORD, str(rows[0]["password_hash"]))


def test_short_password_exits_non_zero() -> None:
    runner = CliRunner()

    result = runner.invoke(main, [ALEX], input="short\nshort\n")

    assert result.exit_code == 1
    assert "at least" in result.output
