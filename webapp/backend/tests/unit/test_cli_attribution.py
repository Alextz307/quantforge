"""Unit tests for ``scripts._attribution``.

Covers user resolution (hit, miss-with-TTY, miss-without-TTY),
auto-create with the password prompt, idempotent attribution, and the
non-interactive error path.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import click
import pytest

from scripts._attribution import (
    UserNotFoundNonInteractiveError,
    attribute_artifact,
    resolve_or_create_attributing_user,
)
from webapp.backend.app.core.types import Role
from webapp.backend.app.schemas.jobs import JobKind
from webapp.backend.app.services.auth_service import MIN_PASSWORD_LENGTH
from webapp.backend.app.services.user_service import create_user

_PASSWORD = "alex-password"
_EID = "20260101_120000_strat_sha_hash"


def _seed(conn: sqlite3.Connection, username: str, role: Role = Role.USER) -> int:
    user = create_user(conn, username=username, password=_PASSWORD, role=role)
    return int(user.id)


def test_resolve_returns_existing_user(db_conn: sqlite3.Connection) -> None:
    expected = _seed(db_conn, "alex")
    assert resolve_or_create_attributing_user(db_conn, "alex") == expected


def test_resolve_errors_on_non_tty_for_missing_user(
    db_conn: sqlite3.Connection,
) -> None:
    """Without a TTY we must NOT prompt — raise pointing at scripts/create_user."""
    with patch("scripts._attribution.stdin_is_tty", return_value=False):
        with pytest.raises(UserNotFoundNonInteractiveError, match="not a TTY"):
            resolve_or_create_attributing_user(db_conn, "ghost")


def test_resolve_auto_creates_user_when_tty(
    db_conn: sqlite3.Connection,
) -> None:
    """TTY path: confirm + 2-pass password + create with role=USER."""
    fresh_password = "newpass!secret"
    with (
        patch("scripts._attribution.stdin_is_tty", return_value=True),
        patch("scripts._attribution.click.confirm", return_value=True),
        patch(
            "scripts._attribution.getpass.getpass",
            side_effect=[fresh_password, fresh_password],
        ),
    ):
        user_id = resolve_or_create_attributing_user(db_conn, "newcomer")
    row = db_conn.execute(
        "SELECT id, role FROM users WHERE username = ?", ("newcomer",)
    ).fetchone()
    assert row is not None
    assert int(row["id"]) == user_id
    assert str(row["role"]) == Role.USER.value


def test_resolve_aborts_when_confirm_declined(db_conn: sqlite3.Connection) -> None:
    with (
        patch("scripts._attribution.stdin_is_tty", return_value=True),
        patch("scripts._attribution.click.confirm", return_value=False),
    ):
        with pytest.raises(click.ClickException, match="does not exist"):
            resolve_or_create_attributing_user(db_conn, "newcomer")


def test_resolve_rejects_password_mismatch(db_conn: sqlite3.Connection) -> None:
    with (
        patch("scripts._attribution.stdin_is_tty", return_value=True),
        patch("scripts._attribution.click.confirm", return_value=True),
        patch(
            "scripts._attribution.getpass.getpass",
            side_effect=["password-A", "password-B"],
        ),
    ):
        with pytest.raises(click.ClickException, match="do not match"):
            resolve_or_create_attributing_user(db_conn, "newcomer")


def test_resolve_rejects_short_password(db_conn: sqlite3.Connection) -> None:
    short = "x" * (MIN_PASSWORD_LENGTH - 1)
    with (
        patch("scripts._attribution.stdin_is_tty", return_value=True),
        patch("scripts._attribution.click.confirm", return_value=True),
        patch("scripts._attribution.getpass.getpass", side_effect=[short, short]),
    ):
        with pytest.raises(click.ClickException, match="at least"):
            resolve_or_create_attributing_user(db_conn, "newcomer")


def test_attribute_artifact_inserts_jobs_row(db_conn: sqlite3.Connection) -> None:
    user_id = _seed(db_conn, "alex")
    attribute_artifact(
        db_conn, user_id=user_id, kind=JobKind.RUN, experiment_id=_EID, command="experiment run"
    )
    row = db_conn.execute(
        "SELECT user_id, kind, command, experiment_id FROM jobs WHERE experiment_id = ?",
        (_EID,),
    ).fetchone()
    assert row is not None
    assert int(row["user_id"]) == user_id
    assert str(row["kind"]) == JobKind.RUN.value
    assert str(row["command"]) == "experiment run"


def test_attribute_artifact_is_idempotent(db_conn: sqlite3.Connection) -> None:
    """A second call with the same experiment_id does not overwrite or duplicate."""
    alex_id = _seed(db_conn, "alex")
    bob_id = _seed(db_conn, "bob")

    attribute_artifact(
        db_conn, user_id=alex_id, kind=JobKind.RUN, experiment_id=_EID
    )
    attribute_artifact(
        db_conn, user_id=bob_id, kind=JobKind.RUN, experiment_id=_EID
    )

    rows = db_conn.execute(
        "SELECT user_id FROM jobs WHERE experiment_id = ?", (_EID,)
    ).fetchall()
    assert len(rows) == 1
    assert int(rows[0]["user_id"]) == alex_id
