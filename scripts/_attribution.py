"""
Helpers for the CLI ``--user`` flag: resolve / auto-create webapp users
and stamp synthetic ``jobs`` rows so artifacts the user just launched show
up under their name in the webapp.

The flow is:

1. Resolve the username (CLI flag or ``getpass.getuser()`` default) to a
   webapp ``user_id``.
2. If the user is missing AND stdin is a TTY: confirm the username, prompt
   for a password twice, create the user with ``role=USER``, return the id.
3. If the user is missing AND stdin is NOT a TTY (script / CI): raise so
   automated callers fail loud rather than silently mint a stub.
4. After the artifact lands on disk, the caller invokes
   :func:`attribute_artifact` to insert a synthetic ``jobs`` row keyed by
   the artifact's ``experiment_id``.

The synthetic row mirrors the shape the backfill script writes - same
``status=COMPLETED``, same ``kind`` per artifact type. This keeps the
ownership lookup (``SELECT user_id FROM jobs WHERE experiment_id = ?``)
uniform regardless of whether attribution happened at launch time or via
the backfill.
"""

from __future__ import annotations

import getpass
import sqlite3
import sys
import uuid
from datetime import UTC, datetime
from typing import TextIO

import click

from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.db import bootstrap_schema, open_db
from webapp.backend.app.schemas.jobs import JobKind, JobStatus
from webapp.backend.app.services.auth_service import MIN_PASSWORD_LENGTH
from webapp.backend.app.services.user_service import create_user


class UserNotFoundNonInteractiveError(click.ClickException):
    """
    Raised when ``--user`` names a missing user and stdin can't prompt.

    The error message includes a pointer at ``scripts.create_user`` so the
    operator can pre-create the account out of band.
    """


def resolve_user_id(conn: sqlite3.Connection, username: str) -> int | None:
    """
    Return the active user_id for ``username``, or ``None`` if missing/deleted.
    """

    row = conn.execute(
        "SELECT id FROM users WHERE username = ? AND deleted_at IS NULL",
        (username,),
    ).fetchone()
    return int(row["id"]) if row is not None else None


def stdin_is_tty(stream: TextIO | None = None) -> bool:
    """
    ``True`` iff ``stream`` (default ``sys.stdin``) is attached to a TTY.
    """

    target = stream if stream is not None else sys.stdin
    return bool(getattr(target, "isatty", lambda: False)())


def insert_synthetic_job(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    kind: JobKind,
    experiment_id: str,
    command: str,
    timestamp_iso: str,
) -> None:
    """
    Insert one synthetic ``jobs`` row. Caller is responsible for ``commit()``.
    """

    conn.execute(
        "INSERT INTO jobs ("
        "id, user_id, kind, command, config_path, log_path, "
        "status, started_at, finished_at, experiment_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            uuid.uuid4().hex,
            user_id,
            kind.value,
            command,
            "",
            "",
            JobStatus.COMPLETED.value,
            timestamp_iso,
            timestamp_iso,
            experiment_id,
        ),
    )


def _prompt_and_create(conn: sqlite3.Connection, username: str) -> int:
    """
    Interactive auto-create: confirm + password prompt + insert.

    Raises ``click.ClickException`` on confirm decline, mismatched
    passwords, or too-short passwords.
    """

    click.echo(f"webapp user '{username}' not found.")
    if not click.confirm(f"Create new webapp user '{username}'?", default=False):
        raise click.ClickException(f"webapp user '{username}' does not exist and was not created")
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm:  ")
    if password != confirm:
        raise click.ClickException("passwords do not match")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise click.ClickException(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    user = create_user(
        conn,
        username=username,
        password=password,
        role=Role.USER,
        auto_created=True,
    )
    click.echo(f"created webapp user '{user.username}' (role={user.role.value})")
    return int(user.id)


def resolve_or_create_attributing_user(conn: sqlite3.Connection, username: str) -> int:
    """
    Look up the user id for ``username``; auto-create on TTY only.

    Returns the user_id ready to be stamped onto a synthetic ``jobs`` row.
    Raises :class:`UserNotFoundNonInteractiveError` when the user is
    missing and stdin can't drive a password prompt (CI, cron, piped
    invocations).
    """

    existing = resolve_user_id(conn, username)
    if existing is not None:
        return existing
    if not stdin_is_tty():
        raise UserNotFoundNonInteractiveError(
            f"webapp user '{username}' not found and stdin is not a TTY. "
            f"Pre-create it via `python -m scripts.create_user {username}` "
            f"or run interactively."
        )
    return _prompt_and_create(conn, username)


def attribute_artifact(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    kind: JobKind,
    experiment_id: str,
    command: str = "cli",
) -> None:
    """
    Insert a synthetic ``jobs`` row tying ``experiment_id`` to ``user_id``.

    Idempotent: a re-run that produces the same ``experiment_id`` (e.g.
    deterministic compare/holdout out_name) leaves the existing owner in
    place rather than duplicating the row.
    """

    existing = conn.execute(
        "SELECT 1 FROM jobs WHERE experiment_id = ? LIMIT 1",
        (experiment_id,),
    ).fetchone()
    if existing is not None:
        return
    insert_synthetic_job(
        conn,
        user_id=user_id,
        kind=kind,
        experiment_id=experiment_id,
        command=command,
        timestamp_iso=datetime.now(UTC).isoformat(),
    )
    conn.commit()


def attribute_via_username(
    *,
    username: str,
    kind: JobKind,
    experiment_id: str,
    command: str = "cli",
) -> None:
    """
    Convenience wrapper: open the webapp DB, resolve the user, stamp the row.

    Subcommands call this once at the end of execution. Logs but does not
    re-raise on sqlite errors - attribution must never block the
    artifact-producing path.

    Silently skips when the webapp DB doesn't exist yet: the CLI must not
    lazy-create ``webapp/data/webapp.sqlite`` for users who haven't bootstrapped
    the webapp, and pytest fixtures use a missing path to opt out.
    """

    if not get_settings().db_path.exists():
        return
    try:
        with open_db() as conn:
            bootstrap_schema(conn)
            user_id = resolve_or_create_attributing_user(conn, username)
            attribute_artifact(
                conn,
                user_id=user_id,
                kind=kind,
                experiment_id=experiment_id,
                command=command,
            )
    except sqlite3.Error as exc:
        click.echo(f"warning: failed to attribute artifact to '{username}': {exc}", err=True)


def default_username() -> str:
    """
    The OS username - the default value for ``--user`` flags.
    """

    return getpass.getuser()
