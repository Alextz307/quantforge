"""Admin CLI for inspecting and cleaning up webapp accounts.

Subcommands:

* ``users list [--auto-created]``    Print active users, optionally
                                     filtered to those auto-created by
                                     the CLI ``--user`` flag (typo
                                     accumulation cleanup).
* ``users delete <username>``        Soft-delete a user. Refuses without
                                     ``--yes`` outside a TTY.

The ``auto_created_at`` column on ``users`` is the marker — set only by
:func:`scripts._attribution.resolve_or_create_attributing_user`, never by
``scripts.create_user`` or any webapp flow. Pre-existing rows from before
the column landed stay ``NULL`` (no false positives in the cleanup view).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import click

from scripts._attribution import resolve_user_id, stdin_is_tty
from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.db import bootstrap_schema, open_db
from webapp.backend.app.services.user_service import soft_delete_user


@dataclass(frozen=True)
class _UserRow:
    id: int
    username: str
    role: Role
    created_at: str
    auto_created_at: str | None


def _query_users(conn: sqlite3.Connection, *, auto_created_only: bool) -> list[_UserRow]:
    sql = (
        "SELECT id, username, role, created_at, auto_created_at FROM users WHERE deleted_at IS NULL"
    )
    if auto_created_only:
        sql += " AND auto_created_at IS NOT NULL"
    sql += " ORDER BY id"
    rows = conn.execute(sql).fetchall()
    return [
        _UserRow(
            id=int(row["id"]),
            username=str(row["username"]),
            role=Role(str(row["role"])),
            created_at=str(row["created_at"]),
            auto_created_at=(
                str(row["auto_created_at"]) if row["auto_created_at"] is not None else None
            ),
        )
        for row in rows
    ]


def _format_row(row: _UserRow) -> str:
    auto = row.auto_created_at or "—"
    return f"  {row.id:>4}  {row.username:<24}  {row.role.value:<6}  {row.created_at:<32}  {auto}"


@click.group("users")
def users() -> None:
    """Inspect and clean up webapp accounts."""


@users.command("list")
@click.option(
    "--auto-created",
    is_flag=True,
    default=False,
    help=(
        "Only list accounts created by the CLI ``--user`` flag. Useful "
        "for finding typo-stub accounts that should be soft-deleted."
    ),
)
def list_cmd(auto_created: bool) -> None:
    """List active webapp users, optionally filtered to CLI auto-creates."""
    with open_db() as conn:
        bootstrap_schema(conn)
        rows = _query_users(conn, auto_created_only=auto_created)
    if not rows:
        click.echo("no users match the filter")
        return
    header = f"  {'id':>4}  {'username':<24}  {'role':<6}  {'created_at':<32}  auto_created_at"
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))
    for row in rows:
        click.echo(_format_row(row))


@users.command("delete")
@click.argument("username")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the interactive confirmation prompt (required in non-TTY contexts).",
)
def delete_cmd(username: str, yes: bool) -> None:
    """Soft-delete a webapp user by username.

    The user can no longer log in and disappears from list views, but
    artifacts they own (via ``jobs.user_id``) keep their attribution
    intact — the username on those artifacts will simply no longer
    resolve to a current account in subsequent ``resolve_owner_usernames``
    lookups, which the frontend renders as the ``"system"`` fallback.
    """
    with open_db() as conn:
        bootstrap_schema(conn)
        user_id = resolve_user_id(conn, username)
        if user_id is None:
            raise click.ClickException(f"user '{username}' not found (or already deleted)")
        if not yes:
            if not stdin_is_tty():
                raise click.ClickException(
                    f"refusing to delete '{username}' without --yes in a non-interactive context"
                )
            if not click.confirm(f"Soft-delete user '{username}' (id={user_id})?", default=False):
                raise click.ClickException("aborted")
        if not soft_delete_user(conn, user_id):
            raise click.ClickException(f"failed to soft-delete '{username}'")
    click.echo(f"soft-deleted user '{username}'")


if __name__ == "__main__":
    users()
